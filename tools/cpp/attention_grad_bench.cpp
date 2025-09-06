// Benchmark for AttentionGrad (CPU, fp32)
// Measures runtime for large shapes and optionally checks precision
// against PyTorch by reading reference grads dumped via tools/scripts/attention_grad_torch.py

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "MNN_generated.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

static void writeTxt(const std::string& path, const float* data, size_t n) {
    std::ofstream ofs(path);
    for (size_t i = 0; i < n; ++i) {
        ofs << data[i] << (i + 1 == n ? '\n' : ' ');
    }
}

static bool readTxt(const std::string& path, std::vector<float>& out) {
    std::ifstream ifs(path);
    if (!ifs.good()) return false;
    out.clear();
    float v;
    while (ifs >> v) out.push_back(v);
    return true;
}

struct Args {
    int seq = 4096;
    int kv_seq = 4096;
    int num_head = 8;
    int kv_num_head = -1; // default = num_head
    int head_dim = 40;
    int iters = 5;
    int warmup = 2;
    unsigned seed = 42;
    std::string dumpDir;    // if set, dump q/k/v/do for torch
    std::string compareDir; // if set, read torch grads and compare
};

static void usage() {
    printf("Usage: attention_grad_bench.out [options]\n");
    printf("  --seq N             default 4096\n");
    printf("  --kv-seq N          default 4096\n");
    printf("  --num-head N        default 8\n");
    printf("  --kv-num-head N     default = num-head\n");
    printf("  --head-dim N        default 40\n");
    printf("  --iters N           default 5\n");
    printf("  --warmup N          default 2\n");
    printf("  --seed N            default 42\n");
    printf("  --dump DIR          dump inputs for torch script\n");
    printf("  --compare DIR       compare with torch grads from DIR\n");
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int& out) -> bool { if (i + 1 >= argc) return false; out = std::atoi(argv[++i]); return true; };
        if (a == "--seq") { if (!need(args.seq)) return false; }
        else if (a == "--kv-seq") { if (!need(args.kv_seq)) return false; }
        else if (a == "--num-head") { if (!need(args.num_head)) return false; }
        else if (a == "--kv-num-head") { if (!need(args.kv_num_head)) return false; }
        else if (a == "--head-dim") { if (!need(args.head_dim)) return false; }
        else if (a == "--iters") { if (!need(args.iters)) return false; }
        else if (a == "--warmup") { if (!need(args.warmup)) return false; }
        else if (a == "--seed") { unsigned s; if (!need(*(int*)&s)) return false; args.seed = s; }
        else if (a == "--dump") { if (i + 1 >= argc) return false; args.dumpDir = argv[++i]; }
        else if (a == "--compare") { if (i + 1 >= argc) return false; args.compareDir = argv[++i]; }
        else if (a == "-h" || a == "--help") { return false; }
        else { printf("Unknown arg: %s\n", a.c_str()); return false; }
    }
    if (args.kv_num_head < 0) args.kv_num_head = args.num_head;
    return true;
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) { usage(); return 1; }

    printf("AttentionGrad bench: seq=%d kv_seq=%d num_head=%d kv_num_head=%d head_dim=%d\n",
           args.seq, args.kv_seq, args.num_head, args.kv_num_head, args.head_dim);
    printf("  iters=%d warmup=%d seed=%u\n", args.iters, args.warmup, args.seed);

    // Create inputs
    auto q = _Input({1, args.seq, args.num_head, args.head_dim}, NHWC, halide_type_of<float>());
    auto k = _Input({1, args.kv_seq, args.kv_num_head, args.head_dim}, NHWC, halide_type_of<float>());
    auto v = _Input({1, args.kv_seq, args.kv_num_head, args.head_dim}, NHWC, halide_type_of<float>());
    auto dO = _Input({1, args.seq, args.num_head, args.head_dim}, NHWC, halide_type_of<float>());

    // Initialize deterministic pseudo-random values
    std::mt19937 rng(args.seed);
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    {
        auto qp = q->writeMap<float>();
        auto kp = k->writeMap<float>();
        auto vp = v->writeMap<float>();
        auto dp = dO->writeMap<float>();
        size_t nq = (size_t)args.seq * args.num_head * args.head_dim;
        size_t nkv = (size_t)args.kv_seq * args.kv_num_head * args.head_dim;
        for (size_t i = 0; i < nq; ++i) qp[i] = dist(rng);
        for (size_t i = 0; i < nkv; ++i) kp[i] = dist(rng);
        for (size_t i = 0; i < nkv; ++i) vp[i] = dist(rng);
        for (size_t i = 0; i < nq; ++i) dp[i] = 1.0f; // ones dO by default
    }

    // Build AttentionGrad op: outputs dQ, dK, dV
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_AttentionGrad;
    op->main.type = OpParameter_NONE;
    op->main.value = nullptr;
    std::vector<VARP> agInputs = {q, k, v, dO};
    auto agExpr = Expr::create(std::move(op), agInputs, 3);
    auto gQ = Variable::create(agExpr, 0);
    auto gK = Variable::create(agExpr, 1);
    auto gV = Variable::create(agExpr, 2);

    // Warmup
    for (int i = 0; i < args.warmup; ++i) {
        Variable::compute({gQ, gK, gV});
    }

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < args.iters; ++i) {
        // Mutate one input slightly to avoid cache hit
        {
            auto dp = dO->writeMap<float>();
            dp[0] = dp[0] + 1e-7f;
        }
        Variable::compute({gQ, gK, gV});
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg = ms / std::max(1, args.iters);
    printf("Timing: total %.2f ms over %d iters, avg = %.2f ms/iter\n", ms, args.iters, avg);

    // Optional: dump inputs and our grads for torch comparison
    if (!args.dumpDir.empty()) {
        std::string mk = std::string("mkdir -p ") + args.dumpDir;
        std::system(mk.c_str());
        char buf[256];
        snprintf(buf, sizeof(buf), "%s/case_%d_", args.dumpDir.c_str(), 0);
        std::string prefix(buf);
        size_t nq = (size_t)args.seq * args.num_head * args.head_dim;
        size_t nkv = (size_t)args.kv_seq * args.kv_num_head * args.head_dim;
        auto qrp = q->readMap<float>();
        auto krp = k->readMap<float>();
        auto vrp = v->readMap<float>();
        auto drp = dO->readMap<float>();
        auto gQp = gQ->readMap<float>();
        auto gKp = gK->readMap<float>();
        auto gVp = gV->readMap<float>();
        writeTxt(prefix + "meta.txt", (const float*)&args.seq, 0); // write separately below
        // meta
        {
            std::ofstream meta(prefix + "meta.txt");
            meta << args.seq << " " << args.kv_seq << " " << args.num_head << " " << args.kv_num_head << " " << args.head_dim << "\n";
        }
        writeTxt(prefix + "q.txt", qrp, nq);
        writeTxt(prefix + "k.txt", krp, nkv);
        writeTxt(prefix + "v.txt", vrp, nkv);
        writeTxt(prefix + "do.txt", drp, nq);
        writeTxt(prefix + "qg_mnn.txt", gQp, nq);
        writeTxt(prefix + "kg_mnn.txt", gKp, nkv);
        writeTxt(prefix + "vg_mnn.txt", gVp, nkv);
        printf("Dumped inputs and MNN grads to %s\n", args.dumpDir.c_str());
        printf("To generate PyTorch grads: python3 tools/scripts/attention_grad_torch.py %s\n", args.dumpDir.c_str());
    }

    // Optional: compare to torch grads
    if (!args.compareDir.empty()) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s/case_%d_", args.compareDir.c_str(), 0);
        std::string prefix(buf);
        std::vector<float> qg_t, kg_t, vg_t;
        bool ok = readTxt(prefix + "qg_torch.txt", qg_t) && readTxt(prefix + "kg_torch.txt", kg_t) && readTxt(prefix + "vg_torch.txt", vg_t);
        if (!ok) {
            printf("Compare: Torch grads not found under %s\n", args.compareDir.c_str());
        } else {
            auto gQp = gQ->readMap<float>();
            auto gKp = gK->readMap<float>();
            auto gVp = gV->readMap<float>();
            size_t nq = (size_t)args.seq * args.num_head * args.head_dim;
            size_t nkv = (size_t)args.kv_seq * args.kv_num_head * args.head_dim;
            auto maxAbs = [](const float* a, const float* b, size_t n){ double m=0; for(size_t i=0;i<n;++i) m = std::max(m, (double)std::abs(a[i]-b[i])); return m; };
            double qErr = maxAbs(gQp, qg_t.data(), nq);
            double kErr = maxAbs(gKp, kg_t.data(), nkv);
            double vErr = maxAbs(gVp, vg_t.data(), nkv);
            printf("Compare Torch: max |dQ|=%.3e |dK|=%.3e |dV|=%.3e\n", qErr, kErr, vErr);
        }
    }

    return 0;
}
