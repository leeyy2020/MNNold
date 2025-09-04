// Minimal attention grad check: compares analytic vs finite-difference
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <cstdlib>

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "MNN_generated.h"


using namespace MNN;
using namespace MNN::Express;

// Reference forward: y = softmax(mScale * QK^T) @ V; returns loss = sum(y)
static float computeLossRef(const float* q, const float* k, const float* v,
                            int seq_len, int kv_seq_len, int num_head, int kv_num_head, int head_dim) {
    const float mScale = 1.0f / std::sqrt((float)head_dim);
    const int group = num_head / kv_num_head;
    float loss = 0.0f;
    std::vector<float> logits(kv_seq_len);
    std::vector<float> probs(kv_seq_len);
    for (int h = 0; h < num_head; ++h) {
        int kv_h = h / group;
        for (int i = 0; i < seq_len; ++i) {
            // logits
            float rowMax = -1e30f;
            for (int j = 0; j < kv_seq_len; ++j) {
                float dot = 0.f;
                for (int d = 0; d < head_dim; ++d) {
                    size_t qidx = (size_t)0 * seq_len * num_head * head_dim + (size_t)i * num_head * head_dim + (size_t)h * head_dim + d;
                    size_t kidx = (size_t)0 * kv_seq_len * kv_num_head * head_dim + (size_t)j * kv_num_head * head_dim + (size_t)kv_h * head_dim + d;
                    dot += q[qidx] * k[kidx];
                }
                logits[j] = mScale * dot;
                rowMax = std::max(rowMax, logits[j]);
            }
            // softmax
            float sumExp = 0.f;
            for (int j = 0; j < kv_seq_len; ++j) {
                probs[j] = std::exp(logits[j] - rowMax);
                sumExp += probs[j];
            }
            for (int j = 0; j < kv_seq_len; ++j) probs[j] /= sumExp;
            // y += sum_j p * V
            for (int d = 0; d < head_dim; ++d) {
                float yd = 0.f;
                for (int j = 0; j < kv_seq_len; ++j) {
                    size_t vidx = (size_t)0 * kv_seq_len * kv_num_head * head_dim + (size_t)j * kv_num_head * head_dim + (size_t)kv_h * head_dim + d;
                    yd += probs[j] * v[vidx];
                }
                loss += yd;
            }
        }
    }
    return loss;
}

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

struct CaseCfg { int seq_len, kv_seq_len, num_head, kv_num_head, head_dim; };

int main(int argc, char** argv) {

    std::string dumpDir = "";
    std::string compareDir = "";
    for (int ai = 1; ai < argc; ++ai) {
        std::string a = argv[ai];
        if (a == "--dump" && ai + 1 < argc) dumpDir = argv[++ai];
        else if (a == "--compare" && ai + 1 < argc) compareDir = argv[++ai];
    }

    std::vector<CaseCfg> cases = {
        {2,2,2,2,4},
        {3,3,4,2,8},
        {4,5,8,4,16},
        {1,7,2,1,32}
    };

    float globalMaxQ=0, globalMaxK=0, globalMaxV=0; int passCnt=0;
    for (int ci = 0; ci < (int)cases.size(); ++ci) {
        const int seq_len = cases[ci].seq_len;
        const int kv_seq_len = cases[ci].kv_seq_len;
        const int num_head = cases[ci].num_head;
        const int kv_num_head = cases[ci].kv_num_head;
        const int head_dim = cases[ci].head_dim;

    // Build inputs as trainable tensors
    // Shapes follow Attention op convention: [batch, seq_len, num_head, head_dim]
        auto q = _Input({1, seq_len, num_head, head_dim}, NHWC, halide_type_of<float>());
        auto k = _Input({1, kv_seq_len, kv_num_head, head_dim}, NHWC, halide_type_of<float>());
        auto v = _Input({1, kv_seq_len, kv_num_head, head_dim}, NHWC, halide_type_of<float>());

    // Initialize with small deterministic values
    {
        auto qp = q->writeMap<float>();
        auto kp = k->writeMap<float>();
        auto vp = v->writeMap<float>();
        for (int i = 0; i < seq_len * num_head * head_dim; ++i) qp[i] = 0.1f * std::sin(0.3f * (i+13*ci)) + 0.01f * ((i+7*ci) % 3);
        for (int i = 0; i < kv_seq_len * kv_num_head * head_dim; ++i) kp[i] = 0.07f * std::cos(0.2f * (i+11*ci)) - 0.02f * ((i+5*ci) % 5);
        for (int i = 0; i < kv_seq_len * kv_num_head * head_dim; ++i) vp[i] = 0.05f * std::sin(0.5f * (i+3*ci)) + 0.03f * ((i+2*ci) % 7);
        }

    // Compute reference loss once
        auto qrp = q->readMap<float>();
        auto krp = k->readMap<float>();
        auto vrp = v->readMap<float>();
        float initialLoss = computeLossRef(qrp, krp, vrp, seq_len, kv_seq_len, num_head, kv_num_head, head_dim);
        printf("[case %d] Initial loss (ref): %f\n", ci, initialLoss);

    // Analytic grads via backend AttentionGrad with dO = ones
        auto dO = _Const(1.0f, {1, seq_len, num_head, head_dim});
    // Manually build AttentionGrad op to get all three outputs
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_AttentionGrad;
    op->main.type = OpParameter_NONE;
    op->main.value = nullptr;
    std::vector<VARP> agInputs = {q, k, v, dO};
    auto agExpr = Expr::create(std::move(op), agInputs, 3);
        auto gQ = Variable::create(agExpr, 0);
        auto gK = Variable::create(agExpr, 1);
        auto gV = Variable::create(agExpr, 2);
        Variable::compute({gQ, gK, gV});
        auto gQp = gQ->readMap<float>();
        auto gKp = gK->readMap<float>();
        auto gVp = gV->readMap<float>();

    // Finite difference gradients
    const float eps = 1e-3f;
    float maxErrQ = 0.f, maxErrK = 0.f, maxErrV = 0.f;

    // Baseline loss (ref)
        float baseLoss = computeLossRef(qrp, krp, vrp, seq_len, kv_seq_len, num_head, kv_num_head, head_dim);

    // Check dQ
    {
        auto qp = q->writeMap<float>();
        for (int i = 0; i < seq_len * num_head * head_dim; ++i) {
            float old = qp[i];
            qp[i] = old + eps;
            float loss1 = computeLossRef(qp, krp, vrp, seq_len, kv_seq_len, num_head, kv_num_head, head_dim);
            qp[i] = old - eps;
            float loss2 = computeLossRef(qp, krp, vrp, seq_len, kv_seq_len, num_head, kv_num_head, head_dim);
            qp[i] = old;
            float numGrad = (loss1 - loss2) / (2 * eps);
            maxErrQ = std::max(maxErrQ, std::abs(numGrad - gQp[i]));
        }
    }
    // Check dK
    {
        auto kp = k->writeMap<float>();
        for (int i = 0; i < kv_seq_len * kv_num_head * head_dim; ++i) {
            float old = kp[i];
            kp[i] = old + eps;
            float loss1 = computeLossRef(qrp, kp, vrp, seq_len, kv_seq_len, num_head, kv_num_head, head_dim);
            kp[i] = old - eps;
            float loss2 = computeLossRef(qrp, kp, vrp, seq_len, kv_seq_len, num_head, kv_num_head, head_dim);
            kp[i] = old;
            float numGrad = (loss1 - loss2) / (2 * eps);
            maxErrK = std::max(maxErrK, std::abs(numGrad - gKp[i]));
        }
    }
    // Check dV
    {
        auto vp = v->writeMap<float>();
        for (int i = 0; i < kv_seq_len * kv_num_head * head_dim; ++i) {
            float old = vp[i];
            vp[i] = old + eps;
            float loss1 = computeLossRef(qrp, krp, vp, seq_len, kv_seq_len, num_head, kv_num_head, head_dim);
            vp[i] = old - eps;
            float loss2 = computeLossRef(qrp, krp, vp, seq_len, kv_seq_len, num_head, kv_num_head, head_dim);
            vp[i] = old;
            float numGrad = (loss1 - loss2) / (2 * eps);
            maxErrV = std::max(maxErrV, std::abs(numGrad - gVp[i]));
        }
    }

        printf("[case %d] AttentionGrad numeric check:\n", ci);
        printf("  max |dQ_num - dQ_ana| = %.6e\n", maxErrQ);
        printf("  max |dK_num - dK_ana| = %.6e\n", maxErrK);
        printf("  max |dV_num - dV_ana| = %.6e\n", maxErrV);
        bool ok = maxErrQ < 5e-3 && maxErrK < 5e-3 && maxErrV < 5e-3;
        printf("  Result: %s\n", ok ? "PASS" : "FAIL");
        if (ok) passCnt++;
        globalMaxQ = std::max(globalMaxQ, maxErrQ);
        globalMaxK = std::max(globalMaxK, maxErrK);
        globalMaxV = std::max(globalMaxV, maxErrV);

        // Optional dump for PyTorch comparison
        if (!dumpDir.empty()) {
            // ensure directory exists
            std::string mk = std::string("mkdir -p ") + dumpDir;
            std::system(mk.c_str());
            char buf[256];
            snprintf(buf, sizeof(buf), "%s/case_%d_", dumpDir.c_str(), ci);
            std::string prefix(buf);
            size_t nq = (size_t)seq_len * num_head * head_dim;
            size_t nkv = (size_t)kv_seq_len * kv_num_head * head_dim;
            writeTxt(prefix + "q.txt", qrp, nq);
            writeTxt(prefix + "k.txt", krp, nkv);
            writeTxt(prefix + "v.txt", vrp, nkv);
            std::vector<float> ones((size_t)seq_len * num_head * head_dim, 1.0f);
            writeTxt(prefix + "do.txt", ones.data(), ones.size());
            // also dump our grads
            writeTxt(prefix + "qg_mnn.txt", gQp, nq);
            writeTxt(prefix + "kg_mnn.txt", gKp, nkv);
            writeTxt(prefix + "vg_mnn.txt", gVp, nkv);
            // meta
            std::ofstream meta(prefix + "meta.txt");
            meta << seq_len << " " << kv_seq_len << " " << num_head << " " << kv_num_head << " " << head_dim << "\n";
        }

        // Optional compare with torch output if present
        if (!compareDir.empty()) {
            char buf[256];
            snprintf(buf, sizeof(buf), "%s/case_%d_", compareDir.c_str(), ci);
            std::string prefix(buf);
            std::vector<float> qg_t, kg_t, vg_t;
            if (readTxt(prefix + "qg_torch.txt", qg_t) && readTxt(prefix + "kg_torch.txt", kg_t) && readTxt(prefix + "vg_torch.txt", vg_t)) {
                size_t nq = (size_t)seq_len * num_head * head_dim;
                size_t nkv = (size_t)kv_seq_len * kv_num_head * head_dim;
                float maxEq=0, maxEk=0, maxEv=0;
                for (size_t i = 0; i < nq; ++i) maxEq = std::max(maxEq, std::abs(qg_t[i] - gQp[i]));
                for (size_t i = 0; i < nkv; ++i) maxEk = std::max(maxEk, std::abs(kg_t[i] - gKp[i]));
                for (size_t i = 0; i < nkv; ++i) maxEv = std::max(maxEv, std::abs(vg_t[i] - gVp[i]));
                printf("  Torch compare: max |dQ|=%.3e |dK|=%.3e |dV|=%.3e\n", maxEq, maxEk, maxEv);
            } else {
                printf("  Torch compare: grads not found under %s\n", compareDir.c_str());
            }
        }
    }

    printf("Summary: %d/%d cases pass. Global max errs Q=%.3e K=%.3e V=%.3e\n", passCnt, (int)cases.size(), globalMaxQ, globalMaxK, globalMaxV);
    if (!dumpDir.empty()) {
        printf("To generate PyTorch grads, run:\n  python3 tools/scripts/attention_grad_torch.py %s\nThen re-run:\n  ./build/attention_grad_test.out --compare %s\n", dumpDir.c_str(), dumpDir.c_str());
    }
    return passCnt == (int)cases.size() ? 0 : 1;
}