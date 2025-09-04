//
//  CPUAttentionGrad.cpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <limits>
#include <cmath>
#include "CPUAttentionGrad.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"

#if defined (__aarch64__)
#define FLOAT16_T __fp16
#else
#define FLOAT16_T float
#endif

namespace MNN {

CPUAttentionGrad::CPUAttentionGrad(Backend *backend) : Execution(backend) {
}

CPUAttentionGrad::~CPUAttentionGrad() {
}

ErrorCode CPUAttentionGrad::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend *>(backend())->functions();
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mThreadNum = ((CPUBackend *)backend())->threadNumber();
    unit  = core->pack;
    bytes = core->bytes;
    return NO_ERROR;
}

ErrorCode CPUAttentionGrad::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // inputs: [query, key, value, output_grad, mask(optional), sinks(optional)]
    // outputs: [query_grad, key_grad, value_grad]
    // Debug print helper: tensor shape and f32 summary (printed once)
    
    auto printTensorShape = [](const Tensor* t) {
        if (!t) { MNN_PRINT("null"); return; }
        int dims = t->buffer().dimensions;
        MNN_PRINT("[");
        for (int i = 0; i < dims; ++i) {
            MNN_PRINT("%d%s", t->buffer().dim[i].extent, (i+1<dims)?", ":"");
        }
        MNN_PRINT("]");
    };
    auto printF32Summary = [&](const Tensor* t, const char* name) {
        if (!t) { MNN_PRINT("  %s: <null>\n", name); return; }
        if (t->getType() != halide_type_of<float>()) {
            MNN_PRINT("  %s: dtype=%d (not f32), skip\n", name, t->getType().code);
            return;
        }
        size_t n = t->elementSize();
        if (n == 0) { MNN_PRINT("  %s: empty\n", name); return; }
        const float* p = t->host<float>();
        float mn = p[0], mx = p[0], sum = 0.f;
        size_t k = n < 16 ? n : 16;
        for (size_t i = 0; i < n; ++i) {
            float v = p[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        float mean = sum / (float)n;
        MNN_PRINT("  %s: shape=", name);
        printTensorShape(t);
        MNN_PRINT(", n=%zu, min=%.6f, max=%.6f, mean=%.6f, head=", n, mn, mx, mean);
        for (size_t i = 0; i < k; ++i) {
            MNN_PRINT("%s%.6f", (i==0?"":" "), p[i]);
        }
        if (k < n) MNN_PRINT(" ...");
        MNN_PRINT("\n");
    };
    static bool sPrintedOnce = false;
    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto output_grad = inputs[3];
    const Tensor* mask = nullptr;
    const Tensor* sinks = nullptr;

    // Shapes: query/key/value are [batch, seq_len, num_head, head_dim]
    int seq_len = query->length(1);
    int kv_seq_len = key->length(1);
    mNumHead = query->length(2);
    mHeadDim = query->length(3);
    mKvNumHead = key->length(2);

    if (inputs.size() > 4) {
        mask = inputs[4];
    }
    if (inputs.size() > 5) {
        sinks = inputs[5];
    }

    auto query_grad = outputs[0];
    auto key_grad = outputs[1];
    auto value_grad = outputs[2];

    // Initialize gradients to zero
    ::memset(query_grad->host<char>(), 0, query_grad->size());
    ::memset(key_grad->host<char>(), 0, key_grad->size());
    ::memset(value_grad->host<char>(), 0, value_grad->size());

    // Scaling as in forward: effective scale is 1/sqrt(head_dim)
    const float mScale = 1.0f / sqrtf((float)mHeadDim);

    // Parallelize over kv-head groups to avoid write conflicts on K/V grads
    int group_size = mNumHead / mKvNumHead;
    int tileCount = UP_DIV(mKvNumHead, mThreadNum);

    // Treat tensors as float32 for gradient math (inputs are float in forward)
    auto q_fp32 = query->host<float>();
    auto k_fp32 = key->host<float>();
    auto v_fp32 = value->host<float>();
    auto og_fp32 = output_grad->host<float>();

    auto qg_fp32 = query_grad->host<float>();
    auto kg_fp32 = key_grad->host<float>();
    auto vg_fp32 = value_grad->host<float>();

    float* sinksPtr = nullptr;
    if (sinks) {
        sinksPtr = const_cast<float*>(sinks->host<float>());
    }

    // Helper lambdas for load/store with dtype handling
    auto loadQ = [&](int i, int h, int d) -> float {
        size_t idx = (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim + d;
        return q_fp32[idx];
    };
    auto loadK = [&](int j, int kvh, int d) -> float {
        size_t idx = (size_t)j * mKvNumHead * mHeadDim + (size_t)kvh * mHeadDim + d;
        return k_fp32[idx];
    };
    auto loadV = [&](int j, int kvh, int d) -> float {
        size_t idx = (size_t)j * mKvNumHead * mHeadDim + (size_t)kvh * mHeadDim + d;
        return v_fp32[idx];
    };
    auto loadOG = [&](int i, int h, int d) -> float {
        size_t idx = (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim + d;
        return og_fp32[idx];
    };
    auto addQG = [&](int i, int h, int d, float v) {
        size_t idx = (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim + d;
        qg_fp32[idx] += v;
    };
    auto addKG = [&](int j, int kvh, int d, float v) {
        size_t idx = (size_t)j * mKvNumHead * mHeadDim + (size_t)kvh * mHeadDim + d;
        kg_fp32[idx] += v;
    };
    auto addVG = [&](int j, int kvh, int d, float v) {
        size_t idx = (size_t)j * mKvNumHead * mHeadDim + (size_t)kvh * mHeadDim + d;
        vg_fp32[idx] += v;
    };

    // Mask helpers
    const float NEG_INF = -1e30f;
    const bool hasMask = (mask != nullptr);
    const bool maskIsFloat = hasMask && (mask->getType() == halide_type_of<float>());
    const bool maskIsInt = hasMask && !maskIsFloat;
    const float* maskF = maskIsFloat ? mask->host<float>() : nullptr;
    const int* maskI = maskIsInt ? mask->host<int>() : nullptr;

    std::function<void(int)> mCompute = [&](int tId) {
        int kv_start = tId * tileCount;
        int kv_end = ALIMIN(kv_start + tileCount, mKvNumHead);
        std::vector<float> logits(kv_seq_len + 1);
        std::vector<float> probs(kv_seq_len);
        std::vector<float> dP(kv_seq_len);
        for (int kv_h = kv_start; kv_h < kv_end; ++kv_h) {
            for (int g = 0; g < group_size; ++g) {
                int h = kv_h * group_size + g;
                if (h >= mNumHead) break;
                // For each token i, compute softmax row, then backprop
                for (int i = 0; i < seq_len; ++i) {
                    // 1) logits = mScale * Q[i,h,:] · K[j,kv_h,:] + mask
                    for (int j = 0; j < kv_seq_len; ++j) {
                        float dot = 0.f;
                        // dot(Q_i_h, K_j_kvh)
                        for (int d = 0; d < mHeadDim; ++d) {
                            dot += loadQ(i, h, d) * loadK(j, kv_h, d);
                        }
                        float val = mScale * dot;
                        if (hasMask) {
                            if (maskIsFloat) {
                                if (mask->elementSize() == seq_len * kv_seq_len) {
                                    val += maskF[i * kv_seq_len + j];
                                } else {
                                    // square mask for generation token
                                    int offset = kv_seq_len - seq_len;
                                    if (j >= offset) {
                                        val += maskF[i * seq_len + (j - offset)];
                                    }
                                }
                            } else { // int mask
                                int m = maskI[i * kv_seq_len + j];
                                if (!m) {
                                    val = NEG_INF;
                                }
                            }
                        }
                        logits[j] = val;
                    }
                    // 2) softmax over kv_seq_len (+ sink if provided)
                    float rowMax = -std::numeric_limits<float>::infinity();
                    for (int j = 0; j < kv_seq_len; ++j) rowMax = ALIMAX(rowMax, logits[j]);
                    bool useSink = (sinksPtr != nullptr);
                    float sinkVal = 0.f;
                    if (useSink) {
                        sinkVal = sinksPtr[h];
                        rowMax = ALIMAX(rowMax, sinkVal);
                    }
                    // exp and sum
                    float sumExp = 0.f;
                    for (int j = 0; j < kv_seq_len; ++j) {
                        float e = expf(logits[j] - rowMax);
                        probs[j] = e; // store temporary as exp unnormalized
                        sumExp += e;
                    }
                    if (useSink) {
                        sumExp += expf(sinkVal - rowMax);
                    }
                    for (int j = 0; j < kv_seq_len; ++j) {
                        probs[j] = probs[j] / sumExp;
                    }
                    // 3) dV += P^T @ dO_row; dP = dO_row @ V^T
                    // dO_row
                    // dP and dV
                    for (int j = 0; j < kv_seq_len; ++j) {
                        float dp = 0.f;
                        float p = probs[j];
                        if (p == 0.f) {
                            dP[j] = 0.f;
                        } else {
                            for (int d = 0; d < mHeadDim; ++d) {
                                float dO = loadOG(i, h, d);
                                float v = loadV(j, kv_h, d);
                                // accumulate dV
                                addVG(j, kv_h, d, p * dO);
                                dp += dO * v;
                            }
                            dP[j] = dp;
                        }
                    }
                    // 4) softmax backward: dS = (dP - (dP·P_total)) ⊙ P
                    float sum_dP_P = 0.f;
                    for (int j = 0; j < kv_seq_len; ++j) {
                        sum_dP_P += dP[j] * probs[j];
                    }
                    // 5) dQ and dK from dS
                    for (int j = 0; j < kv_seq_len; ++j) {
                        float dS = (dP[j] - sum_dP_P) * probs[j];
                        float coeff = mScale * dS;
                        // dQ[i,h,:] += coeff * K[j,kv_h,:]
                        // dK[j,kv_h,:] += coeff * Q[i,h,:]
                        if (coeff != 0.f) {
                            for (int d = 0; d < mHeadDim; ++d) {
                                float kval = loadK(j, kv_h, d);
                                float qval = loadQ(i, h, d);
                                addQG(i, h, d, coeff * kval);
                                addKG(j, kv_h, d, coeff * qval);
                            }
                        }
                    }
                } // i
            } // g
        } // kv_h
    };

    MNN_CONCURRENCY_BEGIN(tId, mThreadNum) {
        mCompute((int)tId);
    }
    MNN_CONCURRENCY_END();


    return NO_ERROR;
}

bool CPUAttentionGrad::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new CPUAttentionGrad(bn);
    *dst = tmp;
    return true;
}

class CPUAttentionGradCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUAttentionGrad(backend);
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUAttentionGradCreator, OpType_AttentionGrad);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE