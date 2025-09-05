//
//  CPUAttentionGrad.cpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <MNN/AutoTime.hpp>
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

// Pack query helper function adapted from CPUAttention.cpp
static void pack_query_for_grad(const float* query_src, float* pack_q, int seq_len, int head_dim, int eP) {
    for (int i = 0; i < seq_len; i++) {
        int out_index = i / eP;
        int in_index = i % eP;
        for (int j = 0; j < head_dim; j++) {
            pack_q[out_index * head_dim * eP + j * eP + in_index] = query_src[i * head_dim + j];
        }
    }
}

// Pack key for matmul (transpose=true for K^T)
static void pack_key_for_grad(const float* key_src, float* pack_k, int kv_seq_len, int head_dim, int hP, Backend* backend) {
    auto core = static_cast<CPUBackend*>(backend)->functions();
    core->MNNPackForMatMul_B(pack_k, key_src, (size_t)head_dim, (size_t)kv_seq_len, true);
}

CPUAttentionGrad::CPUAttentionGrad(Backend *backend) : Execution(backend) {
}

CPUAttentionGrad::~CPUAttentionGrad() {
}

ErrorCode CPUAttentionGrad::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
auto core = static_cast<CPUBackend*>(backend())->functions();
        core->MNNGetMatMulPackMode(&mEP, &mLP, &mHP);
        mBytes = core->bytes;
        if (mBytes != 4) {
            // 仅支持 float32
            return NOT_SUPPORT;
        }
        // 只需要保存基本尺寸，不再大规模分配中间矩阵
        auto Q = inputs[0];
        auto K = inputs[1];
        mSeq     = Q->length(1);
        mNumHead = Q->length(2);
        mHeadDim = Q->length(3);
        mKvSeq   = K->length(1);
        mKvNumHead = K->length(2);
        mBatch = Q->length(0);
        mThreads = static_cast<CPUBackend*>(backend())->threadNumber();
        
        // Allocate pack buffers for optimization
        mPackQ.reset(Tensor::createDevice<float>({mThreads, UP_DIV(mSeq, mEP), mHeadDim, mEP}));
        mPackK.reset(Tensor::createDevice<float>({mThreads, UP_DIV(mKvSeq, mHP), mHeadDim, mHP}));
        backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mPackK.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mPackK.get(), Backend::DYNAMIC);
        
        return NO_ERROR;
}



ErrorCode CPUAttentionGrad::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    AUTOTIME;
const Tensor* Q = inputs[0];
        const Tensor* K = inputs[1];
        const Tensor* V = inputs[2];
        const Tensor* mask = nullptr;
        const Tensor* dY   = nullptr;
        if (inputs.size() == 5) {
            mask = inputs[3];
            dY   = inputs[4];
        } else { // 4
            dY = inputs[3];
        }
        Tensor* dQ = outputs[0];
        Tensor* dK = outputs[1];
        Tensor* dV = outputs[2];

        // 清零输出
        ::memset(dQ->host<void>(), 0, dQ->size());
        ::memset(dK->host<void>(), 0, dK->size());
        ::memset(dV->host<void>(), 0, dV->size());

        const float* Qptr = Q->host<float>();
        const float* Kptr = K->host<float>();
        const float* Vptr = V->host<float>();
        const float* dYptr = dY->host<float>();

        float* dQptr = dQ->host<float>();
        float* dKptr = dK->host<float>();
        float* dVptr = dV->host<float>();

        const bool hasMask = (mask != nullptr);
        const bool maskIsFloat = hasMask && (mask->getType() == halide_type_of<float>());
        const float* maskF = (hasMask && maskIsFloat) ? mask->host<float>() : nullptr;
        const int*   maskI = (hasMask && !maskIsFloat) ? mask->host<int>() : nullptr;

        const float scale = 1.0f / std::sqrt((float)mHeadDim);

        // 每线程处理一段 head
        const int tilePerThread = UP_DIV(mNumHead, mThreads);

        // 线程本地缓冲：dK / dV 大小 [KV_S, KV_H, D]
        std::vector<std::vector<float>> dK_locals(mThreads);
        std::vector<std::vector<float>> dV_locals(mThreads);
        for (int t = 0; t < mThreads; ++t) {
            dK_locals[t].assign((size_t)mKvSeq * mKvNumHead * mHeadDim, 0.0f);
            dV_locals[t].assign((size_t)mKvSeq * mKvNumHead * mHeadDim, 0.0f);
        }

        MNN_CONCURRENCY_BEGIN(tId, mThreads) {
            int hBegin = (int)tId * tilePerThread;
            int hEnd   = std::min(hBegin + tilePerThread, mNumHead);
            // 为每个线程循环复用的缓冲
            std::vector<float> Khead(mKvSeq * mHeadDim);
            std::vector<float> Vhead(mKvSeq * mHeadDim);
            std::vector<float> dKhead(mKvSeq * mHeadDim);
            std::vector<float> dVhead(mKvSeq * mHeadDim);
            std::vector<float> dQhead(mSeq   * mHeadDim);
            std::vector<float> logits(mKvSeq);
            std::vector<float> probs(mKvSeq);
            std::vector<float> dP(mKvSeq);

            for (int h = hBegin; h < hEnd; ++h) {
                // 当前 head 对应的 kv_head
                int group = std::max(1, mNumHead / std::max(1, mKvNumHead));
                int kv_h = h / group;
                // 提取当前 kv_head 的 K / V 切片 (layout: [kvSeq, kvHead, headDim])
                for (int j = 0; j < mKvSeq; ++j) {
                    const float* srcK = Kptr + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                    const float* srcV = Vptr + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                    ::memcpy(&Khead[j * mHeadDim], srcK, sizeof(float) * mHeadDim);
                    ::memcpy(&Vhead[j * mHeadDim], srcV, sizeof(float) * mHeadDim);
                }
                // 清零局部梯度
                std::fill(dKhead.begin(), dKhead.end(), 0.f);
                std::fill(dVhead.begin(), dVhead.end(), 0.f);
                std::fill(dQhead.begin(), dQhead.end(), 0.f);

                // Pack K head for this attention head - optimized using MNNPackedMatMul
                auto core = static_cast<CPUBackend*>(backend())->functions();
                auto pack_k = mPackK->host<float>() + (size_t)tId * UP_DIV(mKvSeq, mHP) * mHeadDim * mHP;
                pack_key_for_grad(Khead.data(), pack_k, mKvSeq, mHeadDim, mHP, backend());

                // Process queries one at a time using optimized packed matrix multiplication
                auto pack_q = mPackQ->host<float>() + (size_t)tId * UP_DIV(mSeq, mEP) * mHeadDim * mEP;

                for (int i = 0; i < mSeq; ++i) {
                    // 取当前 head 的 Q_i 和 dY_i
                    const float* Qi  = Qptr  + (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim;
                    const float* dYi = dYptr + (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim;

                    // 1) logits using optimized packed matrix multiplication
                    // Pack single query Qi for efficient matmul with packed K
                    pack_query_for_grad(Qi, pack_q, 1, mHeadDim, mEP);
                    
                    // Compute Q@K^T using MNNPackedMatMulRemain (replaces manual dot product loops)
                    std::vector<float> qk_result(mKvSeq);
                    size_t shapeParameters[7] = {(size_t)mEP * sizeof(float), (size_t)mHeadDim, (size_t)mKvSeq, (size_t)1 * sizeof(float), 0, 0, 0};
                    core->MNNPackedMatMulRemain(qk_result.data(), pack_q, pack_k, 1, shapeParameters, nullptr, nullptr, nullptr, nullptr);
                    
                    // Apply scale and mask
                    for (int j = 0; j < mKvSeq; ++j) {
                        float val = scale * qk_result[j];
                        if (hasMask) {
                            if (maskIsFloat) {
                                val += maskF[i * mKvSeq + j];
                            } else {
                                // int mask: 0 -> -inf, 非0 -> keep
                                val = maskI[i * mKvSeq + j] ? val : -1e30f;
                            }
                        }
                        logits[j] = val;
                    }

                    // 2) softmax
                    float rowMax = logits[0];
                    for (int j = 1; j < mKvSeq; ++j) rowMax = std::max(rowMax, logits[j]);
                    float sumExp = 0.f;
                    for (int j = 0; j < mKvSeq; ++j) {
                        float e = std::exp(logits[j] - rowMax);
                        probs[j] = e;
                        sumExp  += e;
                    }
                    float invSum = 1.f / std::max(sumExp, 1e-12f);
                    for (int j = 0; j < mKvSeq; ++j) {
                        probs[j] *= invSum;
                    }

                    // 3) dV 累加 + dP 计算
                    // dP_j = Σ_d ( dY_i_d * V_j_d )
                    for (int j = 0; j < mKvSeq; ++j) {
                        const float* Vj = &Vhead[j * mHeadDim];
                        float p = probs[j];
                        float dot_dY_V = 0.f;
                        for (int d = 0; d < mHeadDim; ++d) {
                            float dy = dYi[d];
                            dVhead[j * mHeadDim + d] += p * dy; // dV 累加
                            dot_dY_V += dy * Vj[d];
                        }
                        dP[j] = dot_dY_V;
                    }

                    // 4) softmax backward  dS_j = (dP_j - Σ_k dP_k * P_k) * P_j
                    float sum_dP_P = 0.f;
                    for (int j = 0; j < mKvSeq; ++j) {
                        sum_dP_P += dP[j] * probs[j];
                    }

                    // 5) 用 dS (乘 scale) 更新 dQ_i, dK_j
                    for (int j = 0; j < mKvSeq; ++j) {
                        float dS = (dP[j] - sum_dP_P) * probs[j];
                        float coeff = dS * scale; // 链上前面的 scale
                        if (coeff == 0.f) continue;
                        const float* Kj = &Khead[j * mHeadDim];
                        // dQ_i
                        for (int d = 0; d < mHeadDim; ++d) {
                            dQhead[i * mHeadDim + d] += coeff * Kj[d];
                        }
                        // dK_j
                        for (int d = 0; d < mHeadDim; ++d) {
                            dKhead[j * mHeadDim + d] += coeff * Qi[d];
                        }
                    }
                } // i

                // 把局部梯度写回 / 累加
                for (int i = 0; i < mSeq; ++i) {
                    float* dst = dQptr + (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim;
                    ::memcpy(dst, &dQhead[i * mHeadDim], sizeof(float) * mHeadDim);
                }
                // 累加 dK / dV 到线程本地缓冲，避免全局竞争
                auto& dKlocal = dK_locals[(int)tId];
                auto& dVlocal = dV_locals[(int)tId];
                for (int j = 0; j < mKvSeq; ++j) {
                    float* dstK = dKlocal.data() + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                    float* dstV = dVlocal.data() + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                    for (int d = 0; d < mHeadDim; ++d) {
                        dstK[d] += dKhead[j * mHeadDim + d];
                        dstV[d] += dVhead[j * mHeadDim + d];
                    }
                }
            } // head
        }
        MNN_CONCURRENCY_END();

        // 线程本地缓冲归约到全局 dK / dV
        for (int t = 0; t < mThreads; ++t) {
            const auto& dKlocal = dK_locals[t];
            const auto& dVlocal = dV_locals[t];
            size_t n = (size_t)mKvSeq * mKvNumHead * mHeadDim;
            for (size_t idx = 0; idx < n; ++idx) {
                dKptr[idx] += dKlocal[idx];
                dVptr[idx] += dVlocal[idx];
            }
        }

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
