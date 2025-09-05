//
//  CPUAttentionGrad.cpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#define MNN_SUPPORT_TRANSFORMER_FUSE
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <MNN/AutoTime.hpp>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstring>
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

namespace {
// Simple matrix helpers used to compute attention gradients. They avoid the
// overhead of packing routines while still providing cache friendly access
// patterns similar to CPUAttention.cpp.
static inline void matMul(float* C, const float* A, const float* B, int M, int N, int K) {
    // C[M*N] = A[M*K] * B[K*N]
    for (int i = 0; i < M; ++i) {
        float* cRow = C + i * N;
        const float* aRow = A + i * K;
        for (int k = 0; k < K; ++k) {
            float aval = aRow[k];
            const float* bRow = B + k * N;
            for (int j = 0; j < N; ++j) {
                cRow[j] += aval * bRow[j];
            }
        }
    }
}

static inline void matMulABT(float* C, const float* A, const float* B, int M, int N, int K) {
    // C[M*N] = A[M*K] * B[N*K]^T
    for (int i = 0; i < M; ++i) {
        float* cRow = C + i * N;
        const float* aRow = A + i * K;
        for (int j = 0; j < N; ++j) {
            const float* bRow = B + j * K;
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                sum += aRow[k] * bRow[k];
            }
            cRow[j] = sum;
        }
    }
}

static inline void matMulTransA(float* C, const float* A, const float* B, int M, int N, int K) {
    // C[M*N] = A^T(K*M) * B(K*N)
    for (int m = 0; m < M; ++m) {
        float* cRow = C + m * N;
        for (int n = 0; n < N; ++n) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                sum += A[k * M + m] * B[k * N + n];
            }
            cRow[n] = sum;
        }
    }
}
} // namespace

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

            std::vector<float> Qhead(mSeq * mHeadDim);
            std::vector<float> dYhead(mSeq * mHeadDim);
            std::vector<float> Khead(mKvSeq * mHeadDim);
            std::vector<float> Vhead(mKvSeq * mHeadDim);
            std::vector<float> logits(mSeq * mKvSeq);
            std::vector<float> probs(mSeq * mKvSeq);
            std::vector<float> dP(mSeq * mKvSeq);
            std::vector<float> dQhead(mSeq * mHeadDim);
            std::vector<float> dKhead(mKvSeq * mHeadDim);
            std::vector<float> dVhead(mKvSeq * mHeadDim);

            for (int h = hBegin; h < hEnd; ++h) {
                int group = std::max(1, mNumHead / std::max(1, mKvNumHead));
                int kv_h = h / group;
                // gather Q and dY for this head
                for (int i = 0; i < mSeq; ++i) {
                    const float* srcQ = Qptr + (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim;
                    const float* srcdY = dYptr + (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim;
                    ::memcpy(Qhead.data() + i * mHeadDim, srcQ, sizeof(float) * mHeadDim);
                    ::memcpy(dYhead.data() + i * mHeadDim, srcdY, sizeof(float) * mHeadDim);
                }
                // gather K/V slice for kv head
                for (int j = 0; j < mKvSeq; ++j) {
                    const float* srcK = Kptr + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                    const float* srcV = Vptr + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                    ::memcpy(Khead.data() + j * mHeadDim, srcK, sizeof(float) * mHeadDim);
                    ::memcpy(Vhead.data() + j * mHeadDim, srcV, sizeof(float) * mHeadDim);
                }

                std::fill(logits.begin(), logits.end(), 0.f);
                matMulABT(logits.data(), Qhead.data(), Khead.data(), mSeq, mKvSeq, mHeadDim);
                // apply scale and mask
                for (int i = 0; i < mSeq; ++i) {
                    float* row = logits.data() + i * mKvSeq;
                    for (int j = 0; j < mKvSeq; ++j) {
                        float val = row[j] * scale;
                        if (hasMask) {
                            if (maskIsFloat) {
                                val += maskF[i * mKvSeq + j];
                            } else {
                                val = maskI[i * mKvSeq + j] ? val : -1e30f;
                            }
                        }
                        row[j] = val;
                    }
                }

                // softmax for each query
                for (int i = 0; i < mSeq; ++i) {
                    MNNSoftmax(probs.data() + i * mKvSeq, logits.data() + i * mKvSeq, mKvSeq);
                }

                // dV = P^T * dY
                std::fill(dVhead.begin(), dVhead.end(), 0.f);
                matMulTransA(dVhead.data(), probs.data(), dYhead.data(), mKvSeq, mHeadDim, mSeq);

                // dP = dY * V^T
                matMulABT(dP.data(), dYhead.data(), Vhead.data(), mSeq, mKvSeq, mHeadDim);

                // softmax backward
                for (int i = 0; i < mSeq; ++i) {
                    float* pRow = probs.data() + i * mKvSeq;
                    float* dRow = dP.data() + i * mKvSeq;
                    float sum = 0.f;
                    for (int j = 0; j < mKvSeq; ++j) {
                        sum += dRow[j] * pRow[j];
                    }
                    for (int j = 0; j < mKvSeq; ++j) {
                        dRow[j] = (dRow[j] - sum) * pRow[j];
                    }
                }

                // dQ = dS * K
                std::fill(dQhead.begin(), dQhead.end(), 0.f);
                matMul(dQhead.data(), dP.data(), Khead.data(), mSeq, mHeadDim, mKvSeq);
                for (float& v : dQhead) {
                    v *= scale;
                }

                // dK = dS^T * Q
                std::fill(dKhead.begin(), dKhead.end(), 0.f);
                matMulTransA(dKhead.data(), dP.data(), Qhead.data(), mKvSeq, mHeadDim, mSeq);
                for (float& v : dKhead) {
                    v *= scale;
                }

                // write dQ
                for (int i = 0; i < mSeq; ++i) {
                    float* dst = dQptr + (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim;
                    ::memcpy(dst, dQhead.data() + i * mHeadDim, sizeof(float) * mHeadDim);
                }

                // accumulate dK and dV into thread local buffer
                auto& dKlocal = dK_locals[(int)tId];
                auto& dVlocal = dV_locals[(int)tId];
                for (int j = 0; j < mKvSeq; ++j) {
                    float* dstK = dKlocal.data() + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                    float* dstV = dVlocal.data() + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                    const float* srcK = dKhead.data() + j * mHeadDim;
                    const float* srcV = dVhead.data() + j * mHeadDim;
                    for (int d = 0; d < mHeadDim; ++d) {
                        dstK[d] += srcK[d];
                        dstV[d] += srcV[d];
                    }
                }
            }
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
