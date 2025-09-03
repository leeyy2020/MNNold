//
//  CPUAttentionGrad.cpp
//  MNN
//
//  Backward (gradient) kernel for fused Attention on CPU.
//  This mirrors the packing strategy used in CPUAttention.cpp (fp32 path).
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "CPUBackend.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

// Pack a row-major matrix A (e x l) into [e/eP, l, eP] layout for left operand
template <typename T>
static void packA_e(char* dstPack, const T* src, int e, int l, int eP) {
    auto dst = reinterpret_cast<T*>(dstPack);
    for (int i = 0; i < e; ++i) {
        int out_index = i / eP;
        int in_index  = i % eP;
        for (int j = 0; j < l; ++j) {
            dst[out_index * l * eP + j * eP + in_index] = src[i * l + j];
        }
    }
}

// Pack transposed rows: given src in row-major [l x e], pack A with shape (e x l)
template <typename T>
static void packA_from_transpose(char* dstPack, const T* srcLT, int e, int l, int eP) {
    auto dst = reinterpret_cast<T*>(dstPack);
    for (int i = 0; i < e; ++i) {
        int out_index = i / eP;
        int in_index  = i % eP;
        for (int j = 0; j < l; ++j) {
            // srcLT is [l x e], access srcLT[j, i]
            dst[out_index * l * eP + j * eP + in_index] = srcLT[j * e + i];
        }
    }
}

// Unpack right-hand packed C (result) [h/unit, e, unit] to row-major [e x h]
template <typename T>
static void unpackC_eh(const char* srcPack, T* dst, int e, int h, int unit) {
    auto src = reinterpret_cast<const T*>(srcPack);
    for (int i = 0; i < e; ++i) {
        for (int j = 0; j < h; ++j) {
            int a = j / unit;
            int b = j % unit;
            dst[i * h + j] = src[a * e * unit + i * unit + b];
        }
    }
}

// Unpack QK (left-matmul output) [kv/unit, e, unit] -> row-major [e x kv]
template <typename T>
static void unpack_QK(float* unpack_qk_dst, const char* pack_qk_src, int e, int kv, int unit) {
    auto dst = unpack_qk_dst;
    auto src = reinterpret_cast<const T*>(pack_qk_src);
    for (int i = 0; i < e; i++) {
        for (int j = 0; j < kv; j++) {
            int out_index = j / unit;
            int in_index  = j % unit;
            dst[i * kv + j] = src[out_index * e * unit + i * unit + in_index];
        }
    }
}

// Row-wise softmax
static void row_softmax(float* dst, const float* src, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        const float* s = src + i * cols;
        float* d = dst + i * cols;
        float m = s[0];
        for (int j = 1; j < cols; ++j) m = std::max(m, s[j]);
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            d[j] = std::exp(s[j] - m);
            sum += d[j];
        }
        float inv = 1.0f / std::max(sum, 1e-12f);
        for (int j = 0; j < cols; ++j) d[j] *= inv;
    }
}

class CPUAttentionGrad : public Execution {
public:
    CPUAttentionGrad(Backend* b) : Execution(b) {}
    virtual ~CPUAttentionGrad() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto core = static_cast<CPUBackend*>(backend())->functions();
        core->MNNGetMatMulPackMode(&mEP, &mLP, &mHP);
        mBytes = core->bytes;
        if (mBytes != 4) {
            // Only fp32 path is implemented currently
            return NOT_SUPPORT;
        }
        mThreads = static_cast<CPUBackend*>(backend())->threadNumber();
        // Expect inputs: Q, K, V, [mask], dY; outputs: dQ, dK, dV
        auto Q = inputs[0];
        auto K = inputs[1];
        auto V = inputs[2];
        mSeq = Q->length(1);
        mNumHead = Q->length(2);
        mHeadDim = Q->length(3);
        mKvSeq = K->length(1);
        // Temps per-thread
        mPackA.reset(Tensor::createDevice<uint8_t>({mThreads, UP_DIV(mSeq, mEP), mHeadDim, mEP * mBytes}));
        mPackB_K.reset(Tensor::createDevice<float>({mThreads, UP_DIV(mKvSeq, mHP), mHeadDim, mHP}));
        mPackB_V.reset(Tensor::createDevice<float>({mThreads, UP_DIV(mKvSeq, mHP), mHeadDim, mHP}));
        mPackB_dY.reset(Tensor::createDevice<float>({mThreads, UP_DIV(mHeadDim, mHP), mSeq, mHP}));
        mPackOut.reset(Tensor::createDevice<float>({mThreads, UP_DIV(mKvSeq, mHP), mSeq, mHP}));
        mQK.reset(Tensor::createDevice<float>({mThreads, mSeq, mKvSeq}));
        mProb.reset(Tensor::createDevice<float>({mThreads, mSeq, mKvSeq}));
        mDZ.reset(Tensor::createDevice<float>({mThreads, mSeq, mKvSeq}));
        // Acquire then release as dynamic
        for (auto t : {mPackA.get(), mPackB_K.get(), mPackB_V.get(), mPackB_dY.get(), mPackOut.get(), mQK.get(), mProb.get(), mDZ.get()}) {
            if (!backend()->onAcquireBuffer(t, Backend::DYNAMIC)) return OUT_OF_MEMORY;
        }
        for (auto t : {mPackA.get(), mPackB_K.get(), mPackB_V.get(), mPackB_dY.get(), mPackOut.get(), mQK.get(), mProb.get(), mDZ.get()}) {
            backend()->onReleaseBuffer(t, Backend::DYNAMIC);
        }
        return NO_ERROR;
    }

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto core = static_cast<CPUBackend*>(backend())->functions();
        // Inputs
        auto Q = inputs[0];
        auto K = inputs[1];
        auto V = inputs[2];
        const Tensor* mask = nullptr;
        const Tensor* dY = nullptr;
        if (inputs.size() == 5) {
            mask = inputs[3];
            dY = inputs[4];
        } else {
            dY = inputs.back();
        }
        // Outputs
        auto dQ = outputs[0];
        auto dK = outputs[1];
        auto dV = outputs[2];

        const float scale = 1.0f / std::sqrt((float)mHeadDim);
        const int tilePerThread = UP_DIV(mNumHead, mThreads);

        MNN_CONCURRENCY_BEGIN(tId, mThreads) {
            int hBegin = (int)tId * tilePerThread;
            int hEnd = std::min(hBegin + tilePerThread, mNumHead);
            auto packA = mPackA->host<uint8_t>() + (size_t)tId * UP_DIV(mSeq, mEP) * mHeadDim * mEP * mBytes;
            auto packB_K = mPackB_K->host<float>() + (size_t)tId * UP_DIV(mKvSeq, mHP) * mHeadDim * mHP;
            auto packB_V = mPackB_V->host<float>() + (size_t)tId * UP_DIV(mKvSeq, mHP) * mHeadDim * mHP;
            auto packB_dy = mPackB_dY->host<float>() + (size_t)tId * UP_DIV(mHeadDim, mHP) * mSeq * mHP;
            auto packOut = mPackOut->host<float>() + (size_t)tId * UP_DIV(mKvSeq, mHP) * mSeq * mHP;
            auto qk = mQK->host<float>() + (size_t)tId * mSeq * mKvSeq;
            auto prob = mProb->host<float>() + (size_t)tId * mSeq * mKvSeq;
            auto dz = mDZ->host<float>() + (size_t)tId * mSeq * mKvSeq;

            // Pre-pack K and V (B pack expects row-major [l x h], transpose=true)
            for (int h = hBegin; h < hEnd; ++h) {
                // Head mapping: K/V indexed by kv head equals h (no kv grouping assumed here)
                // Build contiguous per-head K/V blocks: [kvSeq x headDim]
                std::unique_ptr<float[]> Khead(new float[mKvSeq * mHeadDim]);
                std::unique_ptr<float[]> Vhead(new float[mKvSeq * mHeadDim]);
                for (int i = 0; i < mKvSeq; ++i) {
                    auto srcK = K->host<float>() + i * mNumHead * mHeadDim + h * mHeadDim;
                    auto srcV = V->host<float>() + i * mNumHead * mHeadDim + h * mHeadDim;
                    ::memcpy(Khead.get() + i * mHeadDim, srcK, sizeof(float) * mHeadDim);
                    ::memcpy(Vhead.get() + i * mHeadDim, srcV, sizeof(float) * mHeadDim);
                }
                // Pack B from contiguous [l x h] blocks
                core->MNNPackForMatMul_B(packB_K, Khead.get(), (size_t)mHeadDim, (size_t)mKvSeq, true);
                core->MNNPackForMatMul_B(packB_V, Vhead.get(), (size_t)mHeadDim, (size_t)mKvSeq, true);

                // 1) Compute logits: qk = (Q @ K^T) * scale + mask
                // Pack A from Q for this head
                // Gather contiguous Q_head [seq, headDim]
                std::unique_ptr<float[]> Qhead(new float[mSeq * mHeadDim]);
                for (int i = 0; i < mSeq; ++i) {
                    auto src = Q->host<float>() + i * mNumHead * mHeadDim + h * mHeadDim;
                    ::memcpy(Qhead.get() + i * mHeadDim, src, sizeof(float) * mHeadDim);
                }
                packA_e<float>(reinterpret_cast<char*>(packA), Qhead.get(), mSeq, mHeadDim, mEP);
                // MatMul: A [e=mSeq, l=headDim] x B [l=headDim, h=kvSeq] -> C packed [h/mHP, e, mHP]
                size_t shape[7] = {(size_t)mEP * (size_t)mBytes, (size_t)mHeadDim, (size_t)mKvSeq, (size_t)mSeq * (size_t)mHP * (size_t)mBytes, 0, 0, 0};
                int loop_e = mSeq / mEP, remain = mSeq % mEP;
                for (int i = 0; i < loop_e; i++) {
                    core->MNNPackedMatMul((float*)(packOut + i * mEP * mHP), (float*)(packA + i * mHeadDim * mEP * mBytes), (float*)packB_K, shape, nullptr, nullptr, nullptr, nullptr);
                }
                core->MNNPackedMatMulRemain((float*)(packOut + loop_e * mEP * mHP), (float*)(packA + loop_e * mHeadDim * mEP * mBytes), (float*)packB_K, remain, shape, nullptr, nullptr, nullptr, nullptr);
                // Unpack qk to [seq x kv]
                unpack_QK<float>(qk, reinterpret_cast<const char*>(packOut), mSeq, mKvSeq, mHP);
                // Apply scale and mask
                if (mask) {
                    if (mask->getType() == halide_type_of<float>()) {
                        const float* mptr = mask->host<float>();
                        for (int i = 0; i < mSeq * mKvSeq; ++i) qk[i] = qk[i] * scale + mptr[i];
                    } else {
                        const int* mptr = mask->host<int>();
                        for (int i = 0; i < mSeq * mKvSeq; ++i) qk[i] = mptr[i] ? (qk[i] * scale) : (-std::numeric_limits<float>::infinity());
                    }
                } else {
                    for (int i = 0; i < mSeq * mKvSeq; ++i) qk[i] = qk[i] * scale;
                }
                // Softmax probabilities P
                row_softmax(prob, qk, mSeq, mKvSeq);

                // 2) dV = P^T @ dY
                // Pack B from dY [seq x headDim]
                std::unique_ptr<float[]> dYhead(new float[mSeq * mHeadDim]);
                for (int i = 0; i < mSeq; ++i) {
                    auto src = dY->host<float>() + i * mNumHead * mHeadDim + h * mHeadDim;
                    ::memcpy(dYhead.get() + i * mHeadDim, src, sizeof(float) * mHeadDim);
                }
                core->MNNPackForMatMul_B(packB_dy, dYhead.get(), (size_t)mHeadDim, (size_t)mSeq, true);
                // Pack A from P^T: A(e=kvSeq, l=seq)
                packA_from_transpose<float>(reinterpret_cast<char*>(packA), prob, mKvSeq, mSeq, mEP);
                // MatMul -> C packed [h/mHP, e=kvSeq, mHP]
                size_t shapeDV[7] = {(size_t)mEP * (size_t)mBytes, (size_t)mSeq, (size_t)mHeadDim, (size_t)mKvSeq * (size_t)mHP * (size_t)mBytes, 0, 0, 0};
                loop_e = mKvSeq / mEP; remain = mKvSeq % mEP;
                for (int i = 0; i < loop_e; i++) {
                    core->MNNPackedMatMul((float*)(packOut + i * mEP * mHP), (float*)(packA + i * mSeq * mEP * mBytes), (float*)packB_dy, shapeDV, nullptr, nullptr, nullptr, nullptr);
                }
                core->MNNPackedMatMulRemain((float*)(packOut + loop_e * mEP * mHP), (float*)(packA + loop_e * mSeq * mEP * mBytes), (float*)packB_dy, remain, shapeDV, nullptr, nullptr, nullptr, nullptr);
                // Unpack to dV head [kvSeq x headDim]
                std::unique_ptr<float[]> dVhead(new float[mKvSeq * mHeadDim]);
                unpackC_eh<float>(reinterpret_cast<const char*>(packOut), dVhead.get(), mKvSeq, mHeadDim, mHP);
                // Write back to output dV with layout [kvSeq, numHead, headDim]
                for (int i = 0; i < mKvSeq; ++i) {
                    auto dst = dV->host<float>() + i * mNumHead * mHeadDim + h * mHeadDim;
                    ::memcpy(dst, dVhead.get() + i * mHeadDim, sizeof(float) * mHeadDim);
                }

                // 3) dP = dY @ V^T
                // Pack B from V^T (using V row-major with transpose=true, l=headDim, h=kvSeq)
                // packB_V already prepared above
                // Pack A from dY [seq x headDim]
                packA_e<float>(reinterpret_cast<char*>(packA), dYhead.get(), mSeq, mHeadDim, mEP);
                // MatMul -> C packed [kvSeq/unit, seq, unit]
                size_t shapeDP[7] = {(size_t)mEP * (size_t)mBytes, (size_t)mHeadDim, (size_t)mKvSeq, (size_t)mSeq * (size_t)mHP * (size_t)mBytes, 0, 0, 0};
                loop_e = mSeq / mEP; remain = mSeq % mEP;
                for (int i = 0; i < loop_e; i++) {
                    core->MNNPackedMatMul((float*)(packOut + i * mEP * mHP), (float*)(packA + i * mHeadDim * mEP * mBytes), (float*)packB_V, shapeDP, nullptr, nullptr, nullptr, nullptr);
                }
                core->MNNPackedMatMulRemain((float*)(packOut + loop_e * mEP * mHP), (float*)(packA + loop_e * mHeadDim * mEP * mBytes), (float*)packB_V, remain, shapeDP, nullptr, nullptr, nullptr, nullptr);
                // Unpack dP to dz buffer (reuse dz)
                unpack_QK<float>(dz, reinterpret_cast<const char*>(packOut), mSeq, mKvSeq, mHP);
                // 4) softmax backward: dz = dP - sum(dP * P, axis=-1) * P; and scale factor
                for (int i = 0; i < mSeq; ++i) {
                    float dot = 0.0f;
                    const float* pRow = prob + i * mKvSeq;
                    const float* gRow = dz + i * mKvSeq; // currently holds dP
                    for (int j = 0; j < mKvSeq; ++j) dot += gRow[j] * pRow[j];
                    for (int j = 0; j < mKvSeq; ++j) {
                        float val = gRow[j] - dot * pRow[j];
                        dz[i * mKvSeq + j] = val * scale; // chain with scale
                    }
                }

                // 5) dQ = dz @ K
                // Pack B from K (as [l=kvSeq, h=headDim])
                // packB_K already prepared above
                // Pack A from dz [seq x kvSeq]
                packA_e<float>(reinterpret_cast<char*>(packA), dz, mSeq, mKvSeq, mEP);
                size_t shapeDQ[7] = {(size_t)mEP * (size_t)mBytes, (size_t)mKvSeq, (size_t)mHeadDim, (size_t)mSeq * (size_t)mHP * (size_t)mBytes, 0, 0, 0};
                loop_e = mSeq / mEP; remain = mSeq % mEP;
                for (int i = 0; i < loop_e; i++) {
                    core->MNNPackedMatMul((float*)(packOut + i * mEP * mHP), (float*)(packA + i * mKvSeq * mEP * mBytes), (float*)packB_K, shapeDQ, nullptr, nullptr, nullptr, nullptr);
                }
                core->MNNPackedMatMulRemain((float*)(packOut + loop_e * mEP * mHP), (float*)(packA + loop_e * mKvSeq * mEP * mBytes), (float*)packB_K, remain, shapeDQ, nullptr, nullptr, nullptr, nullptr);
                // Unpack and write dQ head [seq x headDim]
                std::unique_ptr<float[]> dQhead(new float[mSeq * mHeadDim]);
                unpackC_eh<float>(reinterpret_cast<const char*>(packOut), dQhead.get(), mSeq, mHeadDim, mHP);
                for (int i = 0; i < mSeq; ++i) {
                    auto dst = dQ->host<float>() + i * mNumHead * mHeadDim + h * mHeadDim;
                    ::memcpy(dst, dQhead.get() + i * mHeadDim, sizeof(float) * mHeadDim);
                }

                // 6) dK = dz^T @ Q
                // Pack B from Q [seq x headDim]
                core->MNNPackForMatMul_B(packB_dy, Qhead.get(), (size_t)mHeadDim, (size_t)mSeq, true);
                // Pack A from dz^T: [kvSeq x seq]
                packA_from_transpose<float>(reinterpret_cast<char*>(packA), dz, mKvSeq, mSeq, mEP);
                size_t shapeDK[7] = {(size_t)mEP * (size_t)mBytes, (size_t)mSeq, (size_t)mHeadDim, (size_t)mKvSeq * (size_t)mHP * (size_t)mBytes, 0, 0, 0};
                loop_e = mKvSeq / mEP; remain = mKvSeq % mEP;
                for (int i = 0; i < loop_e; i++) {
                    core->MNNPackedMatMul((float*)(packOut + i * mEP * mHP), (float*)(packA + i * mSeq * mEP * mBytes), (float*)packB_dy, shapeDK, nullptr, nullptr, nullptr, nullptr);
                }
                core->MNNPackedMatMulRemain((float*)(packOut + loop_e * mEP * mHP), (float*)(packA + loop_e * mSeq * mEP * mBytes), (float*)packB_dy, remain, shapeDK, nullptr, nullptr, nullptr, nullptr);
                // Unpack and write dK head [kvSeq x headDim]
                std::unique_ptr<float[]> dKhead(new float[mKvSeq * mHeadDim]);
                unpackC_eh<float>(reinterpret_cast<const char*>(packOut), dKhead.get(), mKvSeq, mHeadDim, mHP);
                for (int i = 0; i < mKvSeq; ++i) {
                    auto dst = dK->host<float>() + i * mNumHead * mHeadDim + h * mHeadDim;
                    ::memcpy(dst, dKhead.get() + i * mHeadDim, sizeof(float) * mHeadDim);
                }
            }
        }
        MNN_CONCURRENCY_END();

        return NO_ERROR;
    }

private:
    int mEP = 4, mLP = 1, mHP = 4, mBytes = 4;
    int mThreads = 1;
    int mSeq = 0, mKvSeq = 0, mNumHead = 0, mHeadDim = 0;
    std::shared_ptr<Tensor> mPackA, mPackB_K, mPackB_V, mPackB_dY, mPackOut;
    std::shared_ptr<Tensor> mQK, mProb, mDZ;
};

// NOTE: Intentionally not registered here to avoid conflicting with forward creator.
// Hook selection logic in CPUAttention creator if needed (e.g., based on outputs size).

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

// Factory for creator routing
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
namespace MNN {
Execution* createCPUAttentionGrad(Backend* b) {
    return new CPUAttentionGrad(b);
}
}
#endif
