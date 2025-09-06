//
//  CPUAttentionGrad.cpp
//  MNN
//
//  Attention Backward (CPU, fp32)
//  - Block-streaming per head (qBlk) to reduce memory pressure
//  - Use official MNNPackForMatMul_B ("B pack") for dV/dK paths
//  - Vectorized elementwise hot loops (Vec4) for softmax-pre, dS, accumulations
//  - Fused dS*scale into packA (packAWithAlpha) to remove an extra full pass
//  - Cached packed K / V (packKT / packK / packVT)
//
//  NOTE:
//    * Only batch=1 supported
//    * kv_cache backward not implemented (disable kv_cache during training)
//    * Float32 only
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "CPUAttentionGrad.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "math/Vec.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace MNN {

using Vec4 = MNN::Math::Vec<float, 4>;

// ============== Vec helpers ==============
static inline void axpyVec4(float* __restrict dst, const float* __restrict src, float alpha, int dim) {
    if (alpha == 0.f) return;
    int d4 = dim / 4, tail = d4 * 4;
    Vec4 A(alpha);
    for (int i = 0; i < d4; ++i) {
        auto dv = Vec4::load(dst + 4 * i);
        auto sv = Vec4::load(src + 4 * i);
        Vec4::save(dst + 4 * i, dv + sv * A);
    }
    for (int i = tail; i < dim; ++i) dst[i] += src[i] * alpha;
}

// Apply scale and mask to a logits row, output to tmp buffer for softmax input
static inline void applyScaleAndMaskToRow(const float* __restrict logitsRow,
                                          const float* __restrict maskFRow,
                                          const int* __restrict maskIRow,
                                          float scale, int len,
                                          float* __restrict out) {
    const int d4 = len / 4;
    const int tail = d4 * 4;
    const Vec4 S(scale);
    if (maskFRow) {
        for (int i = 0; i < d4; ++i) {
            Vec4 v = Vec4::load(logitsRow + 4 * i) * S;
            v = v + Vec4::load(maskFRow + 4 * i);
            Vec4::save(out + 4 * i, v);
        }
        for (int i = tail; i < len; ++i) out[i] = logitsRow[i] * scale + maskFRow[i];
    } else if (maskIRow) {
        constexpr float C = -1e30f;
        const Vec4 VC(C), ONE(1.f);
        for (int i = 0; i < d4; ++i) {
            Vec4 v = Vec4::load(logitsRow + 4 * i) * S;
            float mf[4];
            const int* mi = maskIRow + 4 * i;
            mf[0] = mi[0] ? 1.f : 0.f;
            mf[1] = mi[1] ? 1.f : 0.f;
            mf[2] = mi[2] ? 1.f : 0.f;
            mf[3] = mi[3] ? 1.f : 0.f;
            Vec4 M = Vec4::load(mf);
            Vec4 res = v * M + VC * (ONE - M);
            Vec4::save(out + 4 * i, res);
        }
        for (int i = tail; i < len; ++i) out[i] = maskIRow[i] ? (logitsRow[i] * scale) : C;
    } else {
        for (int i = 0; i < d4; ++i) {
            Vec4 v = Vec4::load(logitsRow + 4 * i) * S;
            Vec4::save(out + 4 * i, v);
        }
        for (int i = tail; i < len; ++i) out[i] = logitsRow[i] * scale;
    }
}

// ============== pack A (e rows, l cols) ==============
// Original packA without scaling (kept for general use)
void CPUAttentionGrad::packA(const float* src, int e, int l, int eP, std::vector<float>& dst) {
    int blocks = UP_DIV(e, eP);
    dst.resize((size_t)blocks * l * eP);
    for (int b = 0; b < blocks; ++b) {
        int rs = b * eP;
        int realE = std::min(eP, e - rs);
        float* base = dst.data() + (size_t)b * l * eP;
        for (int col = 0; col < l; ++col) {
            for (int r = 0; r < realE; ++r) {
                base[col * eP + r] = src[(size_t)(rs + r) * l + col];
            }
            for (int r = realE; r < eP; ++r) base[col * eP + r] = 0.f;
        }
    }
}

// New: pack A with alpha fusion (dst = alpha * A packed), to fuse dS*scale into packing
static inline void packAWithAlpha(const float* src, int e, int l, int eP, float alpha, std::vector<float>& dst) {
    int blocks = UP_DIV(e, eP);
    dst.resize((size_t)blocks * l * eP);
    if (alpha == 1.0f) {
        // fallback to plain pack pattern
        for (int b = 0; b < blocks; ++b) {
            int rs = b * eP;
            int realE = std::min(eP, e - rs);
            float* base = dst.data() + (size_t)b * l * eP;
            for (int col = 0; col < l; ++col) {
                for (int r = 0; r < realE; ++r) {
                    base[col * eP + r] = src[(size_t)(rs + r) * l + col];
                }
                for (int r = realE; r < eP; ++r) base[col * eP + r] = 0.f;
            }
        }
        return;
    }
    for (int b = 0; b < blocks; ++b) {
        int rs = b * eP;
        int realE = std::min(eP, e - rs);
        float* base = dst.data() + (size_t)b * l * eP;
        for (int col = 0; col < l; ++col) {
            for (int r = 0; r < realE; ++r) {
                base[col * eP + r] = src[(size_t)(rs + r) * l + col] * alpha;
            }
            for (int r = realE; r < eP; ++r) base[col * eP + r] = 0.f;
        }
    }
}

// ============== GEMM C-unpack helper ==============
static inline void unpackC_unit(const float* packC, int e, int h, int unit, float* outRowMajor) {
    // packC layout: [h/unit, e, unit]
    for (int i = 0; i < e; ++i) {
        const float* rowBase = packC + (size_t)i * unit;
        for (int col = 0; col < h; ++col) {
            int ob = col / unit;
            int oi = col % unit;
            outRowMajor[(size_t)i * h + col] = rowBase[(size_t)ob * e * unit + oi];
        }
    }
}

// ============== GEMM wrapper ==============
void CPUAttentionGrad::gemmApackedBpacked(const float* packAptr, int e, int l,
                                          const float* packBptr, int h,
                                          GemmCtx& ctx, const CoreFunctions* core) {
    if (e <= 0 || l <= 0 || h <= 0) return;
    const int unit = core->pack;
    const int bytes = core->bytes;
    size_t param[7] = {
        (size_t)ctx.eP * (size_t)bytes, // e bytes per tile
        (size_t)l,
        (size_t)h,
        (size_t)(e * unit * bytes),     // C stride in bytes
        0,0,0
    };
    size_t need = (size_t)UP_DIV(h, unit) * (size_t)e * (size_t)unit;
    if (ctx.packedOut.size() < need) ctx.packedOut.resize(need);

    int full = e / ctx.eP;
    int remain = e % ctx.eP;
    for (int b = 0; b < full; ++b) {
        float* Cblk = ctx.packedOut.data() + (size_t)b * ctx.eP * unit;
        const float* Ablk = packAptr + (size_t)b * l * ctx.eP;
        core->MNNPackedMatMul(Cblk, Ablk, packBptr, param, nullptr, nullptr, nullptr, nullptr);
    }
    if (remain) {
        float* Cblk = ctx.packedOut.data() + (size_t)full * ctx.eP * unit;
        const float* Ablk = packAptr + (size_t)full * l * ctx.eP;
        core->MNNPackedMatMulRemain(Cblk, Ablk, packBptr, (size_t)remain, param, nullptr, nullptr, nullptr, nullptr);
    }
    if ((int)ctx.rowC.size() < e * h) ctx.rowC.resize((size_t)e * h);
    unpackC_unit(ctx.packedOut.data(), e, h, unit, ctx.rowC.data());
}

// ============== ctor / resize ==============
CPUAttentionGrad::CPUAttentionGrad(Backend *backend)
    : Execution(backend) {}

CPUAttentionGrad::~CPUAttentionGrad() = default;

void CPUAttentionGrad::allocPackedKV() {
    if (mPackAllocated) return;
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int unit = core->pack;
    mPackKT.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mKvSeq,unit), mHeadDim, unit}));
    mPackK .reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mHeadDim,unit), mKvSeq, unit}));
    mPackVT.reset(Tensor::createDevice<float>({mKvNumHead, UP_DIV(mKvSeq,unit), mHeadDim, unit}));
    backend()->onAcquireBuffer(mPackKT.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mPackK .get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mPackVT.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mPackKT.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mPackK .get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mPackVT.get(), Backend::DYNAMIC);
    mPackAllocated = true;
}

ErrorCode CPUAttentionGrad::onResize(const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& /*outputs*/) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    core->MNNGetMatMulPackMode(&mEP, &mLP, &mHP);
    mBytes = core->bytes;
    if (mBytes != 4) return NOT_SUPPORT;
    const Tensor* Q = inputs[0];
    const Tensor* K = inputs[1];
    mBatch     = Q->length(0);
    mSeq       = Q->length(1);
    mNumHead   = Q->length(2);
    mHeadDim   = Q->length(3);
    mKvSeq     = K->length(1);
    mKvNumHead = K->length(2);
    MNN_ASSERT(mBatch == 1);
    MNN_ASSERT(mKvNumHead > 0);
    MNN_ASSERT(mNumHead % mKvNumHead == 0);
    fprintf(stderr, "[Grad:onResize] mSeq=%d mNumHead=%d mHeadDim=%d mKvSeq=%d kvHead=%d eP=%d pack=%d\n",
            mSeq, mNumHead, mHeadDim, mKvSeq, mKvNumHead, mEP, core->pack);
    fflush(stderr);
    mThreads = static_cast<CPUBackend*>(backend())->threadNumber();
    allocPackedKV();
    // Prepare thread buffers
    mThreadBufs.clear();
    mThreadBufs.resize(mThreads);
    for (int t = 0; t < mThreads; ++t) {
        mThreadBufs[t].ctxQK.eP = mEP;
        mThreadBufs[t].ctxDP.eP = mEP;
        mThreadBufs[t].ctxDQ.eP = mEP;
        mThreadBufs[t].ctxDV.eP = mEP;
        mThreadBufs[t].ctxDK.eP = mEP;
        size_t pack = static_cast<CPUBackend*>(backend())->functions()->pack;
        size_t needBQ = (size_t)UP_DIV(mHeadDim, pack) * (size_t)mSeq * pack;
        mThreadBufs[t].packBQ.reserve(needBQ);
        mThreadBufs[t].packBDY.reserve(needBQ);
        size_t maxL = std::max(mSeq, mKvSeq);
        size_t maxE = std::max(mSeq, mKvSeq);
        size_t needABig = (size_t)UP_DIV((int)maxE, mEP) * (size_t)maxL * mEP;
        mThreadBufs[t].packA.reserve(needABig);
        mThreadBufs[t].packAT.reserve(needABig);
        mThreadBufs[t].contigQ.reserve((size_t)mSeq * mHeadDim);
        mThreadBufs[t].contigDY.reserve((size_t)mSeq * mHeadDim);
    }
    // Force enable packATranspose path for dV
    mDVPackATChecked = true;
    mDVPackATUsePackAT = true;
    return NO_ERROR;
}

void CPUAttentionGrad::preparePackedKV(const Tensor* K, const Tensor* V) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    const float* Kptr = K->host<float>();
    const float* Vptr = V->host<float>();
    for (int kv_h = 0; kv_h < mKvNumHead; ++kv_h) {
        std::vector<float> Kbuf(mKvSeq * mHeadDim);
        std::vector<float> Vbuf(mKvSeq * mHeadDim);
        for (int j = 0; j < mKvSeq; ++j) {
            const float* Ks = Kptr + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
            const float* Vs = Vptr + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
            ::memcpy(Kbuf.data() + (size_t)j * mHeadDim, Ks, (size_t)mHeadDim * sizeof(float));
            ::memcpy(Vbuf.data() + (size_t)j * mHeadDim, Vs, (size_t)mHeadDim * sizeof(float));
        }
        int unit = core->pack;
        float* packKTHead = mPackKT->host<float>() +
            (size_t)kv_h * UP_DIV(mKvSeq,unit) * mHeadDim * unit;
        float* packKHead  = mPackK ->host<float>() +
            (size_t)kv_h * UP_DIV(mHeadDim,unit) * mKvSeq * unit;
        float* packVTHead = mPackVT->host<float>() +
            (size_t)kv_h * UP_DIV(mKvSeq,unit) * mHeadDim * unit;
        core->MNNPackForMatMul_B(packKTHead, Kbuf.data(), (size_t)mKvSeq, (size_t)mHeadDim, true);
        core->MNNPackForMatMul_B(packVTHead, Vbuf.data(), (size_t)mKvSeq, (size_t)mHeadDim, true);
        core->MNNPackForMatMul_B(packKHead,  Kbuf.data(), (size_t)mHeadDim, (size_t)mKvSeq, false);
    }
}

// ============== pack A' (transpose view) ==============
static void packATranspose(const float* src, int E, int L, int eP, std::vector<float>& dst) {
    int blocks = UP_DIV(L, eP);
    dst.resize((size_t)blocks * E * eP);
    for (int b = 0; b < blocks; ++b) {
        int rs = b * eP;
        int realE = std::min(eP, L - rs);
        float* base = dst.data() + (size_t)b * E * eP;
        for (int col = 0; col < E; ++col) {
            for (int r = 0; r < realE; ++r) {
                base[col * eP + r] = src[(size_t)col * L + (size_t)(rs + r)];
            }
            for (int r = realE; r < eP; ++r) base[col * eP + r] = 0.f;
        }
    }
}

// ============== Block-streaming execute ==============
void CPUAttentionGrad::executeStreaming(const Tensor* Q, const Tensor* K, const Tensor* V,
                                        const Tensor* mask, const Tensor* dY,
                                        Tensor* dQ, Tensor* dK, Tensor* dV) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    const float* Qptr  = Q->host<float>();
    const float* dYptr = dY->host<float>();
    float* dQptr = dQ->host<float>();
    float* dKptr = dK->host<float>();
    float* dVptr = dV->host<float>();

    preparePackedKV(K, V);

    bool hasMask = (mask != nullptr);
    bool maskIsFloat = hasMask && (mask->getType() == halide_type_of<float>());
    const float* maskF = (hasMask && maskIsFloat) ? mask->host<float>() : nullptr;
    const int*   maskI = (hasMask && !maskIsFloat) ? mask->host<int>()   : nullptr;

    const float scale = 1.f / std::sqrt((float)mHeadDim);
    const int groupSize = mNumHead / mKvNumHead;
    const size_t kvAllSize = (size_t)mKvSeq * mKvNumHead * mHeadDim;

    // thread local accumulators for dK/dV
    std::vector<std::vector<float>> dK_locals(mThreads);
    std::vector<std::vector<float>> dV_locals(mThreads);
    for (int t = 0; t < mThreads; ++t) {
        dK_locals[t].assign(kvAllSize, 0.f);
        dV_locals[t].assign(kvAllSize, 0.f);
    }

    // block size for queries
    const int qBlk = 256; // tune: 128/256/512

    int headsPerThread = UP_DIV(mNumHead, mThreads);

    MNN_CONCURRENCY_BEGIN(tId, mThreads) {
        int hStart = (int)tId * headsPerThread;
        int hEnd   = std::min(hStart + headsPerThread, mNumHead);
        if (hStart >= hEnd) {
            // nothing to do
        } else {
            auto& dKlocal = dK_locals[(int)tId];
            auto& dVlocal = dV_locals[(int)tId];

            auto& tbuf = mThreadBufs[(int)tId];
            auto& ctxQK = tbuf.ctxQK;
            auto& ctxDP = tbuf.ctxDP;
            auto& ctxDQ = tbuf.ctxDQ;
            auto& ctxDV = tbuf.ctxDV;
            auto& ctxDK = tbuf.ctxDK;
            auto& packA_buf    = tbuf.packA;
            auto& packA_tr_buf = tbuf.packAT;
            auto& packB_Q      = tbuf.packBQ;
            auto& packB_dY     = tbuf.packBDY;

            std::vector<float> rowBuf(mKvSeq); // softmax temp

            for (int h = hStart; h < hEnd; ++h) {
                const int kv_h = h / groupSize;
                const int unit = core->pack;
                const float* packKTHead = mPackKT->host<float>() +
                    (size_t)kv_h * UP_DIV(mKvSeq,unit) * mHeadDim * unit;
                const float* packKHead  = mPackK ->host<float>() +
                    (size_t)kv_h * UP_DIV(mHeadDim,unit) * mKvSeq * unit;
                const float* packVTHead = mPackVT->host<float>() +
                    (size_t)kv_h * UP_DIV(mKvSeq,unit) * mHeadDim * unit;

                // loop by query blocks
                for (int q0 = 0; q0 < mSeq; q0 += qBlk) {
                    int qLen = std::min(qBlk, mSeq - q0);

                    // Gather small contiguous blocks for Q and dY (block-local)
                    std::vector<float> Q_blkC((size_t)qLen * mHeadDim);
                    std::vector<float> dY_blkC((size_t)qLen * mHeadDim);
                    for (int bi = 0; bi < qLen; ++bi) {
                        int rowIdx = q0 + bi;
                        const float* sQ = Qptr  + (size_t)rowIdx * mNumHead * mHeadDim + (size_t)h * mHeadDim;
                        const float* sD = dYptr + (size_t)rowIdx * mNumHead * mHeadDim + (size_t)h * mHeadDim;
                        ::memcpy(Q_blkC.data()  + (size_t)bi * mHeadDim, sQ, (size_t)mHeadDim * sizeof(float));
                        ::memcpy(dY_blkC.data() + (size_t)bi * mHeadDim, sD, (size_t)mHeadDim * sizeof(float));
                    }

                    // 1) logits_blk = Q_blk @ K^T
                    packA(Q_blkC.data(), qLen, mHeadDim, mEP, packA_buf);
                    gemmApackedBpacked(packA_buf.data(), qLen, mHeadDim, packKTHead, mKvSeq, ctxQK, core);
                    float* P_blk = ctxQK.rowC.data(); // [qLen x kv_seq]

                    // scale+mask+softmax per row (vectorized preprocessing)
                    for (int bi = 0; bi < qLen; ++bi) {
                        int rowIdx = q0 + bi;
                        float* row = P_blk + (size_t)bi * mKvSeq;
                        const float* mF = (maskF ? maskF + (size_t)rowIdx * mKvSeq : nullptr);
                        const int*   mI = (maskI ? maskI + (size_t)rowIdx * mKvSeq : nullptr);
                        applyScaleAndMaskToRow(row, mF, mI, scale, mKvSeq, rowBuf.data());
                        MNNSoftmax(row, rowBuf.data(), (size_t)mKvSeq);
                    }

                    // 2) dP_blk = dY_blk @ V^T
                    packA(dY_blkC.data(), qLen, mHeadDim, mEP, packA_buf);
                    gemmApackedBpacked(packA_buf.data(), qLen, mHeadDim, packVTHead, mKvSeq, ctxDP, core);
                    float* dP_blk = ctxDP.rowC.data(); // [qLen x kv_seq]

                    // 3) dS_blk = (dP - (dP ⊙ P)1) ⊙ P (vectorized)
                    for (int bi = 0; bi < qLen; ++bi) {
                        float* pRow  = P_blk  + (size_t)bi * mKvSeq;
                        float* dpRow = dP_blk + (size_t)bi * mKvSeq;
                        int d4 = mKvSeq / 4, tail = d4 * 4;
                        Vec4 vacc(0.f);
                        for (int j = 0; j < d4; ++j) {
                            vacc = vacc + Vec4::load(dpRow + 4 * j) * Vec4::load(pRow + 4 * j);
                        }
                        float sum4 = vacc[0] + vacc[1] + vacc[2] + vacc[3];
                        double sum_dp_p = (double)sum4;
                        for (int j = tail; j < mKvSeq; ++j) sum_dp_p += (double)dpRow[j] * (double)pRow[j];
                        float s = (float)sum_dp_p;
                        Vec4 S(s);
                        for (int j = 0; j < d4; ++j) {
                            Vec4 dpv = Vec4::load(dpRow + 4 * j);
                            Vec4 pv  = Vec4::load(pRow  + 4 * j);
                            Vec4::save(dpRow + 4 * j, (dpv - S) * pv);
                        }
                        for (int j = tail; j < mKvSeq; ++j) dpRow[j] = (dpRow[j] - s) * pRow[j];
                    }
                    float* dS_blk = dP_blk;

                    // 4) dQ_blk = (dS_blk * scale) @ K
                    // Fused: pack A with alpha=scale to avoid an extra full pass over dS
                    packAWithAlpha(dS_blk, qLen, mKvSeq, mEP, scale, packA_buf);
                    gemmApackedBpacked(packA_buf.data(), qLen, mKvSeq, packKHead, mHeadDim, ctxDQ, core);
                    // write back to dQ
                    for (int bi = 0; bi < qLen; ++bi) {
                        float* src = ctxDQ.rowC.data() + (size_t)bi * mHeadDim;
                        float* dst = dQptr + ((size_t)(q0 + bi) * mNumHead + (size_t)h) * mHeadDim;
                        ::memcpy(dst, src, (size_t)mHeadDim * sizeof(float));
                    }

                    // Prepare packed B for dV / dK from small contiguous blocks (official packB)
                    int pack = core->pack;
                    size_t needB = (size_t)UP_DIV(mHeadDim, pack) * (size_t)qLen * pack;
                    if (packB_dY.size() < needB) packB_dY.resize(needB);
                    if (packB_Q.size()  < needB) packB_Q.resize(needB);
                    core->MNNPackForMatMul_B(packB_dY.data(), dY_blkC.data(), (size_t)mHeadDim, (size_t)qLen, false);
                    core->MNNPackForMatMul_B(packB_Q .data(),  Q_blkC.data(),  (size_t)mHeadDim, (size_t)qLen, false);

                    // 5) dV += P_blk^T @ dY_blk
                    packATranspose(P_blk, qLen, mKvSeq, mEP, packA_tr_buf);
                    gemmApackedBpacked(packA_tr_buf.data(), mKvSeq, qLen, packB_dY.data(), mHeadDim, ctxDV, core);
                    for (int j = 0; j < mKvSeq; ++j) {
                        float* dstV = dVlocal.data() + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                        const float* src = ctxDV.rowC.data() + (size_t)j * mHeadDim;
                        int d4 = mHeadDim / 4, tail = d4 * 4;
                        for (int u = 0; u < d4; ++u) {
                            Vec4::save(dstV + 4 * u, Vec4::load(dstV + 4 * u) + Vec4::load(src + 4 * u));
                        }
                        for (int u = tail; u < mHeadDim; ++u) dstV[u] += src[u];
                    }

                    // 6) dK += dS_blk^T @ Q_blk  (accumulate with scale)
                    packATranspose(dS_blk, qLen, mKvSeq, mEP, packA_tr_buf);
                    gemmApackedBpacked(packA_tr_buf.data(), mKvSeq, qLen, packB_Q.data(), mHeadDim, ctxDK, core);
                    {
                        const Vec4 S(scale);
                        for (int j = 0; j < mKvSeq; ++j) {
                            float* dstK = dKlocal.data() + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                            const float* src = ctxDK.rowC.data() + (size_t)j * mHeadDim;
                            int d4 = mHeadDim / 4, tail = d4 * 4;
                            for (int u = 0; u < d4; ++u) {
                                Vec4::save(dstK + 4 * u, Vec4::load(dstK + 4 * u) + Vec4::load(src + 4 * u) * S);
                            }
                            for (int u = tail; u < mHeadDim; ++u) dstK[u] += src[u] * scale;
                        }
                    }
                } // q-block
            } // head loop
        }
    }
    MNN_CONCURRENCY_END();

    // reduction to global dK/dV (vectorized)
    const int kvAll = (int)((size_t)mKvSeq * mKvNumHead * mHeadDim);
    for (int t = 0; t < mThreads; ++t) {
        axpyVec4(dKptr, dK_locals[t].data(), 1.f, kvAll);
        axpyVec4(dVptr, dV_locals[t].data(), 1.f, kvAll);
    }
}

// ============== onExecute ==============
ErrorCode CPUAttentionGrad::onExecute(const std::vector<Tensor*>& inputs,
                                      const std::vector<Tensor*>& outputs) {
    const Tensor* Q = inputs[0];
    const Tensor* K = inputs[1];
    const Tensor* V = inputs[2];
    AUTOTIME;
    const Tensor* mask = nullptr;
    const Tensor* dY   = nullptr;
    if (inputs.size() == 5) { mask = inputs[3]; dY = inputs[4]; }
    else { dY = inputs[3]; }
    Tensor* dQ = outputs[0];
    Tensor* dK = outputs[1];
    Tensor* dV = outputs[2];

    ::memset(dQ->host<void>(), 0, dQ->size());
    ::memset(dK->host<void>(), 0, dK->size());
    ::memset(dV->host<void>(), 0, dV->size());

    executeStreaming(Q, K, V, mask, dY, dQ, dK, dV);
    return NO_ERROR;
}

// ============== Clone ==============
bool CPUAttentionGrad::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!dst) return true;
    auto exe = new CPUAttentionGrad(bn);
    *dst = exe;
    return true;
}

class CPUAttentionGradCreator : public CPUBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs,
                        const Op* op,
                        Backend* backend) const override {
        return new CPUAttentionGrad(backend);
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUAttentionGradCreator, OpType_AttentionGrad);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE