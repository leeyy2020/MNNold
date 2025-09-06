//
//  CPUAttentionGrad.cpp
//  MNN
//
//  Attention Backward (CPU, fp32)
//  - Streaming / Head GEMM / Group GEMM execution paths
//  - Cached packed K / V (packKT / packK / packVT)
//  - This version fixes missing local buffer declarations (Q_head / dY_head)
//    in executeGroupGemm that caused compilation errors.
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
static inline float dotVec4(const float* __restrict a, const float* __restrict b, int dim) {
    int d4 = dim / 4;
    int tail = d4 * 4;
    Vec4 acc(0.f);
    for (int i = 0; i < d4; ++i) {
        acc = Vec4::fma(acc, Vec4::load(a + 4 * i), Vec4::load(b + 4 * i));
    }
    float s = acc[0] + acc[1] + acc[2] + acc[3];
    for (int i = tail; i < dim; ++i) s += a[i] * b[i];
    return s;
}
static inline void axpyVec4(float* __restrict dst, const float* __restrict src, float alpha, int dim) {
    if (alpha == 0.f) return;
    int d4 = dim / 4;
    int tail = d4 * 4;
    Vec4 A(alpha);
    for (int i = 0; i < d4; ++i) {
        auto dv = Vec4::load(dst + 4 * i);
        auto sv = Vec4::load(src + 4 * i);
        Vec4::save(dst + 4 * i, dv + sv * A);
    }
    for (int i = tail; i < dim; ++i) dst[i] += src[i] * alpha;
}

// ============== pack A (e rows, l cols) ==============
void CPUAttentionGrad::packA(const float* src, int e, int l, int eP, std::vector<float>& dst) {
    int blocks = UP_DIV(e, eP);
    dst.resize((size_t)blocks * l * eP);
    for (int b = 0; b < blocks; ++b) {
        int rs = b * eP;
        int realE = std::min(eP, e - rs);
        float* base = dst.data() + b * l * eP;
        for (int col = 0; col < l; ++col) {
            for (int r = 0; r < realE; ++r) {
                base[col * eP + r] = src[(rs + r) * l + col];
            }
            for (int r = realE; r < eP; ++r) base[col * eP + r] = 0.f;
        }
    }
}

// ============== unpack PackedMatMulRemain output ==============
static void unpackPackedC(const float* packedC, int e, int h, int cStrideFloat, float* out) {
    int hC4 = UP_DIV(h, 4);
    for (int yb = 0; yb < hC4; ++yb) {
        const float* block = packedC + yb * cStrideFloat;
        int base = yb * 4;
        for (int x = 0; x < e; ++x) {
            const float* row = block + 4 * x;
            for (int k = 0; k < 4; ++k) {
                int col = base + k;
                if (col < h) out[x * h + col] = row[k];
            }
        }
    }
}

// ============== GEMM wrapper ==============
static inline void unpackC_unit(const float* packC, int e, int h, int unit, float* outRowMajor) {
    // packC layout: [h/unit, e, unit]
    for (int i = 0; i < e; ++i) {
        const float* rowBase = packC + (size_t)i * unit;
        for (int col = 0; col < h; ++col) {
            int ob = col / unit;
            int oi = col % unit;
            outRowMajor[(size_t)i * h + col] = rowBase[ob * (size_t)e * unit + oi];
        }
    }
}

void CPUAttentionGrad::gemmApackedBpacked(const float* packAptr, int e, int l,
                                          const float* packBptr, int h,
                                          GemmCtx& ctx, const CoreFunctions* core) {
    if (e <= 0 || l <= 0 || h <= 0) return;
    const int unit = core->pack;
    const int bytes = core->bytes;
    // parameters: e(tile in bytes), l, h, CStride(bytes), AStride, BStride
    size_t param[7] = {
        (size_t)ctx.eP * (size_t)bytes,
        (size_t)l,
        (size_t)h,
        (size_t)(e * unit * bytes),
        0,0,0
    };
    // C packed layout: [h/unit, e, unit]
    size_t need = (size_t)UP_DIV(h, unit) * (size_t)e * (size_t)unit;
    if (ctx.packedOut.size() < need) ctx.packedOut.resize(need);

    int full = e / ctx.eP;
    int remain = e % ctx.eP;
    // full tiles
    for (int b = 0; b < full; ++b) {
        float* Cblk = ctx.packedOut.data() + (size_t)b * ctx.eP * unit;
        const float* Ablk = packAptr + (size_t)b * l * ctx.eP;
        core->MNNPackedMatMul(Cblk, Ablk, packBptr, param, nullptr, nullptr, nullptr, nullptr);
    }
    // remain tile
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
    // Prepare thread buffers pool
    mThreadBufs.clear();
    mThreadBufs.resize(mThreads);
    for (int t = 0; t < mThreads; ++t) {
        mThreadBufs[t].ctxQK.eP = mEP;
        mThreadBufs[t].ctxDP.eP = mEP;
        mThreadBufs[t].ctxDQ.eP = mEP;
        mThreadBufs[t].ctxDV.eP = mEP;
        mThreadBufs[t].ctxDK.eP = mEP;
        // Pre-reserve reasonable capacities (not exceeding actual need)
        size_t pack = static_cast<CPUBackend*>(backend())->functions()->pack;
        // B packs (per head)
        size_t needBQ = (size_t)UP_DIV(mHeadDim, pack) * (size_t)mSeq * pack;
        mThreadBufs[t].packBQ.reserve(needBQ);
        mThreadBufs[t].packBDY.reserve(needBQ);
        // Contig head slices
        mThreadBufs[t].contigQ.reserve((size_t)mSeq * mHeadDim);
        mThreadBufs[t].contigDY.reserve((size_t)mSeq * mHeadDim);
        // A packs (use max of small and big shapes)
        size_t needASmall = (size_t)UP_DIV(mSeq, mEP) * (size_t)mHeadDim * mEP; // Q or dY
        size_t maxL = std::max(mSeq, mKvSeq);
        size_t maxE = std::max(mSeq, mKvSeq);
        size_t needABig = (size_t)UP_DIV((int)maxE, mEP) * (size_t)maxL * mEP; // P^T/dS^T or dS
        mThreadBufs[t].packA.reserve(needABig);
        mThreadBufs[t].packAT.reserve(needABig);
        (void)needASmall;
    }
    // Force enable dV A^T pack via packATranspose (skip validation)
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
            ::memcpy(Kbuf.data() + j * mHeadDim, Ks, mHeadDim * sizeof(float));
            ::memcpy(Vbuf.data() + j * mHeadDim, Vs, mHeadDim * sizeof(float));
        }
        int unit = core->pack;
        float* packKTHead = mPackKT->host<float>() +
            (size_t)kv_h * UP_DIV(mKvSeq,unit) * mHeadDim * unit;
        float* packKHead  = mPackK ->host<float>() +
            (size_t)kv_h * UP_DIV(mHeadDim,unit) * mKvSeq * unit;
        float* packVTHead = mPackVT->host<float>() +
            (size_t)kv_h * UP_DIV(mKvSeq,unit) * mHeadDim * unit;
        // Debug sanity (optional):
        // fprintf(stderr, "[Grad] packKV kv_h=%d unit=%d\n", kv_h, unit);
        // For Q@K^T and dY@V^T we need B with shape (l x h) = (head_dim x kv_seq),
        // while source is [kv_seq x head_dim], so use transpose=true
        core->MNNPackForMatMul_B(packKTHead, Kbuf.data(), (size_t)mKvSeq, (size_t)mHeadDim, true);
        core->MNNPackForMatMul_B(packVTHead, Vbuf.data(), (size_t)mKvSeq, (size_t)mHeadDim, true);
        // For dQ = dS @ K we need B with (l = kv_seq, h = head_dim), and source Kbuf is [kv_seq x head_dim]
        // which matches (l x h), so transpose=false with (h=head_dim, l=kv_seq)
        core->MNNPackForMatMul_B(packKHead,  Kbuf.data(), (size_t)mHeadDim, (size_t)mKvSeq, false);
    }
}



// ============== Streaming path (unchanged core logic) ==============
// Pack A for the transposed view: given src [E x L] row-major, pack rows of A'=[L x E]
static void packATranspose(const float* src, int E, int L, int eP, std::vector<float>& dst) {
    int blocks = UP_DIV(L, eP); // rows' = L
    dst.resize((size_t)blocks * E * eP);
    for (int b = 0; b < blocks; ++b) {
        int rs = b * eP;
        int realE = std::min(eP, L - rs);
        float* base = dst.data() + (size_t)b * E * eP;
        for (int col = 0; col < E; ++col) {
            for (int r = 0; r < realE; ++r) {
                base[col * eP + r] = src[(size_t)col * L + (rs + r)];
            }
            for (int r = realE; r < eP; ++r) base[col * eP + r] = 0.f;
        }
    }
}

// Gather one head plane into a contiguous [seq x dim] buffer from NHWC-like layout
static void gatherHeadContig(const float* src, int seq, int numHead, int dim, int h, std::vector<float>& dst) {
    dst.resize((size_t)seq * dim);
    for (int i = 0; i < seq; ++i) {
        const float* s = src + (size_t)i * numHead * dim + (size_t)h * dim;
        ::memcpy(dst.data() + (size_t)i * dim, s, (size_t)dim * sizeof(float));
    }
}

// Gather K/V one kv-head into contiguous [kv_seq x dim] buffer from NHWC-like layout
static void gatherKVHeadContig(const float* src, int kvSeq, int kvNumHead, int dim, int kv_h, std::vector<float>& dst) {
    dst.resize((size_t)kvSeq * dim);
    for (int j = 0; j < kvSeq; ++j) {
        const float* s = src + (size_t)j * kvNumHead * dim + (size_t)kv_h * dim;
        ::memcpy(dst.data() + (size_t)j * dim, s, (size_t)dim * sizeof(float));
    }
}

void CPUAttentionGrad::executeStreaming(const Tensor* Q, const Tensor* K, const Tensor* V,
                                        const Tensor* mask, const Tensor* dY,
                                        Tensor* dQ, Tensor* dK, Tensor* dV) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto Qptr  = Q->host<float>();
    auto dYptr = dY->host<float>();
    auto dQptr = dQ->host<float>();
    auto dKptr = dK->host<float>();
    auto dVptr = dV->host<float>();
    // Ensure packed K/V ready
    preparePackedKV(K, V);

    bool hasMask = (mask != nullptr);
    bool maskIsFloat = hasMask && (mask->getType() == halide_type_of<float>());
    const float* maskF = (hasMask && maskIsFloat) ? mask->host<float>() : nullptr;
    const int*   maskI = (hasMask && !maskIsFloat) ? mask->host<int>()   : nullptr;

    const float scale = 1.f / std::sqrt((float)mHeadDim);
    int groupSize = mNumHead / mKvNumHead;
    size_t kvAllSize = (size_t)mKvSeq * mKvNumHead * mHeadDim;

    // thread local accumulators for dK/dV
    std::vector<std::vector<float>> dK_locals(mThreads);
    std::vector<std::vector<float>> dV_locals(mThreads);
    for (int t = 0; t < mThreads; ++t) {
        dK_locals[t].assign(kvAllSize, 0.f);
        dV_locals[t].assign(kvAllSize, 0.f);
    }

    int headsPerThread = UP_DIV(mNumHead, mThreads);

    MNN_CONCURRENCY_BEGIN(tId, mThreads) {
        int hStart = (int)tId * headsPerThread;
        int hEnd   = std::min(hStart + headsPerThread, mNumHead);
        if (hStart >= hEnd) {
            // nothing
        } else {
            auto& dKlocal = dK_locals[(int)tId];
            auto& dVlocal = dV_locals[(int)tId];

            // Reusable buffers per thread
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

            // Loop heads for this thread
            for (int h = hStart; h < hEnd; ++h) {
                int kv_h = h / groupSize;

                // pointers to packed K/V for this kv head
                int unit = core->pack;
                const float* packKTHead = mPackKT->host<float>() +
                    (size_t)kv_h * UP_DIV(mKvSeq,unit) * mHeadDim * unit;
                const float* packKHead  = mPackK ->host<float>() +
                    (size_t)kv_h * UP_DIV(mHeadDim,unit) * mKvSeq * unit;
                const float* packVTHead = mPackVT->host<float>() +
                    (size_t)kv_h * UP_DIV(mKvSeq,unit) * mHeadDim * unit;

                // View slices for this head
                auto& Q_head_c = tbuf.contigQ;
                auto& dY_head_c = tbuf.contigDY;
                Q_head_c.resize((size_t)mSeq * mHeadDim);
                dY_head_c.resize((size_t)mSeq * mHeadDim);
                // Gather into pooled buffers
                for (int i = 0; i < mSeq; ++i) {
                    const float* sQ = Qptr + (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim;
                    ::memcpy(Q_head_c.data() + (size_t)i * mHeadDim, sQ, (size_t)mHeadDim * sizeof(float));
                    const float* sD = dYptr + (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim;
                    ::memcpy(dY_head_c.data() + (size_t)i * mHeadDim, sD, (size_t)mHeadDim * sizeof(float));
                }

                // 1) logits = Q @ K^T via GEMM (after packA fix)
                packA(Q_head_c.data(), mSeq, mHeadDim, mEP, packA_buf);
                gemmApackedBpacked(packA_buf.data(), mSeq, mHeadDim, packKTHead, mKvSeq, ctxQK, core);
                float* P = ctxQK.rowC.data();

                // Apply scale + mask, then softmax row-wise
                std::vector<float> rowBuf(mKvSeq);
                for (int i = 0; i < mSeq; ++i) {
                    float* row = P + (size_t)i * mKvSeq;
                    for (int j = 0; j < mKvSeq; ++j) {
                        float v = row[j] * scale;
                        if (hasMask) {
                            if (maskIsFloat) v += maskF[i * mKvSeq + j];
                            else v = maskI[i * mKvSeq + j] ? v : -1e30f;
                        }
                        rowBuf[j] = v;
                    }
                    MNNSoftmax(row, rowBuf.data(), (size_t)mKvSeq);
                }

                // 2) dP = dY @ V^T via GEMM
                packA(dY_head_c.data(), mSeq, mHeadDim, mEP, packA_buf);
                gemmApackedBpacked(packA_buf.data(), mSeq, mHeadDim, packVTHead, mKvSeq, ctxDP, core);
                float* dP = ctxDP.rowC.data();

                // 3) dS = (dP - (dP ⊙ P)1) ⊙ P ; in-place overwrite dP (use double accumulator for better precision)
                for (int i = 0; i < mSeq; ++i) {
                    float* pRow  = P  + (size_t)i * mKvSeq;
                    float* dpRow = dP + (size_t)i * mKvSeq;
                    double sum_dp_p = 0.0;
                    for (int j = 0; j < mKvSeq; ++j) sum_dp_p += (double)dpRow[j] * (double)pRow[j];
                    float s = (float)sum_dp_p;
                    for (int j = 0; j < mKvSeq; ++j) dpRow[j] = (dpRow[j] - s) * pRow[j];
                }
                float* dS = dP; // alias

                // 4) dQ = (dS * scale) @ K via GEMM (pre-scale dS to reduce rounding error)
                std::vector<float> dS_scaled((size_t)mSeq * mKvSeq);
                for (int i = 0; i < mSeq; ++i) {
                    float* rowS = dS_scaled.data() + (size_t)i * mKvSeq;
                    float* rowD = dP + (size_t)i * mKvSeq; // alias dS
                    for (int j = 0; j < mKvSeq; ++j) rowS[j] = rowD[j] * scale;
                }
                packA(dS_scaled.data(), mSeq, mKvSeq, mEP, packA_buf);
                {
                    gemmApackedBpacked(packA_buf.data(), mSeq, mKvSeq, packKHead, mHeadDim, ctxDQ, core);
                    for (int i = 0; i < mSeq; ++i) {
                        float* src = ctxDQ.rowC.data() + (size_t)i * mHeadDim;
                        float* dst = dQptr + (size_t)i * mNumHead * mHeadDim + (size_t)h * mHeadDim;
                        ::memcpy(dst, src, (size_t)mHeadDim * sizeof(float));
                    }
                }

                // Prepare packed B for dV / dK once
                // Pack B = dY_head (seq x dim) as (l=seq, h=dim), source already (l x h), transpose=false
                {
                    int pack = core->pack;
                    size_t need = (size_t)UP_DIV(mHeadDim, pack) * (size_t)mSeq * pack;
                    if (packB_dY.size() < need) packB_dY.resize(need);
                    core->MNNPackForMatMul_B(packB_dY.data(),
                                              dY_head_c.data(),
                                              (size_t)mHeadDim, (size_t)mSeq, false);
                }
                // Pack B = Q_head (seq x dim) as (l=seq, h=dim)
                {
                    int pack = core->pack;
                    size_t need = (size_t)UP_DIV(mHeadDim, pack) * (size_t)mSeq * pack;
                    if (packB_Q.size() < need) packB_Q.resize(need);
                    core->MNNPackForMatMul_B(packB_Q.data(),
                                              Q_head_c.data(),
                                              (size_t)mHeadDim, (size_t)mSeq, false);
                }

                // 5) dV = P^T @ dY via GEMM（分块 packATranspose 验证后选择）
                if (!mDVPackATChecked && h == 0) {
                    // Validate packATranspose vs explicit transpose on first head only
                    // Path A: packATranspose
                    packATranspose(P, mSeq, mKvSeq, mEP, packA_tr_buf);
                    gemmApackedBpacked(packA_tr_buf.data(), mKvSeq, mSeq, packB_dY.data(), mHeadDim, ctxDV, core);
                    std::vector<float> outA = ctxDV.rowC; // kv_seq x dim
                    // Path B: explicit transpose + packA
                    std::vector<float> P_tr((size_t)mKvSeq * mSeq);
                    for (int i = 0; i < mSeq; ++i) {
                        const float* prow = P + (size_t)i * mKvSeq;
                        for (int j = 0; j < mKvSeq; ++j) P_tr[(size_t)j * mSeq + i] = prow[j];
                    }
                    packA(P_tr.data(), mKvSeq, mSeq, mEP, packA_tr_buf);
                    gemmApackedBpacked(packA_tr_buf.data(), mKvSeq, mSeq, packB_dY.data(), mHeadDim, ctxDV, core);
                    std::vector<float> outB = ctxDV.rowC;
                    double maxDiff = 0.0;
                    size_t ncmp = (size_t)mKvSeq * mHeadDim;
                    for (size_t ii = 0; ii < ncmp; ++ii) {
                        double d = std::abs(outA[ii] - outB[ii]);
                        if (d > maxDiff) maxDiff = d;
                    }
                    mDVPackATUsePackAT = (maxDiff < 1e-5);
                    mDVPackATChecked = true;
                    // Accumulate stable result (outB) this time
                    for (int j = 0; j < mKvSeq; ++j) {
                        float* dstV = dVlocal.data() + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                        const float* src = outB.data() + (size_t)j * mHeadDim;
                        for (int d = 0; d < mHeadDim; ++d) dstV[d] += src[d];
                    }
                } else {
                    if (mDVPackATUsePackAT) {
                        packATranspose(P, mSeq, mKvSeq, mEP, packA_tr_buf);
                        gemmApackedBpacked(packA_tr_buf.data(), mKvSeq, mSeq, packB_dY.data(), mHeadDim, ctxDV, core);
                        for (int j = 0; j < mKvSeq; ++j) {
                            float* dstV = dVlocal.data() + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                            const float* src = ctxDV.rowC.data() + (size_t)j * mHeadDim;
                            for (int d = 0; d < mHeadDim; ++d) dstV[d] += src[d];
                        }
                    } else {
                        std::vector<float> P_tr((size_t)mKvSeq * mSeq);
                        for (int i = 0; i < mSeq; ++i) {
                            const float* prow = P + (size_t)i * mKvSeq;
                            for (int j = 0; j < mKvSeq; ++j) P_tr[(size_t)j * mSeq + i] = prow[j];
                        }
                        packA(P_tr.data(), mKvSeq, mSeq, mEP, packA_tr_buf);
                        gemmApackedBpacked(packA_tr_buf.data(), mKvSeq, mSeq, packB_dY.data(), mHeadDim, ctxDV, core);
                        for (int j = 0; j < mKvSeq; ++j) {
                            float* dstV = dVlocal.data() + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                            const float* src = ctxDV.rowC.data() + (size_t)j * mHeadDim;
                            for (int d = 0; d < mHeadDim; ++d) dstV[d] += src[d];
                        }
                    }
                }

                // 6) dK = dS^T @ Q via GEMM（分块 packATranspose）
                packATranspose(dS, mSeq, mKvSeq, mEP, packA_tr_buf);
                {
                    gemmApackedBpacked(packA_tr_buf.data(), mKvSeq, mSeq, packB_Q.data(), mHeadDim, ctxDK, core);
                    for (int j = 0; j < mKvSeq; ++j) {
                        float* dstK = dKlocal.data() + (size_t)j * mKvNumHead * mHeadDim + (size_t)kv_h * mHeadDim;
                        const float* src = ctxDK.rowC.data() + (size_t)j * mHeadDim;
                        for (int d = 0; d < mHeadDim; ++d) dstK[d] += src[d] * scale;
                    }
                }
            } // for h
        }
    }
    MNN_CONCURRENCY_END();

    // reduction to global dK/dV
    size_t kvAll = (size_t)mKvSeq * mKvNumHead * mHeadDim;
    for (int t = 0; t < mThreads; ++t) {
        const auto& dk = dK_locals[t];
        const auto& dv = dV_locals[t];
        for (size_t i = 0; i < kvAll; ++i) {
            dKptr[i] += dk[i];
            dVptr[i] += dv[i];
        }
    }
}


// ============== onExecute ==============
ErrorCode CPUAttentionGrad::onExecute(const std::vector<Tensor*>& inputs,
                                      const std::vector<Tensor*>& outputs) {
    // debug entry
    // fprintf(stderr, "[Grad] onExecute enter, inputs=%zu outputs=%zu\n", inputs.size(), outputs.size()); fflush(stderr);
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


    // fprintf(stderr, "[Grad] call executeStreaming\n"); fflush(stderr);
    executeStreaming(Q, K, V, mask, dY, dQ, dK, dV);
    // fprintf(stderr, "[Grad] onExecute exit\n"); fflush(stderr);

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
