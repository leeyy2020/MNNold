//
//  CPUAttentionGrad.hpp
//  MNN
//
//  Attention Backward (CPU, fp32)
//  - Streaming / Head GEMM / Group GEMM execution paths
//  - Cached packed K / V (packKT / packK / packVT)
//  NOTE:
//    * Only batch = 1 supported currently
//    * Training must disable forward kv_cache for correct gradients
//    * Float32 only
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#ifndef CPUATTENTIONGRAD_HPP
#define CPUATTENTIONGRAD_HPP

#include <memory>
#include <vector>
#include "core/Execution.hpp"
#include "MNN/ErrorCode.hpp"
// Need CoreFunctions declaration
#include "compute/CommonOptFunction.h"   // <-- Fix: brings in struct CoreFunctions

namespace MNN {

// (If you prefer to avoid the include above, you could instead forward declare:
// struct CoreFunctions;
// but including the header is safer in case of conditional members.)

class CPUAttentionGrad : public Execution {
public:
    CPUAttentionGrad(Backend *backend);
    ~CPUAttentionGrad() override;

    ErrorCode onResize(const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs) override;
    ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                        const std::vector<Tensor *> &outputs) override;
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    // pack parameters
    int mEP = 4, mLP = 1, mHP = 4;
    int mBytes = 4;

    // runtime / shape
    int mThreads = 1;
    int mBatch = 1;
    int mSeq = 0, mKvSeq = 0;
    int mNumHead = 0, mKvNumHead = 0;
    int mHeadDim = 0;

    // Packed K / V caches
    // packKT: K^T  (h = kvSeq,   l = headDim, transpose=true)
    // packK : K    (h = headDim, l = kvSeq,   transpose=false)
    // packVT: V^T  (h = kvSeq,   l = headDim, transpose=true)
    std::shared_ptr<Tensor> mPackKT;
    std::shared_ptr<Tensor> mPackK;
    std::shared_ptr<Tensor> mPackVT;
    bool mPackAllocated = false;

    void allocPackedKV();                       // allocate once (onResize)
    void preparePackedKV(const Tensor* K, const Tensor* V); // fill each execution

    // Execution path enum
    enum class ExecPath {
        Streaming,
        HeadGemm,
        GroupGemm
    };

    // Paths
    void executeStreaming(const Tensor* Q, const Tensor* K, const Tensor* V,
                          const Tensor* mask, const Tensor* dY,
                          Tensor* dQ, Tensor* dK, Tensor* dV);

    // Utilities
    static void packA(const float* src, int e, int l, int eP, std::vector<float>& dst);
    struct GemmCtx {
        int eP;
        std::vector<float> packedOut; // hC4 * e * 4
        std::vector<float> rowC;      // e * h
    };
    static void gemmApackedBpacked(const float* packAptr, int e, int l,
                                   const float* packBptr, int h,
                                   GemmCtx& ctx, const CoreFunctions* core);

    // Thresholds (tunable)
    static constexpr int64_t kHeadGemmThreshold  = 32 * 1024;   // seq * kv * dim
    static constexpr int64_t kGroupGemmThreshold = 96 * 1024;   // seq * kv * dim (and groupSize > 1)

    // Thread-local pooled buffers to reduce realloc / memset
    struct ThreadBuf {
        // Packed A buffers
        std::vector<float> packA;
        std::vector<float> packAT;
        // Packed B buffers for Q and dY
        std::vector<float> packBQ;
        std::vector<float> packBDY;
        // Contiguous head slices
        std::vector<float> contigQ;
        std::vector<float> contigDY;
        // GEMM contexts (packed C and row-major C)
        GemmCtx ctxQK{4};
        GemmCtx ctxDP{4};
        GemmCtx ctxDQ{4};
        GemmCtx ctxDV{4};
        GemmCtx ctxDK{4};
    };
    std::vector<ThreadBuf> mThreadBufs;

    // Heuristic switch for dV A^T packing path (validated once per resize)
    bool mDVPackATChecked = false;
    bool mDVPackATUsePackAT = false;
};

} // namespace MNN
#endif // CPUATTENTIONGRAD_HPP
#endif // MNN_SUPPORT_TRANSFORMER_FUSE
