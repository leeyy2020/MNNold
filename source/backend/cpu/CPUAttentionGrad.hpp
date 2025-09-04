//
//  CPUAttentionGrad.hpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef CPUATTENTIONGRAD_HPP
#define CPUATTENTIONGRAD_HPP

#include <functional>
#include "core/Execution.hpp"
#include "core/OpCommonUtils.hpp"
#include "MNN/ErrorCode.hpp"

namespace MNN {

class CPUAttentionGrad : public Execution {
public:
    CPUAttentionGrad(Backend *backend);
    virtual ~CPUAttentionGrad();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    int mEP = 4, mLP = 1, mHP = 4, mBytes = 4;
    int mThreads = 1;
    int mSeq = 0, mKvSeq = 0, mNumHead = 0, mKvNumHead = 0, mHeadDim = 0;
    int mBatch = 1;
};

} // namespace MNN

#endif // CPUATTENTIONGRAD_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
