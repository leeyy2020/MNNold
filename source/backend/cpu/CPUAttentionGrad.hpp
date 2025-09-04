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
    int mThreadNum = 1;
    int bytes = 4;
    int eP, lP, hP, unit; // float matmul packing
    int eP8, lP8, hP8;    // GemmInt8 packing
    int mNumHead, mKvNumHead, mHeadDim;
};

} // namespace MNN

#endif // CPUATTENTIONGRAD_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE