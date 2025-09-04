//
//  CPUAttentionGrad.hpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef CPUATTENTIONGRAD_HPP
#define CPUATTENTIONGRAD_HPP

#include "core/Execution.hpp"
#include "MNN/ErrorCode.hpp"

namespace MNN {

class CPUAttentionGrad : public Execution {
public:
    explicit CPUAttentionGrad(Backend* backend);
    virtual ~CPUAttentionGrad();
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    int mThreadNum = 1;
    int eP = 0, lP = 0, hP = 0, unit = 0;
    int bytes = 4;
    int mNumHead = 0, mHeadDim = 0, mKvNumHead = 0;
};

} // namespace MNN

#endif // CPUATTENTIONGRAD_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
