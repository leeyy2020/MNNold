//
//  AttentionGrad.cpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "AttentionGrad.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
using namespace std;
namespace MNN {
using namespace MNN::Express;

class AttentionGrad : public OpGrad {
public:
    AttentionGrad() {
        mType = NO_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        MNN_ASSERT(backwardOutput.size() == 1);

        auto inputs = expr->inputs();
        MNN_ASSERT(inputs.size() >= 3); // query, key, value at minimum

        auto query = inputs[0];
        auto key = inputs[1];
        auto value = inputs[2];
        auto output_grad = backwardOutput[0];

        // Optional inputs
        VARP mask = nullptr;
        VARP sinks = nullptr;
        if (inputs.size() > 3) {
            mask = inputs[3];
        }
        if (inputs.size() > 4) {
            sinks = inputs[4];
        }

        // Call the attention gradient function from MathOp.cpp
        // This function will create an AttentionGrad operation that calls the CPUAttentionGrad backend
        auto query_grad = _AttentionGrad(query, key, value, output_grad, mask, sinks);

        // For now, return only query gradient
        // In a complete implementation, this would return gradients for all inputs
        return {query_grad};
    }
};

static void _create() {
    static AttentionGrad _c;
    OpGrad::insert(OpType_Attention, &_c);
}

REGISTER_GRAD(AttentionGrad_cpp, _create);
};