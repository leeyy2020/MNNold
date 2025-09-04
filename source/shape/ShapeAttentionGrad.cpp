//
//  ShapeAttentionGrad.cpp
//  MNN
//
//  Created by MNN on 2025/08/30.
//
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

class AttentionGradSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() >= 4);
        MNN_ASSERT(outputs.size() == 3);
        auto q = inputs[0];
        auto k = inputs[1];
        auto v = inputs[2];
        auto qg = outputs[0];
        auto kg = outputs[1];
        auto vg = outputs[2];

        // query_grad shape = query shape (3 dims: [seq_len, num_head, head_dim])
        qg->buffer().dimensions = q->buffer().dimensions;
        for (int i = 0; i < q->buffer().dimensions; ++i) {
            qg->buffer().dim[i].extent = q->buffer().dim[i].extent;
        }
        qg->buffer().type = q->buffer().type;
        TensorUtils::getDescribe(qg)->dimensionFormat = TensorUtils::getDescribe(q)->dimensionFormat;

        // key_grad shape = key shape (3 dims: [kv_seq_len, kv_num_head, head_dim])
        kg->buffer().dimensions = k->buffer().dimensions;
        for (int i = 0; i < k->buffer().dimensions; ++i) {
            kg->buffer().dim[i].extent = k->buffer().dim[i].extent;
        }
        kg->buffer().type = k->buffer().type;
        TensorUtils::getDescribe(kg)->dimensionFormat = TensorUtils::getDescribe(k)->dimensionFormat;

        // value_grad shape = value shape
        vg->buffer().dimensions = v->buffer().dimensions;
        for (int i = 0; i < v->buffer().dimensions; ++i) {
            vg->buffer().dim[i].extent = v->buffer().dim[i].extent;
        }
        vg->buffer().type = v->buffer().type;
        TensorUtils::getDescribe(vg)->dimensionFormat = TensorUtils::getDescribe(v)->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE(AttentionGradSizeComputer, OpType_AttentionGrad);
#endif
} // namespace MNN
