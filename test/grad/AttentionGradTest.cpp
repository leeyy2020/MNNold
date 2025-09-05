#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "../tools/train/source/grad/OpGrad.hpp"

using namespace MNN;
using namespace MNN::Express;

class AttentionGradTest : public MNNTestCase {
public:
    AttentionGradTest() { OpGrad::init(); }
    virtual ~AttentionGradTest() = default;
    virtual bool run(int precision) {
        std::vector<float> qData = {0.1f, 0.2f, 0.3f, 0.4f};
        std::vector<float> kData = {0.5f, 0.6f, 0.7f, 0.8f};
        std::vector<float> vData = {0.9f, 1.0f, 1.1f, 1.2f};
        std::vector<float> dyData = {0.01f, 0.02f, 0.03f, 0.04f};
        auto q = _Const(qData.data(), {1, 2, 1, 2}, NCHW);
        auto k = _Const(kData.data(), {1, 2, 1, 2}, NCHW);
        auto v = _Const(vData.data(), {1, 2, 1, 2}, NCHW);
        auto dy = _Const(dyData.data(), {1, 2, 1, 2}, NCHW);

        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_AttentionGrad;
        op->main.type = OpParameter_NONE;
        op->main.value = nullptr;
        auto expr = Expr::create(std::move(op), {q, k, v, dy}, 3);
        auto dq = Variable::create(expr, 0);
        auto dk = Variable::create(expr, 1);
        auto dv = Variable::create(expr, 2);

        auto dqPtr = dq->readMap<float>();
        auto dkPtr = dk->readMap<float>();
        auto dvPtr = dv->readMap<float>();

        std::vector<float> expectDQ = {2.12037e-04f, 2.12037e-04f, 4.93764e-04f, 4.93764e-04f};
        std::vector<float> expectDK = {-8.46664e-04f, -1.19956e-03f, 8.46664e-04f, 1.19956e-03f};
        std::vector<float> expectDV = {1.91521e-02f, 2.87988e-02f, 2.08479e-02f, 3.12012e-02f};

        const float thres = 1e-6f;
        for (int i = 0; i < 4; ++i) {
            if (fabs(dqPtr[i] - expectDQ[i]) > thres) {
                MNN_ERROR("dQ mismatch %d: %f vs %f\n", i, dqPtr[i], expectDQ[i]);
                return false;
            }
            if (fabs(dkPtr[i] - expectDK[i]) > thres) {
                MNN_ERROR("dK mismatch %d: %f vs %f\n", i, dkPtr[i], expectDK[i]);
                return false;
            }
            if (fabs(dvPtr[i] - expectDV[i]) > thres) {
                MNN_ERROR("dV mismatch %d: %f vs %f\n", i, dvPtr[i], expectDV[i]);
                return false;
            }
        }
        return true;
    }
};

MNNTestSuiteRegister(AttentionGradTest, "grad/attention");
