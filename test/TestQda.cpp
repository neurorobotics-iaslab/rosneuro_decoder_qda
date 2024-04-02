#include "rosneuro_decoder_qda/Qda.hpp"
#include <gtest/gtest.h>

namespace rosneuro {
namespace decoder {

class QdaTestSuite : public ::testing::Test {
public:
    QdaTestSuite() {}
    ~QdaTestSuite() {}
    void SetUp() { qda = new Qda(); }
    void TearDown() { delete qda; }
    Qda* qda;
};

void config_qda(std::map<std::string, XmlRpc::XmlRpcValue>& params) {
    params["name"] = "qda";
    params["filename"] = "file1";
    params["subject"] = "s1";
    params["nclasses"] = 2;
    params["nfeatures"] = 2;
    params["lambda"] = 0.5;

    XmlRpc::XmlRpcValue priors, classlbs, idchans;
    priors[0] = 0.5;
    priors[1] = 0.5;
    params["priors"] = priors;

    classlbs[0] = 771;
    classlbs[1] = 773;
    params["classlbs"] = classlbs;

    idchans[0] = 1;
    idchans[1] = 2;
    params["idchans"] = idchans;

    params["freqs"] = "10; 20;";
    params["means"] = "0.4948 3.4821;"
                      "0.4647 3.4921;";

    params["covs"] = "0.9273  0.9325;"
                     "-0.0187 -0.0144;"
                     "-0.0187 -0.0144;"
                     "0.9120  0.8959;";
}

TEST_F(QdaTestSuite, Constructor) {
    EXPECT_EQ(qda->name(), "qda");
    EXPECT_EQ(qda->is_configured_, false);
}

TEST_F(QdaTestSuite, Configure) {
    config_qda(qda->params_);
    ASSERT_TRUE(qda->configure());
}

TEST_F(QdaTestSuite, CheckDimensionMeans) {
    config_qda(qda->params_);
    qda->params_["means"] = "1 2;";
    ASSERT_FALSE(qda->configure());
}

TEST_F(QdaTestSuite, CheckDimensionCovs) {
    config_qda(qda->params_);
    qda->params_["covs"] = "1 2;";
    ASSERT_FALSE(qda->configure());
}

TEST_F(QdaTestSuite, CheckDimensionSize) {
    config_qda(qda->params_);
    XmlRpc::XmlRpcValue priors;
    priors[0] = 0.5;
    qda->params_["priors"] = priors;
    ASSERT_FALSE(qda->configure());
}

TEST_F(QdaTestSuite, Apply) {
    config_qda(qda->params_);
    qda->configure();

    Eigen::VectorXf in(5);
    in << 1, 2, 3, 4, 5;

    Eigen::VectorXf expected(2);
    expected << 0.9591, 0.0408;

    ASSERT_TRUE(qda->apply(in).isApprox(expected, 0.001));
}

}
}

int main(int argc, char **argv) {
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Fatal);
    ros::init(argc, argv, "test_qda");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}