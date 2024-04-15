#include "rosneuro_decoder_qda/Qda.hpp"
#include <fstream>
#include <ros/package.h>
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

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> readCSV(const std::string& filename) {
            std::vector<double> values;
            std::ifstream file(filename);
            std::string row;
            std::string entry;
            int n_rows = 0;

            while (getline(file, row)) {
                std::stringstream rowstream(row);
                while (getline(rowstream, entry, ',')) {
                    values.push_back(std::stod(entry));
                }
                n_rows++;
            }

            return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), n_rows, values.size() / n_rows);

        }

        void config_qda(std::map<std::string, XmlRpc::XmlRpcValue>& params) {
            params["name"] = "qda";
            params["filename"] = "file1";
            params["subject"] = "s1";
            params["n_classes"] = 2;
            params["n_features"] = 2;
            params["lambda"] = 0.5;

            XmlRpc::XmlRpcValue priors, class_lbs, idchans;
            priors[0] = 0.5;
            priors[1] = 0.5;
            params["priors"] = priors;

            class_lbs[0] = 771;
            class_lbs[1] = 773;
            params["class_lbs"] = class_lbs;

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
            EXPECT_EQ(qda->getName(), "qda");
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

        TEST_F(QdaTestSuite, Integration) {
            config_qda(qda->params_);

            std::string package_path = ros::package::getPath("rosneuro_decoder_qda");
            const std::string file_input  = package_path + "/test/data/features.csv";
            const std::string file_output = package_path + "/test/data/output.csv";

            Eigen::MatrixXd input = readCSV(file_input);
            Eigen::MatrixXd output = readCSV(file_output);

            ASSERT_TRUE(qda->configure());

            for(int i = 0; i < input.rows(); i++){
                Eigen::VectorXf vector_casted = input.row(i).cast<float>();
                Eigen::VectorXf result = qda->apply(vector_casted);
                Eigen::VectorXf expected = output.row(i).cast<float>();
                ASSERT_TRUE(result.isApprox(expected, 1e-5));
            }
        }
    }
}

int main(int argc, char **argv) {
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Fatal);
    ros::init(argc, argv, "test_qda");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}