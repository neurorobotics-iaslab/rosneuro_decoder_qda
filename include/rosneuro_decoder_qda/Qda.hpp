#ifndef ROSNEURO_DECODER_QDA_HPP
#define ROSNEURO_DECODER_QDA_HPP

#include <pluginlib/class_list_macros.h>
#include <dynamic_reconfigure/server.h>
#include <ros/ros.h>
#include "rosneuro_decoder/GenericDecoder.h"

namespace rosneuro{
    namespace decoder{
        typedef struct {
        	std::string 		   filename;
        	std::string		       subject;
        	std::uint32_t		   n_classes;
            double                 lambda;

            std::vector<double>     priors;
        	std::vector<uint32_t>  class_lbs;
        	std::uint32_t      	   n_features;

        	std::vector<uint32_t>               idchans;
        	std::vector<std::vector<uint32_t>>  freqs;

        } qda_configuration;

        class Qda : public GenericDecoder{
            public:
                Qda(void);
                ~Qda(void);

                bool configure(void);
                bool isSet(void);
                Eigen::VectorXf apply(const Eigen::VectorXf& in);
                Eigen::VectorXf getFeatures(const Eigen::MatrixXf& in);

                std::string getPath(void);
                std::vector<int> getClasses(void);

            private:
                Eigen::VectorXf computePosteriorProbabilities(const std::vector<double>& likelihoods, double posterior_denominator) const;
                double calculateExponent(const Eigen::VectorXf& input, int class_index, const Eigen::MatrixXf& covariance) const;
                double calculateCoefficient(int input_size, const Eigen::MatrixXf& covariance) const;
                double calculateLikelihoods(const Eigen::VectorXf& input, std::vector<double>& likelihoods);

                template<typename T>
                bool getParamAndCheck(const std::string& param_name, T& param_value);
                bool checkDimension(void);
                Eigen::MatrixXf rebuildCovariance(const Eigen::MatrixXf& in);

                ros::NodeHandle p_nh_;
                Eigen::MatrixXf means_;
                Eigen::MatrixXf covs_;
                qda_configuration config_;

                FRIEND_TEST(QdaTestSuite, Constructor);
                FRIEND_TEST(QdaTestSuite, Configure);
                FRIEND_TEST(QdaTestSuite, CheckDimensionMeans);
                FRIEND_TEST(QdaTestSuite, CheckDimensionCovs);
                FRIEND_TEST(QdaTestSuite, CheckDimensionSize);
                FRIEND_TEST(QdaTestSuite, Apply);
                FRIEND_TEST(QdaTestSuite, Integration);
        };

    }
}

#endif