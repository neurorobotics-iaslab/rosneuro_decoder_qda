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
        	std::uint32_t		   nclasses;
            double                 lambda;

            std::vector<double>     priors;
        	std::vector<uint32_t>  classlbs;
        	std::uint32_t      	   nfeatures;

            // for features extraction
        	std::vector<uint32_t>               idchans;
        	std::vector<std::vector<uint32_t>>  freqs;

        } qdaconfig_t;

        class Qda : public GenericDecoder{
            public:
                Qda(void);
                ~Qda(void);

                bool configure(void);
                bool isSet(void);
                Eigen::VectorXf apply(const Eigen::VectorXf& in);
                Eigen::VectorXf getFeatures(const Eigen::MatrixXf& in);

                std::string path(void);
                std::vector<int> classes(void);

            private:
                bool check_dimension(void);
                Eigen::MatrixXf rebuild_cov(const Eigen::MatrixXf& in);

            private:
                ros::NodeHandle p_nh_;
                Eigen::MatrixXf means_;
                Eigen::MatrixXf covs_;
                qdaconfig_t config_;
        };

    }
}

#endif