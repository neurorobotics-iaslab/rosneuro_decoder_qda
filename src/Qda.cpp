#ifndef ROSNEURO_DECODER_QDA_CPP
#define ROSNEURO_DECODER_QDA_CPP

#include "rosneuro_decoder_qda/Qda.hpp"

namespace rosneuro{
    namespace decoder{

        Qda::Qda(void) : p_nh_("~"){
            this->setName("qda");
            this->is_configured_ = false;
        }

        Qda::~Qda(void){}

        template<typename T>
        bool Qda::getParamAndCheck(const std::string& param_name, T& param_value) {
            if (!GenericDecoder::getParam(param_name, param_value)) {
                ROS_ERROR("[%s] Cannot find param '%s'", this->getName().c_str(), param_name.c_str());
                return false;
            }
            return true;
        }

        bool Qda::configure(void){
            if (!getParamAndCheck("filename", this->config_.filename)) return false;
            if (!getParamAndCheck("subject", this->config_.subject)) return false;
            if (!getParamAndCheck("n_classes", this->config_.n_classes)) return false;
            if (!getParamAndCheck("class_lbs", this->config_.class_lbs)) return false;
            if (!getParamAndCheck("n_features", this->config_.n_features)) return false;
            if (!getParamAndCheck("lambda", this->config_.lambda)) return false;
            if (!getParamAndCheck("idchans", this->config_.idchans)) return false;
            if (!getParamAndCheck("priors", this->config_.priors)) return false;

            std::string freqs_str, means_str, covs_str;
            if (!getParamAndCheck("freqs", freqs_str)) return false;
            if (!getParamAndCheck("means", means_str)) return false;
            if (!getParamAndCheck("covs", covs_str)) return false;


            if(!this->loadVectorOfVector(freqs_str, this->config_.freqs)){
                ROS_ERROR("[%s] Cannot convert param 'freqs' to vctor of vector", this->getName().c_str());
                return false;
            }

            this->means_ = Eigen::MatrixXf::Zero(this->config_.n_features, this->config_.n_classes);
            if(!this->loadEigen(means_str, this->means_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for means", this->getName().c_str());
                return false;
            }

            this->covs_ = Eigen::MatrixXf::Zero(this->config_.n_features * this->config_.n_features, this->config_.n_classes);
            if(!this->loadEigen(covs_str, this->covs_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for covs", this->getName().c_str());
                return false;
            }

            if(!this->checkDimension()){
                ROS_ERROR("[%s] Error in the dimension", this->getName().c_str());
                return false;
            }

            this->is_configured_ = true;
            return this->is_configured_;
        }

        bool Qda::isSet(void){
            if(!this->is_configured_){
                ROS_ERROR("[%s] Decoder not configured", this->getName().c_str());
                return false;
            }
            return this->is_configured_;
        }

        Eigen::VectorXf Qda::apply(const Eigen::VectorXf& in) {
            std::vector<double> likelihoods;
            double posterior_denominator = calculateLikelihoods(in, likelihoods);

            Eigen::VectorXf posterior_probabilities = computePosteriorProbabilities(likelihoods, posterior_denominator);

            return posterior_probabilities;
        }

        double Qda::calculateLikelihoods(const Eigen::VectorXf& input, std::vector<double>& likelihoods){
            double denominator = 0.0;
            for (int i = 0; i < this->config_.n_classes; i++) {
                Eigen::MatrixXf class_covariance = rebuildCovariance(this->covs_.col(i));
                double coefficient = calculateCoefficient(input.size(), class_covariance);
                double exponent = calculateExponent(input, i, class_covariance);
                double class_likelihood = coefficient * std::exp(exponent);

                likelihoods.push_back(class_likelihood);
                denominator += class_likelihood * this->config_.priors.at(i);
            }
            return denominator;
        }

        double Qda::calculateCoefficient(int input_size, const Eigen::MatrixXf& covariance) const {
            double coefficient = 1 / (std::sqrt((std::pow(2.0 * M_PI, input_size)) * covariance.determinant()));
            return coefficient;
        }

        double Qda::calculateExponent(const Eigen::VectorXf& input, int class_index, const Eigen::MatrixXf& covariance) const {
            Eigen::VectorXf mean_difference = input - this->means_.col(class_index);
            double exponent = -0.5 * ((mean_difference.transpose() * covariance.inverse() * mean_difference)(0, 0));
            return exponent;
        }

        Eigen::VectorXf Qda::computePosteriorProbabilities(const std::vector<double>& likelihoods, double posterior_denominator) const {
            Eigen::VectorXf posterior_probabilities(likelihoods.size(), 1);
            for (int i = 0; i < likelihoods.size(); i++) {
                double posterior = (likelihoods.at(i) * this->config_.priors.at(i)) / posterior_denominator;
                posterior_probabilities(i, 0) = posterior;
            }
            return posterior_probabilities;
        }

        Eigen::MatrixXf Qda::rebuildCovariance(const Eigen::MatrixXf& in) {
            if (in.size() != this->config_.n_features * this->config_.n_features) {
                ROS_ERROR("[%s] Wrong dimension in the covariance", this->getName().c_str());
            }

            Eigen::MatrixXf covariance_matrix(this->config_.n_features, this->config_.n_features);
            int index = 0;
            for (int i = 0; i < this->config_.n_features; i++) {
                for (int j = 0; j < this->config_.n_features; j++) {
                    covariance_matrix(j, i) = in(index, 0);
                    index++;
                }
            }
            return covariance_matrix;
        }

        std::string Qda::getPath(){
            this->isSet();
            return this->config_.filename;
        }

        std::vector<int> Qda::getClasses(void){
            this->isSet();
            std::vector<int> classes_lbs;
            for(int i = 0; i < this->config_.class_lbs.size(); i++){
                classes_lbs.push_back((int) this->config_.class_lbs.at(i));
            }
            return classes_lbs;
        }

        Eigen::VectorXf Qda::getFeatures(const Eigen::MatrixXf& in){
            Eigen::VectorXf out(this->config_.n_features);
            this->isSet();

            int c_feature = 0;
            for(int it_chan = 0; it_chan < this->config_.idchans.size(); it_chan++){
                int idchan = this->config_.idchans.at(it_chan) - 1;
                for(const auto& freq : this->config_.freqs.at(it_chan)){
                    int idfreq = (int) freq/2.0;
                    out(c_feature) = in(idchan, idfreq);
                    c_feature ++;
                }
            }

            return out.transpose();
        }

        bool Qda::checkDimension(void){
            if(this->means_.rows() != this->config_.n_features ||
               this->means_.cols() != this->config_.n_classes){
                ROS_ERROR("[%s] Wrong dimensions in the 'means' parameter", this->getName().c_str());
                return false;
            }

            if(this->covs_.rows() != this->config_.n_features * this->config_.n_features ||
               this->covs_.cols() != this->config_.n_classes){
                ROS_ERROR("[%s] Wrong dimensions in the 'covs' parameter", this->getName().c_str());
                return false;
            }

            if(this->config_.priors.size() != this->config_.n_classes |\
               this->config_.n_classes != this->config_.class_lbs.size()){
                ROS_ERROR("[%s] Wrong dimensions in the given classes parameters", this->getName().c_str());
                return false;
            }

            int sum = 0;
            for(int i = 0; i < this->config_.freqs.size(); i++){
                std::vector<uint32_t> temp = this->config_.freqs.at(i);
                sum = sum + temp.size();
            } 
            if(sum != this->config_.n_features){
                ROS_ERROR("[%s] Incorrect dimension for 'freqs' different from 'n_features'", this->getName().c_str());
                return false;
            }

            return true;
        }

        PLUGINLIB_EXPORT_CLASS(rosneuro::decoder::Qda, rosneuro::decoder::GenericDecoder);
    }
}

#endif