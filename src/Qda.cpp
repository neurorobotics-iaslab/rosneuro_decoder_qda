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

        bool Qda::configure(void){
            if(!GenericDecoder::getParam(std::string("filename"), this->config_.filename)){
                ROS_ERROR("[%s] Cannot find param 'filename'", this->getName().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("subject"), this->config_.subject)){
                ROS_ERROR("[%s] Cannot find param 'subject'", this->getName().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("n_classes"), this->config_.n_classes)){
                ROS_ERROR("[%s] Cannot find param 'n_classes'", this->getName().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("class_lbs"), this->config_.class_lbs)){
                ROS_ERROR("[%s] Cannot find param 'class_lbs'", this->getName().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("n_features"), this->config_.n_features)){
                ROS_ERROR("[%s] Cannot find param 'n_features'", this->getName().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("lambda"), this->config_.lambda)){
                ROS_ERROR("[%s] Cannot find param 'lambda'", this->getName().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("idchans"), this->config_.idchans)){
                ROS_ERROR("[%s] Cannot find param 'idchans'", this->getName().c_str());
                return false;
            }
            std::string freqs_str;
            if(!GenericDecoder::getParam(std::string("freqs"), freqs_str)){
                ROS_ERROR("[%s] Cannot find param 'freqs'", this->getName().c_str());
                return false;
            }
            if(!this->loadVectorOfVector(freqs_str, this->config_.freqs)){
                ROS_ERROR("[%s] Cannot convert param 'freqs' to vctor of vector", this->getName().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("priors"), this->config_.priors)){
                ROS_ERROR("[%s] Cannot find param 'priors'", this->getName().c_str());
                return false;
            }
            std::string means_str, covs_str;
            if(!GenericDecoder::getParam(std::string("means"), means_str)){
                ROS_ERROR("[%s] Cannot find param 'means'", this->getName().c_str());
                return false;
            }
            this->means_ = Eigen::MatrixXf::Zero(this->config_.n_features, this->config_.n_classes);
            if(!this->loadEigen(means_str, this->means_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for means", this->getName().c_str());
                return false;
            }
            if(!GenericDecoder::getParam(std::string("covs"), covs_str)){
                ROS_ERROR("[%s] Cannot find param 'covs'", this->getName().c_str());
                return false;
            }
            this->covs_ = Eigen::MatrixXf::Zero(this->config_.n_features * this->config_.n_features, this->config_.n_classes);
            if(!this->loadEigen(covs_str, this->covs_)){
                ROS_ERROR("[%s] Failed to load eigen matrix for covs", this->getName().c_str());
                return false;
            }

            // fast check for the correct dimension for the features
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

        Eigen::VectorXf Qda::apply(const Eigen::VectorXf& in){

            std::vector<double> lh;
            double den = 0.0;
            for(int i = 0; i < this->config_.n_classes; i++){
                Eigen::MatrixXf c_cov = this->rebuildCovariance(this->covs_.col(i));
                double c_coeff = 1/(std::sqrt((std::pow(2.0 * M_PI, in.size()))* c_cov.determinant()));
                double c_exp = -0.5 * ((in - this->means_.col(i)).transpose() * c_cov.inverse() * (in - this->means_.col(i)))(0,0);
                double c_lh = c_coeff * std::exp(c_exp);

                lh.push_back(c_lh);

                den = den + c_lh * this->config_.priors.at(i);
            }

            // Compute the posterior probability
            Eigen::VectorXf output(lh.size(), 1);
            for(int i = 0; i < this->config_.n_classes; i++){
                double c_post = (lh.at(i) * this->config_.priors.at(i)) / den;
                output(i,0) = c_post;
            }

            return output;
        }

        Eigen::MatrixXf Qda::rebuildCovariance(const Eigen::MatrixXf& in){
            // check the dimensions
            if(in.size() != this->config_.n_features * this->config_.n_features){
                ROS_ERROR("[%s] Wrong dimension in the covariance", this->getName().c_str());
            }

            Eigen::MatrixXf out(this->config_.n_features, this->config_.n_features);
            int cont = 0;
            for(int i = 0; i < this->config_.n_features; i++){
                for(int j = 0; j < this->config_.n_features; j++){
                    out(j,i) = in(cont, 0);
                    cont ++;
                }
            }

            return out;
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
            // check if set and prepare for the correct dimension
            Eigen::VectorXf out(this->config_.n_features);
            this->isSet();

            // iterate over channels
            int c_feature = 0;
            for(int it_chan = 0; it_chan < this->config_.idchans.size(); it_chan++){
                int idchan = this->config_.idchans.at(it_chan) - 1; // -1 bc: channels starts from 1 and not 0
                // iterate over freqs for that channel
                for(const auto& freq : this->config_.freqs.at(it_chan)){
                    // we have the freq value and not the id of that freq
                    int idfreq = (int) freq/2.0;
                    out(c_feature) = in(idchan, idfreq);
                    c_feature ++;
                }
            }

            return out.transpose();
        }

        bool Qda::checkDimension(void){
            // check the means
            if(this->means_.rows() != this->config_.n_features ||
               this->means_.cols() != this->config_.n_classes){
                ROS_ERROR("[%s] Wrong dimensions in the 'means' parameter", this->getName().c_str());
                return false;
            }

            // check the cov
            if(this->covs_.rows() != this->config_.n_features * this->config_.n_features ||
               this->covs_.cols() != this->config_.n_classes){
                ROS_ERROR("[%s] Wrong dimensions in the 'covs' parameter", this->getName().c_str());
                return false;
            }

            // check classes size
            if(this->config_.priors.size() != this->config_.n_classes |\
               this->config_.n_classes != this->config_.class_lbs.size()){
                ROS_ERROR("[%s] Wrong dimensions in the given classes parameters", this->getName().c_str());
                return false;
            }

            // check the features
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