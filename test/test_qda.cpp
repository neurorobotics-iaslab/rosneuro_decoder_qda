#include <rosneuro_decoder/Decoder.h>
#include <rosneuro_decoder_qda/Qda.hpp>

template<typename T>
void writeCSV(const std::string& filename, const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&  matrix) {
	const static Eigen::IOFormat format(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream file(filename);
	if (file.is_open()) {
		file << matrix.format(format);
		file.close();
	}
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> readCSV(const std::string& filename) {

	std::vector<T> values;

	std::ifstream file(filename);
	std::string row;
	std::string entry;
	int nrows = 0;

	while (getline(file, row)) {
		std::stringstream rowstream(row);

		while (getline(rowstream, entry, ',')) {
			values.push_back(std::stod(entry));
		}
		nrows++; 
	}

	return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), nrows, values.size() / nrows);

}

int  main(int argc, char** argv){
    ros::init(argc, argv, "test_qda");

    std::string datapath;
    if(ros::param::get("~datapath", datapath) == false){
        ROS_ERROR("Miss mandatory paramter 'datapath'");
        return false;
    }

    rosneuro::decoder::Decoder* decoder = new rosneuro::decoder::Decoder();

    const std::string fileinput  = datapath + "/test/features.csv";
    const std::string fileoutput = datapath + "/test/output.csv";

    Eigen::MatrixXd input = readCSV<double>(fileinput);

    if(!decoder->configure()){
        ROS_ERROR("[%s] decoder not confgured correctly", decoder->name().c_str());
        return false;
    }

    std::ofstream outputFile(fileoutput);
    if(outputFile.is_open()){
        std::cout << "file opened" << std::endl;
        for(int i = 0; i < input.rows(); i++){
            Eigen::VectorXf temp = input.row(i).cast<float>();
            Eigen::VectorXf temp2 = decoder->apply(temp);
            outputFile << temp2.transpose() << std::endl;
        }
    }

    outputFile.close();
    std::cout << "file closed" << std::endl;

}