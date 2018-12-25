#include "./test_get_file_peak.h"
#include <stdlib.h>


float test_stringToFloat (const std::string& str)
{
	std::istringstream iss(str);
	float num;
	iss >> num;
	return num;
}

float test_get_file_peak(std::string path) {

	float peak = -1;
	std::string str_line;
	std::ifstream flux_file;

	try {
		flux_file.open(path);
		std::stringstream scene_stream;
		// read file's buffer contents into streams
		scene_stream << flux_file.rdbuf();
		flux_file.close();
		while (getline(scene_stream, str_line)) {
			std::stringstream input(str_line);
			std::string tmp;
			while (getline(input, tmp, ',')) {
				float tmp_peak = test_stringToFloat(tmp);
				if (tmp_peak > peak) {
					peak = tmp_peak;
				}
			}
		}
	}catch (std::ifstream::failure e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		return false;
	}
	return peak;
}