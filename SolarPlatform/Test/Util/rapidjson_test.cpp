#include "./rapidjson_test.h"

// include rapidjson header
#include "../../Util/rapidjson/rapidjson.h"
#include "../../Util/rapidjson/document.h"
#include "../../Util/rapidjson/filereadstream.h"
#include "../../Util/rapidjson/filewritestream.h"
#include "../../Util/rapidjson/writer.h"

// file processor
#include <fstream>
#include <iostream>

int testRapidjsonReadConf(std::string filename) {
	std::ifstream configure_file(filename);
	std::string str((std::istreambuf_iterator<char>(configure_file)),
		std::istreambuf_iterator<char>());

	rapidjson::Document document;
	document.Parse(str.c_str());

	// 读取配置文件的数据
	if (document.HasMember("sun_dir")) {
		const rapidjson::Value& value = document["sun_dir"];
		assert(value.Size() == 3);
		std::cout << "sun_dir x: " << value[0].GetDouble()
			<< "\n      y: "<< value[1].GetDouble()
			<< "\n      z: " << value[2].GetDouble()
			<< std::endl;
	}

	if (document.HasMember("num_of_sunshape_groups")) {
		const rapidjson::Value& value = document["num_of_sunshape_groups"];
		std::cout << "num_of_sunshape_groups: " << value.GetInt() << std::endl;
	}

	if (document.HasMember("num_per_sunshape_group")) {
		const rapidjson::Value& value = document["num_per_sunshape_group"];
		std::cout << "num_per_sunshape_group: " << value.GetInt() << std::endl;
	}

	if (document.HasMember("csr")) {
		const rapidjson::Value& value = document["csr"];
		std::cout << "csr: " << value.GetDouble() << std::endl;
	}

	if (document.HasMember("dni")) {
		const rapidjson::Value& value = document["dni"];
		std::cout << "dni: " << value.GetDouble() << std::endl;
	}

	if (document.HasMember("receiver_pixel_length")) {
		const rapidjson::Value& value = document["receiver_pixel_length"];
		std::cout << "receiver_pixel_length: " << value.GetDouble() << std::endl;
	}

	if (document.HasMember("helio_disturb_std")) {
		const rapidjson::Value& value = document["helio_disturb_std"];
		std::cout << "helio_disturb_std: " << value.GetDouble() << std::endl;
	}

	if (document.HasMember("helio_reflected_rate")) {
		const rapidjson::Value& value = document["helio_reflected_rate"];
		std::cout << "helio_reflected_rate: " << value.GetDouble() << std::endl;
	}

	if (document.HasMember("helio_pixel_length")) {
		const rapidjson::Value& value = document["helio_pixel_length"];
		std::cout << "helio_pixel_length: " << value.GetDouble() << std::endl;
	}

	if (document.HasMember("scene_file_path")) {
		const rapidjson::Value& value = document["scene_file_path"];
		std::cout << "scene_file_path: " << value.GetString() << std::endl;
	}

	return 0;
}