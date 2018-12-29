#include "./scene_file_proc.h"

inline void readHelioParamter(stringstream &scene_stream, Heliostat *heliostat) {
	std::string line_buf;
	getline(scene_stream, line_buf);
	float3 size;
	std::stringstream line_stream_buf;
	line_stream_buf << line_buf;
	line_stream_buf >> size.x >> size.y >> size.z;
	heliostat->size_ = size;
};


SceneFileProc::SceneFileProc() {
	string_value_read_map["pos"] = StringValue::pos;
	string_value_read_map["size"] = StringValue::size;
	string_value_read_map["norm"] = StringValue::norm;
	string_value_read_map["face"] = StringValue::face;
	string_value_read_map["end"] = StringValue::end;
	string_value_read_map["gap"] = StringValue::gap;
	string_value_read_map["matrix"] = StringValue::matrix;
	string_value_read_map["helio"] = StringValue::helio;
	string_value_read_map["inter"] = StringValue::inter;
	string_value_read_map["n"] = StringValue::n;
	string_value_read_map["type"] = StringValue::type;
}

StringValue SceneFileProc::Str2Value(string str) {
	if (string_value_read_map.count(str)) {
		return string_value_read_map[str];
	}
	else {
		std::cerr << str << " is not define in the string_value_read_map" << std::endl;
		return StringValue::illegal;
	}
}

void SceneFileProc::SceneNormalRead(std::string filepath, std::vector<float3> &norm_vec) {
	std::ifstream scene_normal_file;
	scene_normal_file.open(filepath);
	stringstream scene_normal_stream;
	// read file's buffer contents into streams
	scene_normal_stream << scene_normal_file.rdbuf();
	scene_normal_file.close();

	std::string str_line;
	norm_vec.clear();
	while (getline(scene_normal_stream, str_line)) {
		std::stringstream line_stream;
		line_stream << str_line;
		float3 norm;
		line_stream >> norm.x >> norm.y >> norm.z;
		norm_vec.push_back(norm);
	}
}


bool SceneFileProc::SceneFileRead(SolarScene *solarscene, std::string filepath) {
	Receiver *receiver;
	RectGrid *grid0;
	Heliostat *heliostat;
	//to save the global settings
	int helio_type_buf;
	bool gap_set = false;
	bool matrix_set = false;
	float2 gap_buf;
	int2 matrix_buf;
	// prama to ensure the memery
	int helio_input_num = 0;
	int helio_input_total = 0;
	int grid_start_helio_pos = 0;

	InputMode inputMode = InputMode::none;
	solarScene_ = solarscene;
	std::string str_line;
	std::ifstream scene_file;
	// ensure ifstream objects can throw exceptions:
	scene_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try {
		//openfile
		scene_file.open(filepath);
		stringstream scene_stream;
		// read file's buffer contents into streams
		scene_stream << scene_file.rdbuf();
		//close the file handlers
		scene_file.close();
		while (getline(scene_stream, str_line)) {
			if (str_line[0] == '#' || str_line == "") { continue; }
#ifdef _DEBUG
	//		std::cout << str_line << std::endl;
#endif
			std::stringstream line_stream;
			line_stream << str_line;  
			//get the line flag
			std::string line_head;
			line_stream >> line_head;

			//chose the mode to read
			if (line_head == "ground") {  
				inputMode = InputMode::ground;
				float ground_length, ground_width;
				line_stream >> ground_length >> ground_width;
				solarScene_->ground_length_ = ground_length;
				solarScene_->ground_width_ = ground_width;
				continue;
			}
			else if (line_head == "Recv") {  // init the receiver
				inputMode = InputMode::receiver;
				int receive_type;
				line_stream >> receive_type;
				switch (receive_type) 
				{
					case 0: receiver = new RectangleReceiver();
						break;
					case 1: receiver = new CylinderReceiver();
						break;
					case 2: receiver = new CircularTruncatedConeReceiver();
						break;
					default:
						std::cerr << "Receiver type not define!" << std::endl;
						break;
				}
				receiver->type_ = receive_type;
				continue;

			}
			else if (line_head == "Grid"){
				//vertify the last grid
				if (helio_input_num != helio_input_total) {
					std::cerr << "helistat number is wrong" << std::endl;
				}
				helio_input_num = 0;
				helio_input_total = 0;

				inputMode = InputMode::grid;
				int grid_type;
				line_stream >> grid_type;
				switch (grid_type)
				{
				case 0: grid0 = new RectGrid();
					break;
				case 1:
					break;
				default:
					std::cerr << "Grid type not define!" << std::endl;
					break;
				}
				grid0->type_ = grid_type;
				helio_input_num = 0;
				grid0->start_helio_pos_ = grid_start_helio_pos;
				//reset the gap 
				gap_set = false;
				matrix_set = false;
				continue;
			}
			/*************  switch section  ************************/
			switch (inputMode)
			{
				case InputMode::none:
#ifdef _DEBUG
					std::cout << "none mode: ignore this line" << std::endl;
#endif
					break;
				case InputMode::ground:
					int grid_num;
					if (line_head == "ngrid") {
						line_stream >> grid_num;
						solarScene_->grid_num_ = grid_num;
					}
					break;
				case InputMode::receiver:
					switch (string_value_read_map[line_head]) {
						case StringValue::pos:
							float3 pos;
							line_stream >> pos.x >> pos.y >> pos.z;
							receiver->pos_ = pos;
							break;
						case StringValue::size:
							float3 size;
							line_stream >> size.x >> size.y >> size.z;
							receiver->size_ = size;
							break;
						case StringValue::norm:
							float3 norm;
							line_stream >> norm.x >> norm.y >> norm.z;
							receiver->normal_ = norm;
							break;
						case StringValue::face:
							int face_num;
							line_stream >> face_num;
							receiver->face_num_ = face_num;
							break;
						case StringValue::end: //push the receiver
							solarScene_->receivers.push_back(receiver);  
							receiver = nullptr;
							break;
						default:
							break;
					}

					break;
				case InputMode::grid:
					if (grid0->type_ == 0) {
						switch (string_value_read_map[line_head]) {
						case StringValue::pos:
							float3 pos;
							line_stream >> pos.x >> pos.y >> pos.z;
							grid0->pos_ = pos;
							break;
						case StringValue::size:
							float3 size;
							line_stream >> size.x >> size.y >> size.z;
							grid0->size_ = size;
							break;
						case StringValue::inter:
							float3 inter;
							line_stream >> inter.x >> inter.y >> inter.z;
							grid0->interval_ = inter;
							break;
						case StringValue::n:   //update the helio size
							int n;
							line_stream >> n;
							grid0->num_helios_ = n;
							helio_input_total = n;
							grid_start_helio_pos += n;
							break;
						case StringValue::type:
							int helio_type;
							line_stream >>helio_type;
							grid0->helio_type_ = helio_type;
							break;
						case StringValue::end:
							solarScene_->grid0s.push_back(grid0);
							// change the mode
							inputMode = InputMode::heliostat;
							helio_type_buf = grid0->helio_type_;
							//delete grid0;
							grid0 = nullptr;
							break;
						default:
							break;
						}
					}
					else {
						std::cerr << "this type grid is not support";
					}
					break;
				case InputMode::heliostat:
					switch (string_value_read_map[line_head]) {
						case StringValue::gap:
							float2 gap;
							line_stream >> gap.x >> gap.y;
							gap_buf = gap;
							gap_set = true;
							break;
						case StringValue::matrix:
							int2 matrix;
							line_stream >> matrix.x >> matrix.y;
							matrix_buf = matrix;
							matrix_set = true;
							break;
						case StringValue::helio:
							if (!gap_set || !matrix_set) {
								std::cerr << "did not set the gap and matrix"<< std::endl;
								break;
							}
							//ensure the member
							if (helio_input_num >= helio_input_total) {
								std::cerr << "too many helistat" << std::endl;
							}
							if (helio_type_buf == 0) {
								heliostat = new RectangleHelio;
								float3 pos;
								line_stream >> pos.x >> pos.y >> pos.z;
								heliostat->pos_ = pos;
								heliostat->gap_ = gap_buf;//gap
								heliostat->row_col_ = matrix_buf;//matrix
								if (helio_type_buf == 0) {
									readHelioParamter(scene_stream, heliostat);
								}
								solarScene_->heliostats.push_back(heliostat);
								heliostat = nullptr; //make sure heliostat is null
								++helio_input_num;
							}
							break;
						default:
							break;
					}
					break;
				default:
					std::cout << "illeagle parameter" << std::endl;
					break;
			}
		}

	}
	catch (std::ifstream::failure e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		return false;
	}
	return true;
}