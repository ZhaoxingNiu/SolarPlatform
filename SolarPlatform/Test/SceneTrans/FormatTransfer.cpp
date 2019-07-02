#include "./FormatTransfer.h"


void FormatTransfer::LoadFile(SolarScene * solar_scene,std::string src_path)
{
	Receiver* receiver;
	RectGrid* grid0;
	Heliostat* heliostat;

	string line, word;
	stringstream scene_stream;
	fstream inFile(src_path);
	if (inFile.fail()) {
		cerr << "Can't open the file!" << endl;
	}

	int grid_type = 0,	helio_type = 0;		//defalut type for grid and helio
	int grid_num = 1, helio_num = 0;
	int start_helio_pos = 0;
	int2 matrix;
	float2 gap;
	float3 interval, pos, size;
	float3 grid_min, grid_max, grid_pos, grid_size;

	while (getline(inFile, line)) {
		scene_stream.clear();
		scene_stream.str(line);
		scene_stream >> word;
		
		if (word == "//") {
			scene_stream >> word;
			if (word == "Scene") {
				getline(inFile, line);
				scene_stream.clear();
				scene_stream.str(line);
				scene_stream >> solar_scene->ground_length_ >> solar_scene->ground_width_;

			}
			else if (word == "Receiver") {
				int recv_type, face_num;
				float3 norm;
				getline(inFile, line);
				scene_stream.clear();
				scene_stream.str(line);
				scene_stream >> pos.x >> pos.y >> pos.z >> size.x >> size.y >> size.z >> recv_type;
				switch (recv_type)
				{
				case 0: receiver = new RectangleReceiver();
					break;
				case 1: receiver = new CylinderReceiver();
					break;
				case 2: receiver = new CircularTruncatedConeReceiver();
					break;
				default: //cerr << "Receiver type not define!" << endl;
					receiver = new RectangleReceiver();
					break;
				}
				
				face_num = 0;							// HCT's file didn't define the face number and normal.
				norm.x = 0; norm.y = 0; norm.z = -1;		// Self-define these parameters 
			
				receiver->type_ = recv_type;
				receiver->pos_ = pos;
				receiver->size_ = size;
				receiver->face_num_ = face_num;
				receiver->normal_ = norm;
				solar_scene->receivers.push_back(receiver);
				receiver = nullptr;
			}
			else if (word == "Filed") {
				getline(inFile, line);
				scene_stream.clear();
				scene_stream.str(line);
				scene_stream >> grid_type;
			}
			else if (word == "heliostat") {
				getline(inFile, line);
				scene_stream.clear();
				scene_stream.str(line);
				scene_stream >> interval.x >> interval.z;
				interval.y = fmax(interval.x, interval.z);

			}
			else if (word == "Heliostat") {
				scene_stream >> word >> word;
				if (word == "matrix") {
					getline(inFile, line);
					scene_stream.clear();
					scene_stream.str(line);
					scene_stream >> matrix.x >> matrix.y;
				}
				else if (word == "gap") {
					getline(inFile, line);
					scene_stream.clear();
					scene_stream.str(line);
					scene_stream >> gap.x >> gap.y;
				}
			}
			else if (word == "Reflector") {
				scene_stream >> word;
				if (word == "numbers") {
					getline(inFile, line);
					scene_stream.clear();
					scene_stream.str(line);
					scene_stream >> helio_num;
				}
				else if (word == "position") {
					grid_min.x = INT_MAX; grid_min.y = INT_MAX; grid_min.z = INT_MAX;
					grid_max.x = INT_MIN; grid_max.y = INT_MIN; grid_max.z = INT_MIN;
					while (getline(inFile, line)) {
						scene_stream.clear();
						scene_stream.str(line);
						scene_stream >> pos.x >> pos.y >> pos.z >> size.x >> size.z >> size.y;
						pos.z *= -1;
						//size.y = fmax(size.x, size.z);

						grid_min.x = fmin(grid_min.x, pos.x);
						grid_min.y = fmin(grid_min.y, pos.y);
						grid_min.z = fmin(grid_min.z, pos.z);
						grid_max.x = fmax(grid_max.x, pos.x);
						grid_max.y = fmax(grid_max.y, pos.y);
						grid_max.z = fmax(grid_max.z, pos.z);

						switch (helio_type)
						{
						case 0:
							heliostat = new RectangleHelio();
							heliostat->gap_ = gap;
							heliostat->row_col_ = matrix;
							heliostat->pos_ = pos;
							heliostat->size_ = size;
							solar_scene->heliostats.push_back(heliostat);
							heliostat = nullptr;
							break;
						default:
							break;
						}
					}
					grid_pos.x = grid_min.x - 0.5*interval.x;
					grid_pos.y = fmax(0, grid_min.y - 0.5 * fmax(interval.x,interval.z));
					//grid_pos.z = grid_max.z + size.y / 2.0f;
					grid_pos.z = grid_min.z - 0.5*interval.z;
					grid_size.x = fabs(grid_max.x - grid_min.x + interval.x);
					grid_size.y = fabs(fmax(interval.x, interval.z));
					grid_size.z = fabs(grid_max.z - grid_min.z + interval.z);
					
				}
			}
		}
	}

	solar_scene->grid_num_ = grid_num;				// Only consider 1 grid and rectangle grid for now
	switch (grid_type)
	{
	case 0:
		grid0 = new RectGrid();
		grid0->helio_type_ = helio_type;
		grid0->type_ = grid_type;
		grid0->interval_ = interval;
		grid0->num_helios_ = helio_num;
		grid0->size_ = grid_size;
		grid0->pos_ = grid_pos;
		grid0->start_helio_pos_ = start_helio_pos;
		solar_scene->grid0s.push_back(grid0);
		grid0 = nullptr;
	default:
		break;
	}

	inFile.close();
}

void FormatTransfer::saveFile(SolarScene * solar_scene, std::string res_path)
{
	fstream outFile(res_path, ios::out|ios::app);
	if (outFile.fail()) {
		cerr << "Can't write to this file!" << endl;
	}

	outFile << "# Ground Boundary" << endl;
	outFile << "ground " << solar_scene->ground_length_ << ' ' << solar_scene->ground_width_ << endl;
	outFile << "ngrid " << solar_scene->grid_num_ << endl;

	outFile << "\n# Receiver attributes" << endl;
	for (auto&recv : solar_scene->receivers) {
		outFile << "Recv " << recv->type_ << endl;
		outFile << "pos " << recv->pos_.x << ' ' << recv->pos_.y << ' ' << recv->pos_.z << endl;
		outFile << "size " << recv->size_.x << ' ' << recv->size_.y << ' ' << recv->size_.z << endl;
		outFile << "norm " << recv->normal_.x << ' ' << recv->normal_.y << ' ' << recv->normal_.z << endl;
		outFile << "face " << recv->face_num_ << endl;
	}
	outFile << "end" << endl;

	for (int i = 1; i <= solar_scene->grid_num_; i++) {
		outFile << "\n# Grid" << i << " attributes" << endl;

		switch (solar_scene->grid0s[i-1]->type_)
		{
		case 0: {
			RectGrid* rect_grid = (RectGrid*)solar_scene->grid0s[i-1];
			outFile << "Grid " << rect_grid->type_ << endl;
			outFile << "pos " << rect_grid->pos_.x << ' ' << rect_grid->pos_.y << ' ' << rect_grid->pos_.z << endl;
			outFile << "size " << rect_grid->size_.x << ' ' << rect_grid->size_.y << ' ' << rect_grid->size_.z << endl;
			outFile << "inter " << rect_grid->interval_.x << ' ' << rect_grid->interval_.y << ' ' << rect_grid->interval_.z << endl;
			outFile << "n " << rect_grid->num_helios_ << endl;
			outFile << "type " << rect_grid->helio_type_ << endl;
			outFile << "end" << endl;
			break;
		}
		default:
			break;
		}

		outFile << "\n# Heliostats" << endl;
		float2 gap;
		int2 matrix;
		int cnt = 0;
		for (auto&helio : solar_scene->heliostats) {
			if (cnt == 0) {
				outFile << "gap " << helio->gap_.x << ' ' << helio->gap_.y << endl;
				outFile << "matrix " << helio->row_col_.x << ' ' << helio->row_col_.y << endl;
				gap = helio->gap_;
				matrix = helio->row_col_;
			}
			else {
				if (gap.x != helio->gap_.x || gap.y!=helio->gap_.y) {
					outFile << "gap " << helio->gap_.x << ' ' << helio->gap_.y << endl;
					gap = helio->gap_;
				}
				if (matrix.x != helio->row_col_.x || matrix.y != helio->row_col_.y) {
					outFile << "matrix " << helio->row_col_.x << ' ' << helio->row_col_.y << endl;
					matrix = helio->row_col_;
				}
			}
			outFile << "helio " << helio->pos_.x << ' ' << helio->pos_.y << ' ' << helio->pos_.z << endl;
			outFile	<< helio->size_.x << ' ' << helio->size_.y << ' ' << helio->size_.z << endl;
			cnt++;

		}
	}
	outFile.close();
}


bool testSceneFormatTransfer() {
	std::string file_in = "../SceneData/paper/ps10/PS10_field_data.scn";
	std::string file_out = "../SceneData/paper/ps10/test.scn";

	//just init 
	solarenergy::scene_filepath = "../SceneData/paper/helioField_scene_shadow.scn";

	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	// ´æÔÚÄÚ´æÐ¹Â©
	solar_scene->grid0s.clear();
	solar_scene->heliostats.clear();
	solar_scene->receivers.clear();

	FormatTransfer trans;
	trans.LoadFile(solar_scene, file_in);
	trans.saveFile(solar_scene, file_out);


	return true;
}

bool testSceneFormatTransferPs10() {


	//just init 
	solarenergy::scene_filepath = "../SceneData/paper/ps10/ps10.scn";

	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	/*   //
	auto receiver0 = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[0]);
	auto receiver1 = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[1]);
	auto receiver2 = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[2]);
	auto receiver3 = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[3]);
	*/

	solar_scene->InitContent();
	return true;
}