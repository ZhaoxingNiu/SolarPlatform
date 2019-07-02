#include "./FocusHeliosSplits.h"
#include <cstdlib>
#include <random>

void FocusHeliosSplit::init(SolarScene* solar_scene) {
	solar_scene_ = solar_scene;
	helio_num_ = (solar_scene_->heliostats).size();
	auto helios0 = solar_scene_->heliostats[0];
	sub_num_ = helios0->row_col_.x * helios0->row_col_.y;
	
	row_col_ = helios0->row_col_;
	gap_length_ = helios0->gap_;
	helio_size_ = helios0->size_;
	helio_sub_size_.x = (helio_size_.x - gap_length_.x*(row_col_.x - 1)) / row_col_.x;
	helio_sub_size_.y = 0;
	helio_sub_size_.z = (helio_size_.z - gap_length_.y*(row_col_.y - 1)) / row_col_.y;
	
}

void FocusHeliosSplit::localCoor() {
	// 生成初始位置以及size
	for (int helio_index = 0; helio_index < helio_num_; ++helio_index) {
		//生成 局部坐标,保存
		auto helios0 = solar_scene_->heliostats[helio_index];
		float3 start_pos = helios0->size_ / -2; 
		start_pos.x += helio_sub_size_.x/2;
		start_pos.y = 0;
		start_pos.z += helio_sub_size_.z/2;

		float step_x = helio_sub_size_.x + gap_length_.x;
		float step_z = helio_sub_size_.z + gap_length_.y;
		for (int i = 0; i < row_col_.x; ++i) {
			for (int j = 0; j < row_col_.y; ++j) {
				float3 pos;
				pos.x = start_pos.x + i * step_x;
				pos.y = start_pos.y;
				pos.z = start_pos.z + j * step_z;
				sub_pos_.push_back(pos);
				sub_size_.push_back(helio_sub_size_);
			}
		}
	}
}

void FocusHeliosSplit::setSurface() {
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene_->receivers[0]);
	for (int helio_index = 0; helio_index < helio_num_; helio_index++) {
		auto helios0 = solar_scene_->heliostats[helio_index];
		float true_dis = length(helios0->pos_
			- rectrece->focus_center_);

		float focus = true_dis * FOCUS_LENGTH_RATE;
		focus_length_.push_back(focus);
	}
}

void FocusHeliosSplit::moveToSurface() {
	// 移动到平面上，并且设置局部坐标中的法向
	for (int helio_index = 0; helio_index < helio_num_; helio_index++) {
		float focus = focus_length_[helio_index];
		// 子平面中心移动到曲面上  $$ x^2 + z^2 = 2py  p = 2*focus $$
		// 点移动到平面上
		for (int sub_helio_index = helio_index*sub_num_;
			sub_helio_index < (helio_index + 1)*sub_num_; ++sub_helio_index) {
			float3 pos = sub_pos_[sub_helio_index];
			pos.y = (pos.x*pos.x + pos.y*pos.y) / 4 / focus;
			sub_pos_[sub_helio_index] = pos;

			// 根据坐标位置生成法向
			float3 sub_helio_normal = make_float3(-2*pos.x, 4*focus, -2*pos.z);
			sub_helio_normal = normalize(sub_helio_normal);
			sub_normal_.push_back(sub_helio_normal);
		}
	}
}

void FocusHeliosSplit::rotate() {
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene_->receivers[0]);
	for (int helio_index = 0; helio_index < helio_num_; helio_index++) {
		auto helios = solar_scene_->heliostats[helio_index];
		float3 pos = helios->pos_;
		float3 focus_center = make_float3(0.0f, 102.5f,0.0f);
		float rand_rate = ((float)rand() / (RAND_MAX));
		rand_rate = (rand_rate - 0.5f) * 2;
		float3 x_axis = make_float3(1.0f, 0.0f, 0.0f);
		focus_center += x_axis*rand_rate * 5;
		
		float3 reflect_dir = focus_center - pos;
		reflect_dir = normalize(reflect_dir);
		float3 dir = reflect_dir - solarenergy::sun_dir;
		float3 helio_normal = normalize(dir);
		for (int sub_helio_index = helio_index*sub_num_;
			sub_helio_index < (helio_index + 1)*sub_num_; ++sub_helio_index) {
			float3 pos = sub_pos_[sub_helio_index];
			float3 normal = sub_normal_[sub_helio_index];
			float3 new_pos = global_func::local2world(pos, helio_normal);
			float3 new_normal = global_func::local2world(normal, helio_normal);
			sub_pos_[sub_helio_index] = new_pos;
			sub_normal_[sub_helio_index] = new_normal;
		}
	}
}

void FocusHeliosSplit::transform() {
	for (int helio_index = 0; helio_index < helio_num_; helio_index++) {
		auto helios = solar_scene_->heliostats[helio_index];
		// 从局部坐标移动到全局坐标系中
		for (int sub_helio_index = helio_index*sub_num_;
			sub_helio_index < (helio_index + 1)*sub_num_; ++sub_helio_index) {
			float3 pos = sub_pos_[sub_helio_index];
			sub_pos_[sub_helio_index] = pos + helios->pos_;
		}
	}
}

void FocusHeliosSplit::split() {
	localCoor();
	setSurface();
	moveToSurface();
	rotate();
	transform();
}

void FocusHeliosSplit::saveFile(std::string file_out_pos, std::string file_out_norm) {
	fstream outFile(file_out_pos, ios::out | ios::app);
	if (outFile.fail()) {
		cerr << "Can't write to this file!" << endl;
	}

	fstream outFileNorm(file_out_norm, ios::out | ios::app);
	if (outFileNorm.fail()) {
		cerr << "Can't write to this file!" << endl;
	}

	int total_sub_helios_num = helio_num_ * sub_num_;
	outFile << "\n# Heliostats" << endl;
	outFile << "gap  0.02 0.02"<< endl;
	outFile << "matrix 1 1"<< endl;
	for (int i = 0; i < total_sub_helios_num; ++i) {
		outFile << "helio " << sub_pos_[i].x << ' ' << sub_pos_[i].y << ' ' << sub_pos_[i].z << endl;
		outFile << sub_size_[i].x << ' ' << sub_size_[i].y << ' ' << sub_size_[i].z << endl;
		outFileNorm << sub_normal_[i].x << ' ' << sub_normal_[i].y << ' ' << sub_normal_[i].z << endl;
	}
	outFile.close();
	outFileNorm.close();
}


bool test_focus_helios_split() {
	//just init 
	solarenergy::scene_filepath = "../SceneData/paper/ps10/ps10_flat_rece.scn";
	std::string res_path = "../SceneData/paper/ps10/ps10_real_rece_s2.scn";
	std::string res_norm_path = "../SceneData/paper/ps10/ps10_real_rece_s2_norm.scn";
	solarenergy::sun_dir = make_float3(0.0f, -0.79785f, -1.0f);
	solarenergy::sun_dir = normalize(solarenergy::sun_dir);

	SolarScene *solar_scene;
	solar_scene = SolarScene::GetInstance();
	solar_scene->InitContent();

	FocusHeliosSplit focus_split;
	focus_split.init(solar_scene);
	focus_split.split();
	focus_split.saveFile(res_path,res_norm_path);

	return true;
}