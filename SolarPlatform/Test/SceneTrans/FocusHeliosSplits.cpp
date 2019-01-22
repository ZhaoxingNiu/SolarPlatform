#include "./FocusHeliosSplits.h"
#include <cstdlib>
#include <random>

void FocusHeliosSplit::init(SolarScene* solar_scene) {
	solar_scene_ = solar_scene;
	helio_num = (solar_scene_->heliostats).size();
	auto helios0 = solar_scene_->heliostats[0];
	sub_num = helios0->row_col_.x * helios0->row_col_.y;
	
	row_col = helios0->row_col_;
	gap_length = helios0->gap_;
	helio_size = helios0->size_;
	helio_sub_size.x = (helio_size.x - gap_length.x*(row_col.x - 1)) / row_col.x;
	helio_sub_size.y = 0;
	helio_sub_size.z = (helio_size.z - gap_length.y*(row_col.y - 1)) / row_col.y;
	
}

void FocusHeliosSplit::local_coor() {
	// 生成初始位置以及size
	for (int helio_index = 0; helio_index < helio_num; ++helio_index) {
		//生成 局部坐标,保存
		auto helios0 = solar_scene_->heliostats[helio_index];
		float3 start_pos = helios0->size_ / -2; 
		start_pos.x += helio_sub_size.x/2;
		start_pos.y = 0;
		start_pos.z += helio_sub_size.z/2;

		float step_x = helio_sub_size.x + gap_length.x;
		float step_z = helio_sub_size.z + gap_length.y;
		for (int i = 0; i < row_col.x; ++i) {
			for (int j = 0; j < row_col.y; ++j) {
				float3 pos;
				pos.x = start_pos.x + i * step_x;
				pos.y = start_pos.y;
				pos.z = start_pos.z + j * step_z;
				sub_pos.push_back(pos);
				sub_size.push_back(helio_sub_size);
			}
		}
	}
}

void FocusHeliosSplit::set_surface() {
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene_->receivers[0]);
	for (int helio_index = 0; helio_index < helio_num; helio_index++) {
		auto helios0 = solar_scene_->heliostats[helio_index];
		float true_dis = length(helios0->pos_
			- rectrece->focus_center_);

		float focus = true_dis * FOCUS_LENGTH_RATE;
		focus_length.push_back(focus);
	}
}

void FocusHeliosSplit::move_to_surface() {
	// 移动到平面上，并且设置局部坐标中的法向
	for (int helio_index = 0; helio_index < helio_num; helio_index++) {
		float focus = focus_length[helio_index];
		// 子平面中心移动到曲面上  $$ x^2 + z^2 = 2py  p = 2*focus $$
		// 点移动到平面上
		for (int sub_helio_index = helio_index*sub_num;
			sub_helio_index < (helio_index + 1)*sub_num; ++sub_helio_index) {
			float3 pos = sub_pos[sub_helio_index];
			pos.y = (pos.x*pos.x + pos.y*pos.y) / 4 / focus;
			sub_pos[sub_helio_index] = pos;

			// 根据坐标位置生成法向
			float3 sub_helio_normal = make_float3(-2*pos.x, 4*focus, -2*pos.z);
			sub_helio_normal = normalize(sub_helio_normal);
			sub_normal.push_back(sub_helio_normal);
		}
	}
}

void FocusHeliosSplit::rotate() {
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene_->receivers[0]);
	for (int helio_index = 0; helio_index < helio_num; helio_index++) {
		auto helios = solar_scene_->heliostats[helio_index];
		float3 pos = helios->pos_;
		//不清楚聚焦方式 随便尝试吧

		float3 focus_center = make_float3(0.0f, 101.0f,0.0f);
		float rand_rate = ((float)rand() / (RAND_MAX));
		if (rand_rate < 0.10) {
			focus_center = make_float3(2.5f, 101.0f, 0.0f);
		}
		else if(rand_rate < 0.20) {
			focus_center = make_float3(-2.5f, 101.0f, 0.0f);
		}
		else if (rand_rate < 0.261) {
			focus_center = make_float3(7.5f, 101.0f, 0.0f);
		}
		else if (rand_rate < 0.322) {
			focus_center = make_float3(-7.5f, 101.0f, 0.0f);
		}
		else if (rand_rate < 0.92) {
			rand_rate = ((float)rand() / (RAND_MAX));
			if (rand_rate < 0.5f) {
				focus_center = make_float3(0.0f, 104.0f, 0.0f);
			}
			else {
				focus_center = make_float3(0.0f, 98.0f, 0.0f);
			}

			rand_rate = ((float)rand() / (RAND_MAX));
			rand_rate = (rand_rate - 0.5f) * 2;
			float3 x_axis = make_float3(1.0f, 0.0f, 0.0f);
			focus_center += x_axis*rand_rate * 10;

			float3 y_axis = make_float3(0.0f, 1.0f, 0.0f);

			float weitiao = (1 - abs(rand_rate))*0.7;
			if (focus_center.y > 101.0f) {
				focus_center += y_axis*weitiao;
			}
			else {
				focus_center -= y_axis*weitiao;
			}
			
			rand_rate = ((float)rand() / (RAND_MAX));
			rand_rate = (rand_rate - 0.5f) * 2;
			focus_center += y_axis*rand_rate * 3;

		}
		else  if (rand_rate < 0.93) {
			rand_rate = ((float)rand() / (RAND_MAX));
			if (rand_rate < 0.5f) {
				focus_center = make_float3(0.0f, 103.0f, 0.0f);
			}
			else {
				focus_center = make_float3(0.0f, 99.0f, 0.0f);
			}
		}
		else {
			rand_rate = ((float)rand() / (RAND_MAX));
			rand_rate = (rand_rate - 0.5f) * 2;
			float3 x_axis = make_float3(1.0f, 0.0f, 0.0f);
			
			focus_center += x_axis*rand_rate * 11;

			float3 y_axis = make_float3(0.0f, 1.0f, 0.0f);
			rand_rate = ((float)rand() / (RAND_MAX));

			if (rand_rate < 0.5f) {
				rand_rate = ((float)rand() / (RAND_MAX));
				rand_rate = (rand_rate - 0.5f) * 2;
				focus_center += y_axis*rand_rate * 3;

			}else {
				rand_rate = ((float)rand() / (RAND_MAX));
				rand_rate = (rand_rate - 0.5f) * 2;
				focus_center += y_axis*rand_rate * 5;

			}
		}

		float3 reflect_dir = focus_center - pos;
		reflect_dir = normalize(reflect_dir);
		float3 dir = reflect_dir - solarenergy::sun_dir;
		float3 helio_normal = normalize(dir);
		for (int sub_helio_index = helio_index*sub_num;
			sub_helio_index < (helio_index + 1)*sub_num; ++sub_helio_index) {
			float3 pos = sub_pos[sub_helio_index];
			float3 normal = sub_normal[sub_helio_index];
			float3 new_pos = global_func::local2world(pos, helio_normal);
			float3 new_normal = global_func::local2world(normal, helio_normal);
			sub_pos[sub_helio_index] = new_pos;
			sub_normal[sub_helio_index] = new_normal;
		}
	}
}

void FocusHeliosSplit::transform() {
	for (int helio_index = 0; helio_index < helio_num; helio_index++) {
		auto helios = solar_scene_->heliostats[helio_index];
		// 从局部坐标移动到全局坐标系中
		for (int sub_helio_index = helio_index*sub_num;
			sub_helio_index < (helio_index + 1)*sub_num; ++sub_helio_index) {
			float3 pos = sub_pos[sub_helio_index];
			sub_pos[sub_helio_index] = pos + helios->pos_;
		}
	}
}

void FocusHeliosSplit::split() {
	local_coor();
	set_surface();
	move_to_surface();
	rotate();
	transform();
}

void FocusHeliosSplit::saveFile(std::string file_out_pos, std::string file_out_norm) {
	fstream outFile(file_out_pos, ios::out);
	if (outFile.fail()) {
		cerr << "Can't write to this file!" << endl;
	}

	fstream outFileNorm(file_out_norm, ios::out);
	if (outFileNorm.fail()) {
		cerr << "Can't write to this file!" << endl;
	}

	int total_sub_helios_num = helio_num * sub_num;
	outFile << "\n# Heliostats" << endl;
	outFile << "gap  0.02 0.02"<< endl;
	outFile << "matrix 1 1"<< endl;
	for (int i = 0; i < total_sub_helios_num; ++i) {
		outFile << "helio " << sub_pos[i].x << ' ' << sub_pos[i].y << ' ' << sub_pos[i].z << endl;
		outFile << sub_size[i].x << ' ' << sub_size[i].y << ' ' << sub_size[i].z << endl;
		outFileNorm << sub_normal[i].x << ' ' << sub_normal[i].y << ' ' << sub_normal[i].z << endl;
	}
	outFile.close();
	outFileNorm.close();
}


bool test_focus_helios_split() {
	//just init 
	solarenergy::scene_filepath = "../SceneData/paper/ps10/real/ps10_flat_rece.scn";
	std::string res_path = "../SceneData/paper/ps10/real/ps10_tmp18.scn";
	std::string res_norm_path = "../SceneData/paper/ps10/real/ps10_tmp18_norm.scn";
	solarenergy::sun_dir = make_float3(0.0f, -0.790155f, -1.0f);
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