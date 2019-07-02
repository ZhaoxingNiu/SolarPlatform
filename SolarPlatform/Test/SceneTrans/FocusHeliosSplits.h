#ifndef FOCUSHELIOSSPLITS_H
#define FOCUSHELIOSSPLITS_H

#include "../../SceneProcess/solar_scene.h"
#include "../../SceneProcess/scene_file_proc.h"
#include <string>
#include <vector>

#include <fstream>
#include <sstream>

// 定义: 聚焦型定日镜焦距/定日镜与接收器之间的距离
#define FOCUS_LENGTH_RATE 1.5

// 该文件用于确定定日镜场的数据,与实际镜场的聚焦数据并不一致，仅仅是为了验证一下功能

class FocusHeliosSplit {
public:
	void init(SolarScene* solar_scene);
	void split();
	void saveFile(std::string file_out_pos, std::string file_out_norm);

	SolarScene* solar_scene_;
	int helio_num_;
	int sub_num_;

	int2 row_col_;
	float2 gap_length_;
	float3 helio_size_;
	float3 helio_sub_size_;

	std::vector<float3> sub_pos_;
	std::vector<float3> sub_size_;
	std::vector<float3> sub_normal_;
	std::vector<float3> sub_vertex_;
	
	std::vector<float> focus_length_;

private:
	void localCoor();     // 确定局部坐标
	void setSurface();    // 确定抛物面
	void moveToSurface(); // 子定日镜移动到抛物面上
	void rotate();        // 定日镜整体旋转

	void transform();     // 主函数
	
};


bool testFocusHeliosSplit();


#endif