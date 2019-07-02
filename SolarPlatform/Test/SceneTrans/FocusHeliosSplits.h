#ifndef FOCUSHELIOSSPLITS_H
#define FOCUSHELIOSSPLITS_H

#include "../../SceneProcess/solar_scene.h"
#include "../../SceneProcess/scene_file_proc.h"
#include <string>
#include <vector>

#include <fstream>
#include <sstream>

// ����: �۽��Ͷ��վ�����/���վ��������֮��ľ���
#define FOCUS_LENGTH_RATE 1.5

// ���ļ�����ȷ�����վ���������,��ʵ�ʾ����ľ۽����ݲ���һ�£�������Ϊ����֤һ�¹���

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
	void localCoor();     // ȷ���ֲ�����
	void setSurface();    // ȷ��������
	void moveToSurface(); // �Ӷ��վ��ƶ�����������
	void rotate();        // ���վ�������ת

	void transform();     // ������
	
};


bool testFocusHeliosSplit();


#endif