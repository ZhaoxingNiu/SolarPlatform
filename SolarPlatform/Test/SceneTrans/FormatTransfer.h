#ifndef FORMATTRANSER_H
#define FORMATTRANSER_H


#include "../../SceneProcess/solar_scene.h"
#include "../../SceneProcess/scene_file_proc.h"
#include <string>

#include <fstream>
#include <sstream>
using namespace std;

// ���ں�ʦ���ļ��������³�����ʽ�ļ���ת��

class FormatTransfer
{
public:
	void LoadFile(SolarScene* solar_scene, std::string src_path);
	void saveFile(SolarScene* solar_scene, std::string res_path);

private:

};

bool testSceneFormatTransfer();
bool testSceneFormatTransferPs10();


#endif
