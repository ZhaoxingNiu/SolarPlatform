#ifndef FORMATTRANSER_H
#define FORMATTRANSER_H


#include "../../SceneProcess/solar_scene.h"
#include "../../SceneProcess/scene_file_proc.h"
#include <string>

#include <fstream>
#include <sstream>
using namespace std;

// 用于何师兄文件数据与新场景格式文件的转换

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
