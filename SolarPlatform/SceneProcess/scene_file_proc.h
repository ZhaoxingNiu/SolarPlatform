#ifndef SCENE_FILE_PROC
#define SCENE_FILE_PROC

#include "./solar_scene.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <string>

// Value-Defintions of the different String values
enum StringValue {
	pos,
	size,
	norm,
	face,
	end,
	gap,
	matrix,
	helio,
	inter,
	n,
	type,
	illegal
};

// Value-Defintions of the different String values
enum InputMode{
	none,
	ground,
	receiver,
	grid,
	heliostat
};


class SceneFileProc{
public:
	SceneFileProc();
	bool SceneFileRead(SolarScene *solarscene, std::string filepath);

	static void SceneNormalRead(std::string filepath, std::vector<float3> &norm_vec);
private:
	SolarScene *solarScene_;  //eqaul the Scolar::GetInstance
	// Map to associate the strings with the enum values
	std::map<std::string, StringValue> string_value_read_map;
	StringValue Str2Value(string str);

};

#endif // !SCENE_FILE_PROC