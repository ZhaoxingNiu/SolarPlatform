#ifndef SOLAR_SCENE
#define SOLAR_SCENE

#include "../Common/common_var.h"
#include "../Common/random_generator.h"

#include "../DataStructure/grid.h"
#include "../DataStructure/heliostat.cuh"
#include "../DataStructure/receiver.cuh"
#include "../DataStructure/sunray.h"

#include "./Preprocess/scene_instance_process.h"
#include "./scene_destroy.h"

//Singleton design model to  control the  access to resources
class SolarScene {
protected:
	SolarScene();

public:
	static SolarScene* GetInstance();   //static member
	static void InitInstance();
	~SolarScene();

	bool InitSolarScene(string filepath);
	bool InitContent();					// Call the method only if all grids, heliostats and receivers needs initializing. 
	bool ResetHelioNorm(float3 foucupoint);			    // reset heliostats focus point
	bool ResetHelioNorm(const std::vector<float3> &norm_vec);			    // reset heliostats focus point

private:
	static SolarScene *m_instance;		//Singleton
	bool InitSolarScece();              // only used in the InitInstance
	bool LoadSceneFromFile(string filepath);

public:
	float ground_length_;
	float ground_width_;
	int grid_num_;
	
	SunRay *sunray_;
	float3 focus_center_;  // focus point
	//scene object
	vector<Grid *> grid0s;
	vector<Heliostat *> heliostats;
	vector<Receiver *> receivers;
};

#endif // !SOLAR_SCENE