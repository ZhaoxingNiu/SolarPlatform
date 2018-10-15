
#include "./solar_scene.h"
#include "./scene_file_proc.h"

SolarScene* SolarScene::m_instance;
SolarScene* SolarScene::GetInstance()
{
	if (m_instance == NULL) InitInstance();
	return m_instance;
}

void SolarScene::InitInstance()
{
	m_instance = new SolarScene();
}

SolarScene::SolarScene() {
	//init the random
	RandomGenerator::initSeed();
	//init the sunray
	sunray_ = new SunRay(solarenergy::sun_dir,solarenergy::num_sunshape_groups,solarenergy::num_sunshape_lights_per_group,
		solarenergy::dni,solarenergy::csr);
	InitSolarScece();
}

SolarScene::~SolarScene() {
	// 1. free memory on GPU
	free_scene::gpu_free(receivers);
	free_scene::gpu_free(grid0s);
	free_scene::gpu_free(sunray_);

	// 2. free memory on CPU
	free_scene::cpu_free(receivers);
	free_scene::cpu_free(grid0s);
	free_scene::cpu_free(heliostats);
	free_scene::cpu_free(sunray_);
}

bool SolarScene::InitSolarScece() {
	string filepath = solarenergy::scene_filepath;
	return LoadSceneFromFile(filepath);
}

bool SolarScene::InitSolarScene(string filepath) {
	//firstly clean the dataSpace
	// 1. free memory on GPU
	free_scene::gpu_free(receivers);
	free_scene::gpu_free(grid0s);
	free_scene::gpu_free(sunray_);

	// 2. free memory on CPU
	free_scene::cpu_free(receivers);
	free_scene::cpu_free(grid0s);
	free_scene::cpu_free(heliostats);
	free_scene::cpu_free(sunray_);

	receivers.clear();
	grid0s.clear();
	heliostats.clear();

	sunray_ = new SunRay(solarenergy::sun_dir, solarenergy::num_sunshape_groups, solarenergy::num_sunshape_lights_per_group,
		solarenergy::dni, solarenergy::csr);

	return LoadSceneFromFile(filepath);
}

bool SolarScene::LoadSceneFromFile(string filepath) {

	SceneFileProc proc;
	return proc.SceneFileRead(this, filepath);
}

bool SolarScene::InitContent()
{
	sunray_->sun_dir_ = solarenergy::sun_dir;
	// 1. Sunray
	SceneProcessor::set_sunray_content(*this->sunray_);

	// 2. Grid
	SceneProcessor::set_grid_content(this->grid0s, this->heliostats);

	// 3. Receiver
	SceneProcessor::set_receiver_content(this->receivers);

	// 4. Heliostats
	focus_center_ = this->receivers[0]->focus_center_;			// must after receiver init
	SceneProcessor::set_helio_content(this->heliostats, focus_center_, this->sunray_->sun_dir_);

	return true;
}

bool SolarScene::ResetHelioNorm(float3 foucupoint) {
	
	focus_center_ = foucupoint;
	SceneProcessor::set_helio_content(this->heliostats, focus_center_, this->sunray_->sun_dir_);
	return true;
}