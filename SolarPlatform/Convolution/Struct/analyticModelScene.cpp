#include "./analyticModelScene.h"
#include "../3DDDA/dda_interface.h"


AnalyticModelScene* AnalyticModelScene::m_instance;
AnalyticModelScene* AnalyticModelScene::GetInstance() {
	if (m_instance == NULL) InitInstance();
	return m_instance;
}

void AnalyticModelScene::InitInstance()
{
	m_instance = new AnalyticModelScene();
}

AnalyticModelScene::~AnalyticModelScene() {
	cleanContent();
}

bool AnalyticModelScene::InitContent(SolarScene *solar_scene) {
	cleanContent();
	// reset the plane 
	InitProjectionPlane();
	InitGridVertex(solar_scene);
	return true;
}

bool AnalyticModelScene::cleanContent() {
	if (plane != nullptr) {
		delete plane;
	}
	if (plane_total != nullptr) {
		delete plane_total;
	}
	//clean the grid_vertexs
	for (auto h_helio_vertexs : grid_vertexs) {
		delete[] h_helio_vertexs;
		h_helio_vertexs = nullptr;
	}
	grid_vertexs.clear();
	return true;
}

AnalyticModelScene::AnalyticModelScene() {
	plane = nullptr;
	plane_total = nullptr;
}


bool AnalyticModelScene::InitProjectionPlane() {
	plane = new ProjectionPlane(
		solarenergy::image_plane_size.x,
		solarenergy::image_plane_size.y,
		solarenergy::image_plane_pixel_length);
	plane_total = new ProjectionPlane(
		solarenergy::image_plane_size.x,
		solarenergy::image_plane_size.y,
		solarenergy::image_plane_pixel_length);

	return true;
}

bool AnalyticModelScene::InitGridVertex(SolarScene *solar_scene) {
	for (Grid *grid:solar_scene->grid0s) {
		float3 *h_helio_vertexs = nullptr;
		int start_pos = grid->start_helio_pos_;
		int end_pos = start_pos + grid->num_helios_;
		set_helios_vertexes_cpu(solar_scene->heliostats, start_pos, end_pos, 
			h_helio_vertexs);
		grid_vertexs.push_back(h_helio_vertexs);
	}
	return true;
}