#include "./scene_instance_process.h"

//grid
void SceneProcessor::set_grid_content(vector<Grid *> &grids, const vector<Heliostat *> &heliostats)
{
	for (int i = 0; i < grids.size(); ++i)
	{
		grids[i]->Cinit();
		grids[i]->CGridHelioMatch(heliostats);
	}
}

// receiver
void SceneProcessor::set_receiver_content(vector<Receiver *> &receivers)
{
	for (int i = 0; i < receivers.size(); ++i)
		receivers[i]->CInit(int(1.0f / solarenergy::receiver_pixel_length));
}


// helio
void SceneProcessor::set_helio_content(vector<Heliostat *> &heliostats, const float3 &focus_center, const float3 &sunray_dir)
{
	for (int i = 0; i < heliostats.size(); ++i)
	{
		heliostats[i]->Cset_pixel_length(solarenergy::helio_pixel_length);
		heliostats[i]->CRotate(focus_center, sunray_dir);
	}
}

void SceneProcessor::set_helio_content(vector<Heliostat *> &heliostats, const vector<float3> &norm_vec)
{
	if (norm_vec.size() != heliostats.size()) {
		std::cerr << "the norm size is "<< norm_vec.size() <<
			" the heliostat size is "<< heliostats.size() <<std::endl;
		return;
	}
	for (int i = 0; i < heliostats.size(); ++i)
	{
		heliostats[i]->Cset_pixel_length(solarenergy::helio_pixel_length);
		heliostats[i]->CRotate(norm_vec[i]);
	}
}


bool SceneProcessor::set_helio_content(vector<Heliostat *> &heliostats, const float3 *focus_centers, const float3 &sunray_dir, const size_t &size)
{
	if (heliostats.size() != size)
		return false;

	for (int i = 0; i < heliostats.size(); ++i)
	{
		heliostats[i]->Cset_pixel_length(solarenergy::helio_pixel_length);
		heliostats[i]->CRotate(focus_centers[i], sunray_dir);
	}
	return true;
}
