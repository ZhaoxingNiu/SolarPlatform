#include "dda_steps.h"
#include <memory>
// share_ptr : memory 

// get the vertex
bool set_helios_vertexes_cpu(
	const std::vector<Heliostat *> heliostats,
	const int start_pos,
	const int end_pos,
	float3 *&h_helio_vertexs){

	int size = end_pos - start_pos;
	h_helio_vertexs = new float3[size * 3];

	for (int i = start_pos; i < end_pos; ++i) {
		int offset = i - start_pos;
		// only save the 0 1 3 vertexs
		float3 v0, v1, v3;
		heliostats[i]->Cget_vertex(v0, v1, v3);
		h_helio_vertexs[3 * offset] = v0;
		h_helio_vertexs[3 * offset + 1] = v1;
		h_helio_vertexs[3 * offset + 2] = v3;
	}
	return true;
}

bool conv_method_kernel(
	SolarScene *solar_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	kernelType k_type
) {
	// ********************************************************************
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	// Step 1: Initialize the image plane
	ProjectionPlane plane(
		solarenergy::image_plane_size.x,
		solarenergy::image_plane_size.y,
		solarenergy::image_plane_pixel_length);

	// get the receiver and heliostat's information
	RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene->heliostats[helio_index]);
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[rece_index]);

	// calculate the normal
	float3 in_dir = solar_scene->sunray_->sun_dir_;
	float3 out_dir = reflect(in_dir, recthelio->normal_);   // reflect light
	out_dir = normalize(out_dir);

	plane.set_pos(solar_scene->receivers[rece_index]->focus_center_, -out_dir);

	// Step 2: rasterization
	dda_interface(
		*(solar_scene->sunray_),
		plane,
		*recthelio,
		*(solar_scene->grid0s[grid_index]),
		solar_scene->heliostats
	);

#ifdef _DEBUG
	std::string image_path = "../SimulResult/imageplane/image_debug.txt";
	plane.save_data_text(image_path);
#endif

	// Step 4: projection the image plane to the heliostat
	// Step 4.1: get the projection matrix

	oblique_proj_matrix(
		out_dir,
		plane.normal,
		out_dir,
		plane.M,
		plane.offset
	);

	// Step 4.2: load the kernel
	// calc the true angel and distance
	float true_dis = length(recthelio->pos_
		- solar_scene->receivers[rece_index]->focus_center_);
	float true_angel = acosf(dot(-in_dir, out_dir)) * 180 / MATH_PI;
	int round_distance = round(true_dis);
	int round_angel = round(true_angel);
	
	std::shared_ptr<ConvKernel> kernel;
	// chose the proper kernel
	switch (k_type)
	{
	case T_CONV: 
	{
		std::cerr << " the ConvKernel is not support" << std::endl;
		break;
	}
	case T_LOADED_CONV: {
		gen_kernel(true_dis, 500.0f, true_angel);
		std::string fit_kernel_path = "../SimulResult/data/gen_flux/onepoint_angle_" +
			std::to_string(round_angel) + "_distance_" + std::to_string(round_distance) + ".txt";
		kernel = std::make_shared<LoadedConvKernel>(LoadedConvKernel(201, 201, fit_kernel_path));
		break;
	}
	case T_GAUSSIAN_CONV: {
		std::string gaussian_kernel_path = "../SimulResult/data/gen_flux_gau/onepoint_angle_" +
			std::to_string(round_angel) + "_distance_" + std::to_string(round_distance) + ".txt";
		kernel = std::make_shared<LoadedConvKernel>(LoadedConvKernel(201, 201, gaussian_kernel_path));
		break;
	}
	default:
		break;
	}

	kernel->genKernel();
	fastConvolutionDevice(
		plane.get_deviceData(),
		kernel->d_data,
		plane.rows,
		plane.cols,
		kernel->dataH,
		kernel->dataW
	);

#ifdef _DEBUG
	std::string image_path2 = "../SimulResult/imageplane/image_debug2.txt";
	plane.save_data_text(image_path2);
#endif
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	projection_plane_rect(
		(solar_scene->receivers[rece_index])->d_image_,
		plane.get_deviceData(),
		rectrece,
		&plane,
		plane.M,
		plane.offset);

	sdkStopTimer(&hTimer);

	double gpuTime = sdkGetTimerValue(&hTimer);
	printf("projection cost time: (%f ms)\n", gpuTime);
	return true;
}