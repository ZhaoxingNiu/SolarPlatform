#include "../../Common/global_constant.h"
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
	float3 normal,
	kernelType k_type,
	float sigma_2
) {
	// ********************************************************************
	StopWatchInterface *hTimer = NULL;
	double gpuTime = 0.0;
	sdkCreateTimer(&hTimer);

	// Step 1: Initialize the image plane
	ProjectionPlane plane(
		solarenergy::image_plane_size.x,
		solarenergy::image_plane_size.y,
		solarenergy::image_plane_pixel_length);

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// get the receiver and heliostat's information
	RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene->heliostats[helio_index]);
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[rece_index]);

	// calculate the normal
	float3 in_dir = solar_scene->sunray_->sun_dir_;
	float3 out_dir = reflect(in_dir, recthelio->normal_);   // reflect light
	out_dir = normalize(out_dir);

	// if seted normal, otherwise the normal is generated
	if (length(normal) == 0.0f) {
		plane.set_pos(solar_scene->receivers[rece_index]->focus_center_, -out_dir);
	}
	else {
		plane.set_pos(solar_scene->receivers[rece_index]->focus_center_,normal);
	}

	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer);
	solarenergy::total_time += gpuTime;
	printf("calculation the normal: (%f ms)\n", gpuTime);

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

	// Step 3: init the kernel
	// Step 3.1: get the projection matrix
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	oblique_proj_matrix(
		out_dir,
		plane.normal,
		out_dir,
		plane.M,
		plane.offset
	);

	// Step 3.2: load the kernel
	// calc the true angel and distance
	float true_dis = length(recthelio->pos_
		- solar_scene->receivers[rece_index]->focus_center_);
	float true_angel = acosf(dot(-in_dir, out_dir)) * 180 / MATH_PI;
	int round_distance = round(true_dis);
	int round_angel = round(true_angel);
	
	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer);
	solarenergy::total_time += gpuTime;

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
	case T_GAUSSIAN_CONV_MATLAB: {
		std::string gaussian_kernel_path = "../SimulResult/data/gen_flux_gau/onepoint_angle_" +
			std::to_string(round_angel) + "_distance_" + std::to_string(round_distance) + ".txt";
		kernel = std::make_shared<LoadedConvKernel>(LoadedConvKernel(201, 201, gaussian_kernel_path));
		break;
	}
	case T_GAUSSIAN_CONV: {
		float A;
		gen_gau_kernel_param(true_dis, A);
		std::string fit_kernel_path = "../SimulResult/data/gen_flux/onepoint_angle_" +
			std::to_string(round_angel) + "_distance_" + std::to_string(round_distance) + ".txt";
		kernel = std::make_shared<GaussianConvKernel>(GaussianConvKernel(201, 201, A, sigma_2,
			solarenergy::image_plane_pixel_length, solarenergy::image_plane_offset));
		break;
	}
	default:
		break;
	}

	kernel->genKernel();

	// Step 4: convolution calculation
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

	// Step 5: projection
	projection_plane_rect(
		(solar_scene->receivers[rece_index])->d_image_,
		plane.get_deviceData(),
		rectrece,
		&plane,
		plane.M,
		plane.offset);

	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer);
	solarenergy::total_time += gpuTime;
	printf("projection cost time: (%f ms)\n", gpuTime);
	return true;
}

bool conv_method_kernel_HFLCAL(
	SolarScene *solar_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	float3 normal,
	float sigma_2
) {
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	// Step 1: Initialize the image plane
	ProjectionPlane plane(
		solarenergy::image_plane_size.x,
		solarenergy::image_plane_size.y,
		solarenergy::image_plane_pixel_length);

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// get the receiver and heliostat's information
	RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene->heliostats[helio_index]);
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[rece_index]);

	// calculate the normal
	float3 in_dir = solar_scene->sunray_->sun_dir_;
	float3 out_dir = reflect(in_dir, recthelio->normal_);   // reflect light
	out_dir = normalize(out_dir);

	// if seted normal, otherwise the normal is generated
	if (length(normal) == 0.0f) {
		plane.set_pos(solar_scene->receivers[rece_index]->focus_center_, -out_dir);
	}
	else {
		plane.set_pos(solar_scene->receivers[rece_index]->focus_center_, normal);
	}
	// the hflcal model generate the kernel on the image plane 
	float true_dis = length(recthelio->pos_
		- solar_scene->receivers[rece_index]->focus_center_);
	float air_atten = global_func::air_attenuation(true_dis);
	float3 v0, v1, v2, v3;
	recthelio->Cget_all_vertex(v0, v1, v2, v3);
	float area = global_func::cal_rect_area(v0, v1, v2, v3);
	float cos_val = abs(dot(in_dir, recthelio->normal_));
	float total_energy = air_atten*solarenergy::dni*solarenergy::reflected_rate
		*area*abs(dot(in_dir, recthelio->normal_));
	plane.gen_gau_kernel(total_energy,sigma_2);

#ifdef _DEBUG
	std::string image_path = "../SimulResult/imageplane/image_debug.txt";
	plane.save_data_text(image_path);
#endif

	// Step 3: init the kernel
	// Step 3.1: get the projection matrix

	oblique_proj_matrix(
		out_dir,
		plane.normal,
		out_dir,
		plane.M,
		plane.offset
	);

	// projection
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
	solarenergy::total_time += gpuTime;
	return true;
}