#include "../../Common/global_constant.h"
#include "dda_steps.h"
#include <memory>
// share_ptr : memory 

bool conv_method_kernel(
	SolarScene *solar_scene,
	AnalyticModelScene *model_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	float3 normal,
	kernelType k_type,
	float sigma_2
) {
	StopWatchInterface *hTimer = NULL;
	double gpuTime = 0.0;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// Step 1: Initialize the image plane
	ProjectionPlane &plane = *(model_scene->plane);
	plane.clean_image_content();
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
		solar_scene->heliostats,
		model_scene->grid_vertexs[grid_index]
	);
	

#ifdef _DEBUG
	std::string image_path = "../SimulResult/imageplane/ps10_sub_image_"+std::to_string(helio_index)+".txt";
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
	// true_angel 对应的是 入射角 与 出射角 之间的关系
	float true_angel = acosf(dot(-in_dir, out_dir)) * 180 / MATH_PI;
	int round_distance = round(true_dis);
	int round_angel = round(true_angel);
	
	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer);
	solarenergy::total_time += gpuTime;
	std::cout << "calculation distance time: " << gpuTime << " ms" << std::endl;

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
		int round_ori_dis = round(solarenergy::kernel_ori_dis);
		gen_kernel(round_distance, solarenergy::kernel_ori_dis, round_angel, true,
			0.05f, 0.05f, 0.1f, 20.05f, 20.05f, 14.0f);
		//std::string fit_kernel_path = "../SimulResult/data/gen_flux/onepoint_angle_" +
		//	std::to_string(round_angel) + "_distance_" + std::to_string(round_distance) + ".txt";
		std::string fit_kernel_path = "../SimulResult/data/gen_flux_dst/" + std::to_string(round_ori_dis) + "/distance_"
			+ std::to_string(round_distance) + "_angle_" + std::to_string(round_angel) + ".txt";
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

bool conv_method_kernel_focus(
	SolarScene *solar_scene,
	AnalyticModelScene *model_scene,
	int rece_index,
	int helio_index,
	int sub_num,
	int grid_index,
	kernelType k_type,
	float sigma_2
) {
	StopWatchInterface *hTimer = NULL;
	double gpuTime = 0.0;
	std::cout << "\n****** Index: " << std::to_string(helio_index) << "******" << std::endl;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	// Step 1: Initialize the image plane
	ProjectionPlane &plane_total = *(model_scene->plane_total);
	ProjectionPlane &plane = *(model_scene->plane);
	plane_total.clean_image_content();
	plane.clean_image_content();

	float3 average_normal = make_float3(0.0f,0.0f,0.0f);
	float3 average_out_dir = make_float3(0.0f, 0.0f, 0.0f);
	float3 in_dir = solar_scene->sunray_->sun_dir_;
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[rece_index]);

	float average_dis = 0.0f;
	float average_angle = 0.0f;
	// calculatation the  average
	for (int i = helio_index*sub_num; i < (helio_index + 1)*sub_num; ++i) {
		// get the receiver and heliostat's information
		RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene->heliostats[i]);
		average_normal += recthelio->normal_;
		float true_dis = length(recthelio->pos_
			- solar_scene->receivers[rece_index]->focus_center_);
		average_dis += true_dis;
	}
	average_normal = normalize(average_normal);
	average_out_dir = reflect(in_dir, average_normal);   // reflect light
	average_out_dir = normalize(average_out_dir);
	average_dis /= sub_num;
	average_angle  = acosf(dot(-in_dir, average_out_dir)) * 180 / MATH_PI;

	// set the image plane
	plane_total.set_pos(solar_scene->receivers[rece_index]->focus_center_, -average_out_dir);
	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer);
	solarenergy::total_time += gpuTime;
	printf("init the image plane cost : (%f ms)\n", gpuTime);


	// 累加投影区域
	for (int i = helio_index*sub_num; i < (helio_index + 1)*sub_num; ++i) {
		plane.clean_image_content();
		RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene->heliostats[i]);
		plane.set_pos(solar_scene->receivers[rece_index]->focus_center_, -average_out_dir);
		// Step 2: rasterization
		dda_interface(
			*(solar_scene->sunray_),
			plane,
			*recthelio,
			*(solar_scene->grid0s[grid_index]),
			solar_scene->heliostats,
			model_scene->grid_vertexs[grid_index]
		);
#ifdef _DEBUG
		std::string image_path = "../SimulResult/imageplane/ps10_real_plane_#" + std::to_string(helio_index) +
			"_"+ std::to_string(i- helio_index*sub_num) + ".txt";
		plane.save_data_text(image_path);
#endif
		// 累加投影区域能量
		plane_total.accumuluation(plane);
	}

#ifdef _DEBUG
	std::string image_path = "../SimulResult/imageplane/ps10_plane_#"+ std::to_string(helio_index) +".txt";
	plane_total.save_data_text(image_path);
#endif
	// Step 3.2: load the kernel
	// calc the true angel and distance
	int round_distance = round(average_dis);
	int round_angel = round(average_angle);
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
		int round_ori_dis = round(solarenergy::kernel_ori_dis);
		gen_kernel(round_distance,solarenergy::kernel_ori_dis, round_angel, true,
			0.05f,0.05f,0.1f,20.05f,20.05f,14.0f);
		std::string fit_kernel_path = "../SimulResult/data/gen_flux_dst/" + std::to_string(round_ori_dis) + "/distance_"
			+ std::to_string(round_distance) +"_angle_"+ std::to_string(round_angel) + ".txt";
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
		gen_gau_kernel_param(round_distance, A);
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
		plane_total.get_deviceData(),
		kernel->d_data,
		plane_total.rows,
		plane_total.cols,
		kernel->dataH,
		kernel->dataW
	);

#ifdef _DEBUG
	std::string image_path2 = "../SimulResult/imageplane/ps10_plane_energy_#" + std::to_string(helio_index) + ".txt";
	plane_total.save_data_text(image_path2);
#endif
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	oblique_proj_matrix(
		average_out_dir,
		plane_total.normal,
		average_out_dir,
		plane_total.M,
		plane_total.offset
	);

	// Step 5: projection
	projection_plane_rect(
		(solar_scene->receivers[rece_index])->d_image_,
		plane_total.get_deviceData(),
		rectrece,
		&plane_total,
		plane_total.M,
		plane_total.offset);

	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer);
	solarenergy::total_time += gpuTime;
	printf("projection cost time: (%f ms)\n", gpuTime);
	return true;
}


bool conv_method_kernel_HFLCAL(
	SolarScene *solar_scene,
	AnalyticModelScene *model_scene,
	int rece_index,
	int helio_index,
	int grid_index,
	float sigma_2
) {
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	// Step 1: Initialize the image plane
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	ProjectionPlane &plane = *(model_scene->plane);
	plane.clean_image_content();
	// get the receiver and heliostat's information
	RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene->heliostats[helio_index]);
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[rece_index]);

	// calculate the normal
	float3 in_dir = solar_scene->sunray_->sun_dir_;
	float3 out_dir = reflect(in_dir, recthelio->normal_);   // reflect light
	out_dir = normalize(out_dir);

	// if seted normal, otherwise the normal is generated
	plane.set_pos(solar_scene->receivers[rece_index]->focus_center_, -out_dir);

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

bool conv_method_kernel_HFLCAL_focus(
	SolarScene *solar_scene,
	AnalyticModelScene *model_scene,
	int rece_index,
	int helio_index,
	int sub_num,
	int grid_index,
	float sigma_2
) {
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	// Step 1: Initialize the image plane
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	// Step 1: Initialize the image plane
	ProjectionPlane &plane = *(model_scene->plane);
	plane.clean_image_content();

	float3 average_normal = make_float3(0.0f, 0.0f, 0.0f);
	float3 average_out_dir = make_float3(0.0f, 0.0f, 0.0f);
	float3 in_dir = solar_scene->sunray_->sun_dir_;
	float total_area = 0.0f;
	RectangleReceiver *rectrece = dynamic_cast<RectangleReceiver *>(solar_scene->receivers[rece_index]);
	float average_dis = 0.0f;
	float average_angle = 0.0f;
	// calculatation the  average
	for (int i = helio_index*sub_num; i < (helio_index + 1)*sub_num; ++i) {
		// get the receiver and heliostat's information
		RectangleHelio *recthelio = dynamic_cast<RectangleHelio *>(solar_scene->heliostats[i]);
		average_normal += recthelio->normal_;

		float true_dis = length(recthelio->pos_
			- solar_scene->receivers[rece_index]->focus_center_);

		float3 v0, v1, v2, v3;
		recthelio->Cget_all_vertex(v0, v1, v2, v3);
		float area = global_func::cal_rect_area(v0, v1, v2, v3);
		total_area += area;
		average_dis += true_dis;
	}
	average_normal = normalize(average_normal);
	average_out_dir = reflect(in_dir, average_normal); 
	average_out_dir = normalize(average_out_dir);
	average_dis /= sub_num;
	average_angle = acosf(dot(-in_dir, average_normal)) * 180 / MATH_PI;

	// set the image plane
	plane.set_pos(solar_scene->receivers[rece_index]->focus_center_, -average_out_dir);


	float air_atten = global_func::air_attenuation(average_dis);
	float cos_val = abs(dot(-in_dir, average_normal));

	float total_energy = air_atten*solarenergy::dni*solarenergy::reflected_rate
		*total_area*cos_val;
	plane.gen_gau_kernel(total_energy, sigma_2);

#ifdef _DEBUG
	std::string image_path = "../SimulResult/imageplane/image_debug.txt";
	plane.save_data_text(image_path);
#endif

	// Step 3: init the kernel
	// Step 3.1: get the projection matrix

	oblique_proj_matrix(
		average_out_dir,
		plane.normal,
		average_out_dir,
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