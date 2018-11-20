#ifndef RECEIVER_CUH
#define RECEIVER_CUH

#include "../Common/global_function.cuh"
#include "../Common/common_var.h"
#include "../Common/utils.h"
#include "../Common/image_saver.h"
#include <fstream>

// Receivers
class Receiver
{
public:
	// sub-class needs to redefine it
	__device__ __host__ bool GIntersect(const float3 &orig, const float3 &dir, float &t, float &u, float &v) { return true; }
	virtual void CInit(const int &geometry_info) {}

	// sub-class does NOT need to redefine it
	//__device__ void GAddEnergy(const float &u, const float &v, const float &energy);	// add energy to d_image
																							
	void Calloc_image();
	void Cclean_image_content();

	__device__ __host__ Receiver() :d_image_(nullptr) {}


	__device__ __host__ Receiver(const Receiver &rect)
	{
		type_=rect.type_;
		normal_ = rect.normal_;
		pos_ = rect.pos_;
		size_ = rect.size_;
		focus_center_ = rect.focus_center_;
		face_num_ = rect.face_num_;
		pixel_length_ = rect.pixel_length_;
		d_image_ = rect.d_image_;
		resolution_ = rect.resolution_;
	}

	// raytracing's method
	__device__ __host__ void save_result(std::string path){

		float *h_image = nullptr;
		global_func::gpu2cpu(h_image, d_image_, resolution_.x*resolution_.y);
		float dni = solarenergy::dni;
		float sub_helio_area = solarenergy::helio_pixel_length * solarenergy::helio_pixel_length;
		float rou = solarenergy::reflected_rate;
		float nc = solarenergy::num_sunshape_lights_per_group;
		float sub_rece_ares = solarenergy::receiver_pixel_length * solarenergy::receiver_pixel_length;
		for (int p = 0; p < resolution_.x * resolution_.y; ++p)
		{
			h_image[p] = h_image[p] * dni * sub_helio_area * rou / nc / sub_rece_ares;
		}
		ImageSaver::savetxt_conv(path, resolution_.x, resolution_.y, h_image);

		delete[] h_image;
		h_image = nullptr;
	}

	// convolution's method
	__device__ __host__ void save_result_conv(std::string path) {

		float *h_image = nullptr;
		global_func::gpu2cpu(h_image, d_image_, resolution_.x*resolution_.y);
		ImageSaver::savetxt_conv(path, resolution_.x, resolution_.y, h_image);

		delete[] h_image;
		h_image = nullptr;
	}

	__device__ __host__ float peek_value() {
		float *h_image = nullptr;
		global_func::gpu2cpu(h_image, d_image_, resolution_.x*resolution_.y);
		float peek_val = -1.0;
		for (int p = 0; p < resolution_.x * resolution_.y; ++p)
		{
			if (h_image[p] > peek_val) {
				peek_val = h_image[p];
			}
		}
		return peek_val;
	}

	__device__ __host__ ~Receiver()
	{
		if (d_image_)
			d_image_ = nullptr;
	}

	__device__ __host__ void CClear()
	{
		if (d_image_)
		{
			cudaFree(d_image_);
			d_image_ = nullptr;
		}
	}

	int type_;
	float3 normal_;
	float3 pos_;
	float3 size_;
	float3 focus_center_;				// fixed for a scene
	int face_num_;						// the number of receiving face
	float pixel_length_;
	float *d_image_;					// on GPU, size = resolution_.x * resolution_.y
	int2 resolution_;					// resolution.x is columns, resolution.y is rows

private:
	//__device__ __host__ void Cset_resolution(const float3 &geometry_info);
	virtual void Cset_resolution(const int &geometry_info) {}
	virtual void Cset_focuscenter() {}
};

class RectangleReceiver :public Receiver
{
public:
	__device__ __host__ RectangleReceiver() {}
	__device__ __host__ RectangleReceiver(const RectangleReceiver &rect_receiver):Receiver(rect_receiver)
	{
		rect_vertex_[0] = rect_receiver.rect_vertex_[0];
		rect_vertex_[1] = rect_receiver.rect_vertex_[1];
		rect_vertex_[2] = rect_receiver.rect_vertex_[2];
		rect_vertex_[3] = rect_receiver.rect_vertex_[3];
		localnormal_ = rect_receiver.localnormal_;
	}

	__device__ __host__ bool GIntersect(const float3 &orig, const float3 &dir, float &t, float &u, float &v)
	{
		return global_func::rayParallelogramIntersect(orig, dir, rect_vertex_[0], rect_vertex_[1], rect_vertex_[3], t, u, v);	
	}
	
	virtual void CInit(const int &geometry_info);

	float3 rect_vertex_[4];
	float3 u_axis_;
	float3 v_axis_;

private:
	void Cinit_vertex();
	void Cset_localnormal();									// set local normal
	void Cset_localvertex();									// set local vertex position
	void Cset_vertex();											// set world vertex
	void Cset_axis();                                           // set axis
	virtual void Cset_resolution(const int &geometry_info);
	virtual void Cset_focuscenter();							// call this function after Cset_vertex();

	float3 localnormal_;
};

class CylinderReceiver : public Receiver
{
public:
	__device__ __host__ CylinderReceiver() {}
	__device__ __host__ CylinderReceiver(const CylinderReceiver &cylinder_receiver):Receiver(cylinder_receiver)
	{
		radius_hight_ = cylinder_receiver.radius_hight_;
		pos_ = cylinder_receiver.pos_;
	}

	__device__ __host__ bool GIntersect(const float3 &orig, const float3 &dir, float &t, float &u, float &v) { return false; }//empty now
	//__device__ __host__ virtual void CInit();
	virtual void CInit(const int &geometry_info) {}//empty now

	float2 radius_hight_;				// radius_hight.x is radius, while radius_hight.y is hight
	float3 pos_;
private:
	virtual void Cset_resolution(const int &geometry_info) {}//empty now
	virtual void Cset_focuscenter() {}//empty now
};


class CircularTruncatedConeReceiver : public Receiver
{
public:
	__device__ __host__ CircularTruncatedConeReceiver() {}
	__device__ __host__ CircularTruncatedConeReceiver
	(const CircularTruncatedConeReceiver &cirtru_rece): Receiver(cirtru_rece)
	{
		topradius_bottomradius_hight_ = cirtru_rece.topradius_bottomradius_hight_;
	}
	__device__  __host__ bool GIntersect(const float3 &orig, const float3 &dir, float &t, float &u, float &v) { return false; }//empty now
	//__device__ __host__ virtual void CInit();
	virtual void CInit(const int &geometry_info) {}//empty now

	float3 topradius_bottomradius_hight_;	// topradius_bottomradius_hight_.x and while topradius_bottomradius_hight_.y is top radius and bottom radius respectively,
											// while radius_hight.z is hight
	
private:
	virtual void Cset_resolution(const int &geometry_info) {}//empty now
	virtual void Cset_focuscenter() {}//empty now
};

#endif // !RECEIVER_CUH