#ifndef HELIOSTAT_CUH
#define HELIOSTAT_CUH

#include "../Common/utils.h"
#include "../Common/global_function.cuh"

// Heliostats
class Heliostat
{
public:
	float3 pos_;
	float3 size_;
	float3 normal_;
	int2 row_col_;		// How many mirrors compose a heliostat
	float2 gap_;		// The gap between mirrors

	__device__ __host__ Heliostat() {}

	virtual void Cset_pixel_length(const float &pixel_length) = 0;
	virtual void CRotate(const float3 &focus_center, const float3 &sunray_dir) = 0;
	virtual void CRotate(const float3 &norm_dir) = 0;
	virtual void Cget_vertex(float3 &v0, float3 &v1, float3 &v3) = 0;
	virtual void Cget_all_vertex(float3 &v0, float3 &v1, float3 &v2, float3 &v3) = 0;
	virtual bool Cray_intersect(const float3 ori, const float3 dir, float3 &p) const = 0;
};

class RectangleHelio :public Heliostat
{
public:
	__device__ __host__ RectangleHelio() {}
	virtual void Cset_pixel_length(const float &pixel_length);
	//void virtual Cset_sub_row_col(const float &pixel_length);
	//__device__ __host__ virtual bool GIntersect(const float3 &orig, const float3 &dir)	// whether the light with orig and dir can intersect with this heliostat
	//{
	//	float t, u, v;
	//	return global_func::rayParallelogramIntersect(orig, dir, vertex_[0], vertex_[1], vertex_[3], t, u, v);
	//}
	virtual void CRotate(const float3 &focus_center, const float3 &sunray_dir);
	virtual void CRotate(const float3 &norm_dir);
	virtual void Cget_vertex(float3 &v0, float3 &v1, float3 &v3)
	{
		v0 = vertex_[0];
		v1 = vertex_[1];
		v3 = vertex_[3];
	}

	virtual void  Cget_all_vertex(float3 &v0, float3 &v1, float3 &v2, float3 &v3) {
		v0 = vertex_[0];
		v1 = vertex_[1];
		v2 = vertex_[2];
		v3 = vertex_[3];
	}

	virtual bool Cray_intersect(const float3 ori, const float3 dir, float3 &p) const {
		float t = (dot(normal_, pos_) - dot(normal_, ori)) / dot(normal_, dir);
		p = ori + t * dir;
		if (t < 0) {
			return false;
		}
		return true;
	}

	float3 vertex_[4];
	//int2 sub_row_col_;	// How many submirrors compose a mirror
	float pixel_length_;

private:
	void Cset_localvertex();
	void Cset_worldvertex();
	void RectangleHelio::Cset_normal(const float3 &focus_center, const float3 &sunray_dir);
};
 
class ParaboloidHelio :public Heliostat	// has-RectangleHelio
{
public:
	__device__ __host__ ParaboloidHelio() {}
	virtual void Cset_pixel_length(const float &pixel_length)
	{
		invisual_recthelio_.Cset_pixel_length(pixel_length);
	}
	virtual void Cget_vertex(float3 &v0, float3 &v1, float3 &v3)
	{
		invisual_recthelio_.Cget_vertex(v0, v1, v3);
	}
	virtual void  Cget_all_vertex(float3 &v0, float3 &v1, float3 &v2, float3 &v3)
	{
		invisual_recthelio_.Cget_all_vertex(v0, v1, v2, v3);
	}

	//__device__ __host__ virtual bool GIntersect(const float3 &orig, const float3 &dir)	// whether the light with orig and dir can intersect with this heliostat
	//{
	//	return invisual_recthelio_.GIntersect(orig, dir);
	//}

	virtual void CRotate(const float3 &focus_center, const float3 &sunray_dir) {}

	virtual bool Cray_intersect(const float3 ori, const float3 dir, float3 &p) const{
		invisual_recthelio_.Cray_intersect(ori, dir, p);
	}

	float2 a_b;					// y = x^2/a^2 + z^2/b^2

private:
	RectangleHelio invisual_recthelio_;
};

#endif // !HELIOSTAT_CUH