#ifndef ANALYTIC_MODEL_SCENE_H
#define ANALYTIC_MODEL_SCENE_H

#include "./projectionPlane.h"
#include "../../Common/common_var.h"
#include "../../SceneProcess/solar_scene.h"

/*
* description: 解析方法中的变量分配函数 
* 主要用于解析方法中场景重复分配空间的问题，主要包括imageplane，  
* 3D DDA中顶点数组的分配， 卷积计算中空间的分配
*/

class AnalyticModelScene {
public:
	static AnalyticModelScene* GetInstance();   //static member
	static void InitInstance();
	~AnalyticModelScene();

	bool InitContent(SolarScene *solar_scene);
	bool cleanContent();


	ProjectionPlane *plane;
	ProjectionPlane *plane_total;
	std::vector<float3 *> grid_vertexs;

private:
	AnalyticModelScene();
	static AnalyticModelScene *m_instance;		//Singleton
	bool InitProjectionPlane();
	bool InitGridVertex(SolarScene *solar_scene);
};






#endif //ANALYIIC_MODEL_SCENE_H