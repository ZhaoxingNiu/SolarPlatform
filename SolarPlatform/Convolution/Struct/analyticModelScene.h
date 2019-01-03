#ifndef ANALYTIC_MODEL_SCENE_H
#define ANALYTIC_MODEL_SCENE_H

#include "./projectionPlane.h"
#include "../../Common/common_var.h"
#include "../../SceneProcess/solar_scene.h"

/*
* description: ���������еı������亯�� 
* ��Ҫ���ڽ��������г����ظ�����ռ�����⣬��Ҫ����imageplane��  
* 3D DDA�ж�������ķ��䣬 ��������пռ�ķ���
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