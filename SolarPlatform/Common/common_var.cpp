#include "./common_var.h"

//sun ray related default value
float3 solarenergy::sun_dir = make_float3(0.0f, -0.5f, 0.866025404f);

float solarenergy::dni = 1000.0f;
float solarenergy::csr = 0.1f;
float solarenergy::num_sunshape_groups = 8.0f;
float solarenergy::num_sunshape_lights_per_group = 1024.0f;
int solarenergy::num_sunshape_lights_loop = 1;

float solarenergy::helio_pixel_length = 0.01f;
float solarenergy::receiver_pixel_length = 0.05f;
float solarenergy::image_plane_pixel_length = 0.05f;
float solarenergy::reflected_rate = 0.88f;
float solarenergy::disturb_std = 0.001f;

//conv related
int2 solarenergy::image_plane_size = {200, 200};
float solarenergy::image_plane_offset = -5.0f;
float solarenergy::kernel_ori_dis = 500.0f;
float solarenergy::total_time = 0.0f;
int solarenergy::total_times = 0;


//default scene file
string solarenergy::scene_filepath = "../SceneData/example.scn";
string solarenergy::script_filepath = "../Script/solarEnergy";