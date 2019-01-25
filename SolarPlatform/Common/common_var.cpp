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
bool solarenergy::kernel_ori_flush = false;
int solarenergy::kernel_cols = 201; // the genKernel's size
int solarenergy::kennel_rows = 201; // the genKernel's size
float solarenergy::kernel_width = 10.05f;
float solarenergy::kernel_height = 10.05f;

float solarenergy::kernel_step_r = 0.05f; // static size
float solarenergy::kernel_grid_len = 0.05f;
float solarenergy::kernel_distance_threshold = 0.1f;  // a in the paper
float solarenergy::kernel_rece_width = 20.05f;   // receiver's size
float solarenergy::kernel_rece_height = 20.05f;  // receiver's size
float solarenergy::kernel_rece_max_r = 7.0f;

// time settings
float solarenergy::total_time = 0.0f;
int solarenergy::total_times = 0;


//default scene file
string solarenergy::scene_filepath = "../SceneData/example.scn";
string solarenergy::script_filepath = "../Script/solarEnergy";