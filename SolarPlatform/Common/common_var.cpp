#include "./common_var.h"


//sun ray related default value
float3 solarenergy::sun_dir = make_float3(0.0f, -0.5f, 0.866025404f);

float solarenergy::dni = 1000.0f;
float solarenergy::csr = 0.2f;
float solarenergy::num_sunshape_groups = 8;
float solarenergy::num_sunshape_lights_per_group = 1024;

float solarenergy::helio_pixel_length = 0.01;
float solarenergy::receiver_pixel_length = 0.01;
float solarenergy::reflected_rate = 0.88;
float solarenergy::disturb_std = 0.002;

//default scene file
string  solarenergy::scene_filepath = "../SceneData/example.scn";