#ifndef IMAGE_SAVER_H
#define IMAGE_SAVER_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class ImageSaver 
{
public:	
	static void savetxt(const string filename, int w, int h, float *h_data);
	static void savetxt_conv(const string filename, int w, int h, float *h_data);
};

#endif // !IMAGE_SAVER_H