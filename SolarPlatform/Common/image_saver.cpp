#include "image_saver.h"

void ImageSaver::savetxt(const string filename, int w, int h, float *h_data)
{
	ofstream out(filename.c_str());

	int address = 0;
	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{	
			address = (h - 1 - r)*w + c;
			if (c) {
				out <<","<< h_data[address];
			}
			else {
				out << h_data[address];
			}
			
		}
		out << endl;
	}
	out.close();
}