#ifndef SCENE_DESTORY_H
#define SCENE_DESTORY_H

#include <vector>

namespace free_scene
{
	template <typename T>
	inline void cpu_free(T *t)
	{
		if (t != nullptr)
		{
			delete t;
			t = nullptr;
		}
	}

	template <typename T>
	inline void cpu_free(std::vector<T *> Tarray)
	{
		if (Tarray.size() > 0)
		{
			for (int i = 0; i < Tarray.size(); ++i)
			{
				if (Tarray[i] != nullptr)
				{
					delete Tarray[i];
					Tarray[i] = nullptr;
				}
			}
		}
	}

	template <typename T>
	inline void gpu_free(std::vector<T *> Tarray)
	{
		if (Tarray.size() > 0)
		{
			for (int i = 0; i < Tarray.size(); ++i)
			{
				if (Tarray[i] != nullptr)				
					Tarray[i]->CClear();
			}
		}
	}

	template <typename T>
	inline void gpu_free(T * t)
	{
		if (t != nullptr)
			t->CClear();
	}
};


#endif // !SCENE_DESTORY_H