# SolarPlatform
The project is a simulation platform code. The propose of this project is to create a code for the SCRS((solar central receiver systems). The project is divided into two parts: the raytracing code and a convolution model.

# 开发工具及软件版本

 - VS 2015
- cuda 9.1
- C++ 11
- python 3.6

# 目录说明

- SceneData 用于测试的场景文件
- Script 脚本文件，包括卷积核函数拟合部分的代码和绘制的代码
- SolarPlatform C++与CUDA代码部分
	- Common 变量、工具函数
 	- Convolution 卷积计算部分，包括UNIZAR和HFLCAL的计算、直接使用离散卷积核计算
   - DataStructure 数据结构，主要包括定日镜、接收器等的定义
   -  Raytracing 用于计算Raytracing的结果
   - SceneProcess 加载场景数据
   - Test 测试一些子功能的代码
- SimulResult
   - 存放结果的目录，git未跟踪 
# 配置遇到的问题

CUDA 版本不对应，导致程序无法打开
  - 打开 SolarPlatform 文件夹下的 SolarPlatform 中的SolarPlatform.vcxproj 文件
   - CUDA_PATH_V9_1 修改为对应的版本
   - CUDA 9.1.props 改为对应的版本

编译失败 打不开helper_cuda.h
  - 检查CUDA_SDK_PATH 是否有配置，是否配置正确
  - 默认目录C:\ProgramData\NVIDIA Corporation\CUDA Samples\\`version`
  - 检查CUDA版本和显卡版本
  - 根据显卡对应的配置，修改项目配置中的compute和sm值

# Reference

The ray tracing code is from a previous projects: https://github.com/XiaoyueDuan/SolarEnergy

