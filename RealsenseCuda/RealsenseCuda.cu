
#include "cuda_runtime.h"
#include <librealsense2/rs.hpp>
#include <stdio.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "virtual_output.h"

__global__ void gammaKernel(char3* _dst, const char3* _src, const unsigned short* _depth, const char3* _background, int _w, int _h, float scale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;
	int pos = y * _w + x;

	for(int i = 0; i < 3; i++)
	{
		if (_depth[pos] != 0 && /*_src[pos] != {0, 0, 0} &&*/ _depth[pos] * scale < 1.0) {
			_dst[pos] = _src[pos];
		} else {
			_dst[pos] = _background[pos];
		}
	}
}

#define MAX_THREADS 128

enum class direction
{
	to_depth,
	to_color
};

int main(int argc, char* argv[]) try
{
	cv::Mat background = cv::imread(".\\background.jpg", cv::ImreadModes::IMREAD_COLOR);

	const int w = 1280;
	const int h = 720;

	//VirtualOutput virtual_output(w, h, 30, libyuv::FOURCC_BGR3);

	int nPix = w * h;
	char3* gpuImg;
	unsigned short* gpuDepthImg;
	char3* gpuResImg;
	char3* gpuBackgroundImg;
	cudaMalloc((void**)&gpuImg, nPix * sizeof(char3));
	cudaMalloc((void**)&gpuDepthImg, nPix * sizeof(unsigned short));
	cudaMalloc((void**)&gpuResImg, nPix * sizeof(char3));
	cudaMalloc((void**)&gpuBackgroundImg, nPix * sizeof(char3));

	char* cpuImg;
	cpuImg = (char*)malloc(nPix * 3 * sizeof(char));

	std::string serial;
	rs2::pipeline pipe;
	rs2::config cfg;
	if (!serial.empty())
		cfg.enable_device(serial);
	cfg.enable_stream(RS2_STREAM_COLOR, -1, 1280, 720, rs2_format::RS2_FORMAT_BGR8, 0);
	cfg.enable_stream(RS2_STREAM_DEPTH, -1, 1280, 720, rs2_format::RS2_FORMAT_Z16, 0);
	rs2::pipeline_profile profile = pipe.start(cfg);

	rs2::device dev = profile.get_device();
	rs2::depth_sensor ds = dev.query_sensors().front().as<rs2::depth_sensor>();
	float scale = ds.get_depth_scale();

	rs2::align align_to_depth(RS2_STREAM_DEPTH);
	rs2::align align_to_color(RS2_STREAM_COLOR);

	direction dir = direction::to_depth;

	while (true)
	{
		rs2::frameset frameset = pipe.wait_for_frames();

		if (dir == direction::to_depth){
			frameset = align_to_depth.process(frameset);
		} else {
			frameset = align_to_color.process(frameset);
		}

		auto depth = frameset.get_depth_frame();
		auto color = frameset.get_color_frame();

		cudaMemcpy(gpuImg, color.get_data(), nPix * 3 * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(gpuDepthImg, depth.get_data(), nPix * sizeof(unsigned short), cudaMemcpyHostToDevice);
		cudaMemcpy(gpuBackgroundImg, background.data, nPix * 3 * sizeof(char), cudaMemcpyHostToDevice);

		dim3 threadBlock(MAX_THREADS);
		dim3 blockGrid(w / MAX_THREADS + 1, h, 1);

		gammaKernel<<<blockGrid, threadBlock >>> (gpuResImg, gpuImg, gpuDepthImg, gpuBackgroundImg, w, h, scale);

		cudaMemcpy(cpuImg, gpuResImg, nPix * 3 * sizeof(char), cudaMemcpyDeviceToHost);

		cv::Mat my_mat(h, w, CV_8UC3, &cpuImg[0]);

		cv::namedWindow("Image window", cv::WINDOW_AUTOSIZE);
		cv::imshow("Image window", my_mat);

		//virtual_output.send((const uint8_t*)cpuImg);

		if ((char)cv::waitKey(25) == 27)
			break;
		
	}

	//virtual_output.stop();

	cudaFree(gpuImg);
	cudaFree(gpuDepthImg);
	cudaFree(gpuDepthImg);
	cudaFree(gpuResImg);
	free(cpuImg);

	return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception& e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}