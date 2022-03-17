
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <librealsense2/rs.hpp>
#include <stdio.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include "virtual_output.h"

VirtualOutput virtual_output(1280, 720, 30, libyuv::FOURCC_BGR3);

__global__ void gammaKernel(char* _dst, const char* _src, const unsigned short* _depth, const char* _background, int _w, int _h, float scale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;
	int pos = y * _w + x;

	if (x < _w)
	{
		if (_depth[pos / 3] * scale < 1.0 && _src[pos] > 0) {
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
	printf("Hello\n");
	if (!glfwInit())
	{
		printf("Failed to initialize GLFW\n");
		return -1;
	}

	int winWidth = 1280;
	int winHeight = 720;

	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 

	GLFWwindow* window; // (In the accompanying source code, this variable is global for simplicity)
	window = glfwCreateWindow(winWidth, winHeight, "Tutorial 01", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window); // Initialize GLEW
	glewExperimental = true; // Needed in core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	GLuint to_id = 0;
	glGenTextures(1, &to_id);
	glBindTexture(GL_TEXTURE_2D, to_id);
	glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGB8, 1280, 720);

	GLenum errCode1;
	if ((errCode1 = glGetError()) != GL_NO_ERROR)
	{
		printf("First: %s \n", gluErrorString(errCode1));
	}

	GLuint readFboId = 0;
	glGenFramebuffers(1, &readFboId);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, readFboId);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, to_id, 0);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

	GLenum errCode2;
	if ((errCode2 = glGetError()) != GL_NO_ERROR)
	{
		printf("Second: %s \n", gluErrorString(errCode2));
	}

	cv::Mat background = cv::imread(".\\background.jpg", cv::ImreadModes::IMREAD_COLOR);

	const int w = 1280;
	const int h = 720;

	int nPix = w * h;
	char* gpuImg;
	unsigned short* gpuDepthImg;
	char* gpuResImg;
	char* gpuBackgroundImg;
	cudaMalloc((void**)&gpuImg, nPix * 3 * sizeof(char));
	cudaMalloc((void**)&gpuDepthImg, nPix * sizeof(unsigned short));
	cudaMalloc((void**)&gpuResImg, nPix * 3 * sizeof(char));
	cudaMalloc((void**)&gpuBackgroundImg, nPix * 3 * sizeof(char));

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

	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0) // Application still alive?
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
		dim3 blockGrid((w * 3) / MAX_THREADS + 1, h, 1);

		gammaKernel<<<blockGrid, threadBlock >>> (gpuResImg, gpuImg, gpuDepthImg, gpuBackgroundImg, w * 3, h, scale);

		cudaMemcpy(cpuImg, gpuResImg, nPix * 3 * sizeof(char), cudaMemcpyDeviceToHost);

		cv::Mat my_mat(h, w, CV_8UC3, &cpuImg[0]);

		printf("%lu", sizeof(char));
		printf("%lu", sizeof(uint8_t));

		cv::namedWindow("Image window", cv::WINDOW_AUTOSIZE);
		cv::imshow("Image window", my_mat);


		virtual_output.send((const uint8_t*)cpuImg);


		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, cpuImg);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, readFboId);
		glBlitFramebuffer(0, 0, w, h, 0, 0, winWidth, winHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		glfwSwapBuffers(window);
		glfwPollEvents();

		/*
		GLenum errCode;
		if ((errCode = glGetError()) != GL_NO_ERROR)
		{
			printf("%d: %s \n", errCode, gluErrorString(errCode));
			break;
		}*/

		if ((char)cv::waitKey(25) == 27)
			break;
		
	}

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