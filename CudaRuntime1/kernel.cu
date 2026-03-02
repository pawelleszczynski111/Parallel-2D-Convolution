#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

//CPU image convolution
void convolutionCPU(const unsigned char* input,
	unsigned char* output,
	const float* filter,
	int img_rows,
	int img_cols,
	int filter_size)
{
	int half_filter_size = filter_size / 2;

	for (int y_id = 0; y_id < img_rows; y_id++) {
		for (int x_id = 0; x_id < img_cols; x_id++) {

			float sum = 0;

			if ((x_id >= half_filter_size) &&
				(x_id < img_cols - half_filter_size) &&
				(y_id >= half_filter_size) &&
				(y_id < img_rows - half_filter_size))
			{
				for (int filter_y = -half_filter_size; filter_y <= half_filter_size; filter_y++) {
					for (int filter_x = -half_filter_size; filter_x <= half_filter_size; filter_x++) {

						int x = x_id + filter_x;
						int y = y_id + filter_y;

						float img_value = input[y * img_cols + x];
						float filter_value =
							filter[(filter_y + half_filter_size) * filter_size +
							(filter_x + half_filter_size)];

						sum += img_value * filter_value;
					}
				}

				if (sum > 255) sum = 255;
				else if (sum < 0) sum = 0;

				output[y_id * img_cols + x_id] = static_cast<unsigned char>(sum);
			}
		}
	}
}


//GPU parallel image convolution
__global__ void convolution(const unsigned char* input, unsigned char* output, const float* filter, int img_rows, int img_cols, int filter_size) {

	int x_id = blockIdx.x * blockDim.x + threadIdx.x;
	int y_id = blockIdx.y * blockDim.y + threadIdx.y;

	int half_filter_size = filter_size / 2;
	float sum = 0;
	//Rejeceting pixels which would make kernel move beyond image borders
	if ((x_id >= half_filter_size) && (x_id < img_cols - half_filter_size) && (y_id >= half_filter_size) && (y_id < img_rows - half_filter_size))
	{
		//iterarion on image pixels within kernel borders
		for (int filter_y = -half_filter_size; filter_y <= half_filter_size; filter_y++) {
			for (int filter_x = -half_filter_size; filter_x <= half_filter_size; filter_x++) {

				int x = x_id + filter_x;
				int y = y_id + filter_y;
				float img_value = input[y * img_cols + x];


				float filter_value = filter[(filter_y + half_filter_size) * filter_size + (filter_x + half_filter_size)];
				sum = sum + (img_value * filter_value);
			}
		}
		if (sum > 255) sum = 255;
		else if (sum < 0) sum = 0;

		output[y_id * img_cols + x_id] = sum;
	}
}



int main() {
	//Loading image in greyscale
	cv::Mat img = cv::imread("D:/Projekty/zdj.bmp", cv::IMREAD_GRAYSCALE);


	int rows = img.rows;
	int cols = img.cols;

	cv::Mat result_cpu(rows, cols, CV_8UC1);


	//kernel
	const int filter_size = 3;

	float filter2[filter_size * filter_size] = {
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0
	};

	float filter[filter_size * filter_size] = {
	0, -1, 0,
	-1, 4, -1,
	0, -1, 0
	};


	auto start_cpu = std::chrono::high_resolution_clock::now();

	convolutionCPU(img.data, result_cpu.data, filter, rows, cols, filter_size);

	auto end_cpu = std::chrono::high_resolution_clock::now();

	double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();




	//memory allocation

	unsigned char* d_input;
	unsigned char* d_output;
	float* d_filter;

	int img_bytes = rows * cols * sizeof(unsigned char);
	int filter_bytes = filter_size * filter_size * sizeof(float);

	cudaMalloc(&d_input, img_bytes);
	cudaMalloc(&d_output, img_bytes);
	cudaMalloc(&d_filter, filter_bytes);

	//cpu to gpu copying
	

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	cudaEvent_t start_gpu, stop_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);

	cudaEventRecord(start_gpu, 0);

	cudaMemcpy(d_input, img.data, img_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, filter, filter_bytes, cudaMemcpyHostToDevice);

	convolution << <grid, block >> > (d_input, d_output, d_filter, rows, cols, filter_size);

	cudaEventRecord(stop_gpu, 0);
	cudaEventSynchronize(stop_gpu);

	float gpu_time = 0;
	cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

	//results copying
	cv::Mat result(rows, cols, CV_8UC1);
	cudaMemcpy(result.data, d_output, img_bytes, cudaMemcpyDeviceToHost);


	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_filter);


	double speedup = cpu_time / gpu_time;
	std::cout << "\n=== BENCHMARK RESULTS ===\n";
	std::cout << "Image size: " << rows << " x " << cols << "\n";
	std::cout << "CPU time: " << cpu_time << " ms\n";
	std::cout << "GPU time: " << gpu_time << " ms\n";
	std::cout << "Speedup: " << speedup << "x\n";

	cv::imshow("Convolved image", result);
	cv::imwrite("new_img.jpg", result);
	cv::waitKey(0);

	return 0;

}