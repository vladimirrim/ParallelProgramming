#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <CL/cl.h>
#include "cl.hpp"
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;

const size_t N_max = 1024;

int main() {
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);

    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        ifstream cl_file("convolution.cl");
        string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
        cl::Program::Sources source(1, make_pair(cl_string.c_str(), cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try {
            program.build(devices);
        } catch (cl::Error const &e) {
            string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            cout << endl << e.what() << " : " << e.err() << endl;
            cout << log_str;
            return 0;
        }

        // create a message to send to kernel
        size_t const block_size = 16;

        size_t N, M;
        cin >> N >> M;
        vector<double> a(N * N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double num;
                cin >> num;
                a[i * N + j] = num;
            }
        }

        vector<double> kernel(M * M);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                double num;
                cin >> num;
                kernel[i * M + j] = num;
            }
        }
        vector<double> result(N * N, 0);

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * N * N);
        cl::Buffer dev_kernel(context, CL_MEM_READ_ONLY, sizeof(double) * M * M);
        cl::Buffer dev_result(context, CL_MEM_WRITE_ONLY, sizeof(double) * N * N);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * N * N, &a[0]);
        queue.enqueueWriteBuffer(dev_kernel, CL_TRUE, 0, sizeof(double) * M * M, &kernel[0]);

        // load named kernel from opencl source
        cl::Kernel kernel_gmem(program, "convolution");
        kernel_gmem.setArg(0, dev_a);
        kernel_gmem.setArg(1, dev_kernel);
        kernel_gmem.setArg(2, dev_result);
        kernel_gmem.setArg(3, static_cast<int>(N));
        kernel_gmem.setArg(4, static_cast<int>(M));
        queue.enqueueNDRangeKernel(kernel_gmem, cl::NullRange, cl::NDRange(N_max * N_max), cl::NDRange(block_size));

        queue.enqueueReadBuffer(dev_result, CL_TRUE, 0, sizeof(double) * N * N, &result[0]);

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++)
                cout << fixed << setprecision(3) << result[i * N + j] << ' ';
            cout << endl;
        }
    } catch (cl::Error const &e) {
        cout << endl << e.what() << " : " << e.err() << endl;
    }

    return 0;
}
