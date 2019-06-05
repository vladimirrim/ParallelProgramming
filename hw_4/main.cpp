#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>
#include <string>

size_t const BLOCK_SIZE = 512;


int get_workers_count(int size, int block_size) {
    return ((int) ((size + block_size - 1) / block_size)) * block_size;
}

void run_copy(std::vector<double> &input, std::vector<double> &output,
              cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * input.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * output.size());

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * input.size(), &input[0]);

    cl::Kernel kernel(program, "partial_copy");
    kernel.setArg(0, dev_input);
    kernel.setArg(1, dev_output);
    kernel.setArg(2, input.size());
    kernel.setArg(3, output.size());
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(get_workers_count(input.size(), BLOCK_SIZE)),
                               cl::NDRange(BLOCK_SIZE));
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * output.size(), &output[0]);
    output[0] = 0;
}


void run_add(std::vector<double> &input, std::vector<double> &output, cl::Context &context,
             cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input_partial(context, CL_MEM_READ_ONLY, sizeof(double) * input.size());
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * output.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * output.size());

    queue.enqueueWriteBuffer(dev_input_partial, CL_TRUE, 0, sizeof(double) * input.size(), &input[0]);
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * output.size(), &output[0]);

    cl::Kernel kernel(program, "block_add");
    kernel.setArg(0, dev_input_partial);
    kernel.setArg(1, dev_input);
    kernel.setArg(2, dev_output);
    kernel.setArg(3, output.size());
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(get_workers_count(input.size(), BLOCK_SIZE)),
                               cl::NDRange(BLOCK_SIZE));
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * output.size(), &output[0]);
}


void scan(std::vector<double> &arr, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * arr.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * arr.size());

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * arr.size(), &arr[0]);

    cl::Kernel kernel(program, "scan_hillis_steele");
    kernel.setArg(0, dev_input);
    kernel.setArg(1, dev_output);
    kernel.setArg(2, cl::__local(sizeof(double) * BLOCK_SIZE));
    kernel.setArg(3, cl::__local(sizeof(double) * BLOCK_SIZE));
    kernel.setArg(4, arr.size());
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(get_workers_count(arr.size(), BLOCK_SIZE)),
                               cl::NDRange(BLOCK_SIZE));
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * arr.size(), &arr[0]);

    if (arr.size() > BLOCK_SIZE) {
        std::vector<double> prefixes((arr.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);
        run_copy(arr, prefixes, context, program, queue);
        scan(prefixes, context, program, queue);
        run_add(prefixes, arr, context, program, queue);
    }
}


int main() {
    std::freopen("input.txt", "r", stdin);
    std::freopen("output.txt", "w", stdout);

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices);

        // create a message to send to kernel
        size_t N;
        std::vector<double> input;
        std::cin >> N;
        for (size_t i = 0; i < N; ++i) {
            double x;
            std::cin >> x;
            input.push_back(x);
        }

        scan(input, context, program, queue);

        for (auto &elem: input)
            std::cout << std::setprecision(3) << elem << " ";

    }
    catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}