#ifndef LAGRANGE_OPTIMIZER_L2_HPP
#define LAGRANGE_OPTIMIZER_L2_HPP

#include <torch/script.h>
#include <torch/torch.h>

#include "LagrangeOptimizer.hpp"

class LagrangeOptimizerL2 : public LagrangeOptimizer
{
    public:
        // Constructor
        LagrangeOptimizerL2(const char* species, const char* precision, torch::DeviceType device);
        LagrangeOptimizerL2(size_t planeOffset, size_t nodeOffset,
            size_t p, size_t n, size_t vx, size_t vy, const uint8_t species, const uint8_t precision, torch::DeviceType device);

        // Destructor
        ~LagrangeOptimizerL2();
        int computeLagrangeParameters(const double* reconstructedData,
                adios2::Dims blockCount);
        size_t putResult(char* &bufferOut, size_t &bufferOutOffset, const char* precision);
        void setDataFromCharBuffer(double* &dataOut, const char* bufferIn, size_t bufferTotalSize);

    private:
        // APIs
        void reconstructAndCompareErrors(int nodes, int iphi, at::Tensor &recondatain, at::Tensor &b_constant, at::Tensor &outputs);
        size_t putLagrangeParameters(char* &bufferOut, size_t &bufferOutOffset, const char* precision);

    private:
        at::TensorOptions myOption;
        torch::DeviceType device;
};

#endif
