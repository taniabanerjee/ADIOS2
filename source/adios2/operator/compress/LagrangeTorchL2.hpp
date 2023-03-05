#ifndef LAGRANGE_TORCH_L2_HPP
#define LAGRANGE_TORCH_L2_HPP

#include <torch/script.h>
#include <torch/torch.h>

#include "LagrangeTorch.hpp"

class LagrangeTorchL2 : public LagrangeTorch
{
    public:
        // Constructor
        LagrangeTorchL2(const char* species, const char* precision);
        LagrangeTorchL2(size_t planeOffset, size_t nodeOffset,
            size_t p, size_t n, size_t vx, size_t vy, const uint8_t species, const uint8_t precision);

        // Destructor
        ~LagrangeTorchL2();
        // Compute Lagrange Parameters
        int computeLagrangeParameters(const double* reconstructedData,
                adios2::Dims blockCount);

    private:
        void reconstructAndCompareErrors(int nodes, int iphi, at::Tensor &recondatain, at::Tensor &b_constant);

    private:
        // Members
        static at::TensorOptions ourGPUOptions;
        static at::TensorOptions ourCPUOptions;
};

#endif
