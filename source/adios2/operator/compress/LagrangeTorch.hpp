#ifndef LAGRANGE_TORCH_HPP
#define LAGRANGE_TORCH_HPP

#include <torch/script.h>
#include <torch/torch.h>

#include "LagrangeOptimizer.hpp"

class LagrangeTorch : public LagrangeOptimizer
{
    public:
        // Constructor
        LagrangeTorch(const char* species);
        LagrangeTorch(size_t planeOffset, size_t nodeOffset,
            size_t p, size_t n, size_t vx, size_t vy, uint8_t species);

        // Destructor
        ~LagrangeTorch();
        // Compute mesh parameters and QoIs
        void computeParamsAndQoIs(const std::string meshFile,
                adios2::Dims blockStart, adios2::Dims blockCount,
                const double* dataIn);
        // Compute Lagrange Parameters
        void computeLagrangeParameters(const double* reconstructedData,
                adios2::Dims blockCount);
        size_t getTableSize();
        size_t putResult(char* &bufferOut, size_t &bufferOutOffset, const char* precision);
        char* setDataFromCharBuffer(double* &dataOut, const char* bufferIn, size_t bufferTotalSize);

    private:
        // APIs
        void readF0Params(const std::string meshFile);
        void setVolume();
        void setVp();
        void setMuQoi();
        void setVth2();
        void compute_C_qois(int iphi, at::Tensor &density,
            at::Tensor &upara, at::Tensor &tperp,
            at::Tensor &tpara, at::Tensor &n0,
            at::Tensor &t0, at::Tensor &dataIn);
        void compareQoIs(at::Tensor& reconData, at::Tensor& bregData);
        void compareErrorsPD(at::Tensor& dataIn, at::Tensor& reconData, at::Tensor& bregData, const char* etype, int rank);
        size_t putLagrangeParameters(char* &bufferOut, size_t &bufferOutOffset, const char* precision);

    private:
        // Members
        at::Tensor myDataInTorch;
        at::Tensor myGridVolumeTorch;
        at::Tensor myF0TEvTorch;
        at::Tensor myVolumeTorch;
        at::Tensor myVpTorch;
        at::Tensor myMuQoiTorch;
        at::Tensor myVthTorch;
        at::Tensor myVth2Torch;
        at::Tensor myLagrangesTorch;
        static at::TensorOptions ourGPUOptions;
        static at::TensorOptions ourCPUOptions;
};

#endif
