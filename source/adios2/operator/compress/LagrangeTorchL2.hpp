#ifndef LAGRANGE_TORCH_L2_HPP
#define LAGRANGE_TORCH_L2_HPP

#include <torch/script.h>
#include <torch/torch.h>

#include "LagrangeTorch.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

static void displayGPUMemory(std::string msg, int rank)
{
	CUresult uRet;
	size_t free1;
	size_t total1;
	uRet = cuMemGetInfo(&free1, &total1);
	if (uRet == CUDA_SUCCESS)
		printf("%d: %s FreeMemory = %d Mb in TotalMeory = %d Mb\n", rank, msg.c_str(), free1 / 1024 / 1024, total1 / 1024 / 1024);
}

class LagrangeTorchL2 : public LagrangeTorch
{
    public:
        // Constructor
        LagrangeTorchL2(const char* species, const char* precision, torch::DeviceType device);
        LagrangeTorchL2(size_t planeOffset, size_t nodeOffset,
            size_t p, size_t n, size_t vx, size_t vy, const uint8_t species, const uint8_t precision, torch::DeviceType device);

        // Destructor
        ~LagrangeTorchL2();
        /*
        // Compute mesh parameters and QoIs
        void computeParamsAndQoIs(const std::string meshFile,
                adios2::Dims blockStart, adios2::Dims blockCount,
                const double* dataIn);
        // Compute Lagrange Parameters
        */
        int computeLagrangeParameters(const double* reconstructedData,
                adios2::Dims blockCount);
        /*
        size_t getTableSize();
        size_t putResult(char* &bufferOut, size_t &bufferOutOffset, const char* precision);
        char* setDataFromCharBuffer(double* &dataOut, const char* bufferIn, size_t bufferTotalSize);
        */

    private:
        // APIs
        /*
        void readF0Params(const std::string meshFile);
        void setVolume();
        void setVp();
        void setMuQoi();
        void setVth2();
        void compute_C_qois(int iphi, at::Tensor &density,
            at::Tensor &upara, at::Tensor &tperp,
            at::Tensor &tpara, at::Tensor &n0,
            at::Tensor &t0, at::Tensor &dataIn);
        // void compareQoIs(at::Tensor& reconData, at::Tensor& bregData);
        void compareErrorsPD(at::Tensor& dataIn, at::Tensor& reconData, at::Tensor& bregData, const char* etype, int rank);
        size_t putLagrangeParameters(char* &bufferOut, size_t &bufferOutOffset, const char* precision);
        void getUnconvergedIndexes(at::Tensor &diff, std::vector<long>& unconvergedNodeIndex, std::map<long, long> &unconvergedMap);
        int lambdaIterationsRound(int maxIter, double stepsize, at::Tensor &lambdas_torch, std::vector<long>& unconvergedNodeIndex, int nodes, at::Tensor &recon_torch, at::Tensor &orig_torch, at::Tensor &v_torch, at::Tensor &v2_torch, at::Tensor &v3_torch, at::Tensor &v4_torch, at::Tensor &d_torch, at::Tensor &u_torch, at::Tensor &t_torch, at::Tensor &r_torch, double DeB, double UeB, double TperpEB, double TparaEB, double PDeB);
        /*

    private:
        /*
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
        // static at::TensorOptions ourGPUOptions;
        // static at::TensorOptions ourCPUOptions;
        */
        at::TensorOptions myOption;
        torch::DeviceType device;
};

#endif
