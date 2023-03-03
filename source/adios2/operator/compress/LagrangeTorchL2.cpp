#include <sched.h>

#include <math.h>
#include <time.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <map>
#include "LagrangeTorchL2.hpp"
#include "adios2/core/Engine.h"
#include "adios2/helper/adiosFunctions.h"
#include <omp.h>
#include <string_view>
#include <c10/cuda/CUDACachingAllocator.h>

#include <gptl.h>
#include <gptlmpi.h>
#include <stdio.h>

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


// Define static class members
// at::TensorOptions LagrangeTorchL2::ourGPUOptions = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
// at::TensorOptions LagrangeTorchL2::ourCPUOptions = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

LagrangeTorchL2::LagrangeTorchL2(const char* species, const char* precision, torch::DeviceType device)
  : LagrangeTorch(species, precision, device)
{
    this->device = device;
    this->myOption = torch::TensorOptions().dtype(torch::kFloat64).device(device);
}

LagrangeTorchL2::LagrangeTorchL2(size_t planeOffset,
    size_t nodeOffset, size_t p, size_t n, size_t vx, size_t vy,
    const uint8_t species, const uint8_t precision, torch::DeviceType device)
  : LagrangeTorch(planeOffset, nodeOffset, p, n, vx, vy, species, precision, device)
{
    this->device = device;
    this->myOption = torch::TensorOptions().dtype(torch::kFloat64).device(device);
}

LagrangeTorchL2::~LagrangeTorchL2()
{
}

int LagrangeTorchL2::computeLagrangeParameters(
    const double* reconData, adios2::Dims blockCount)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double start, end;
    // MPI_Barrier(MPI_COMM_WORLD);
    // start = MPI_Wtime();
    // std::cout << "came here 4.0" << std::endl;
    displayGPUMemory("#A", my_rank);
    GPTLstart("compute lambdas");
    int ii, i, j, k, l, m;
    auto recondatain = torch::from_blob((void *)reconData, {1, blockCount[1], blockCount[0]*blockCount[2], blockCount[3]}, torch::kFloat64).to(torch::kCUDA)
                  .permute({0, 2, 1, 3});
    // std::cout << "recondatain sizes " << recondatain.sizes() << std::endl;
    auto origdatain = myDataInTorch.reshape({1, blockCount[0]*blockCount[2], blockCount[1], blockCount[3]});
    // std::cout << "myDataInTorch sizes " << myDataInTorch.sizes() << std::endl;
    // std::cout << "origdatain sizes " << origdatain.sizes() << std::endl;
    // recondatain = at::clamp(recondatain, at::Scalar(100));
    int nodes = myNodeCount*myPlaneCount;
    auto V2_torch = myVolumeTorch * myVthTorch.reshape({nodes,1,1}) * myVpTorch.reshape({1, 1, myVyCount});
    auto V3_torch = myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass;
    auto V4_torch = myVolumeTorch * at::pow(myVpTorch, at::Scalar(2)).reshape({1, myVyCount}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass;

    displayGPUMemory("#B", my_rank);
    int breg_index = 0;
    int iphi, idx;
    std::vector <at::Tensor> tensors;
    iphi = 0;
    std::vector<double> D(nodes, 0);
    auto f0_f_torch = origdatain[iphi];
    auto D_torch = (f0_f_torch * myVolumeTorch);
    auto D_torch_sum = D_torch.sum({1, 2});
    auto aD_torch = D_torch_sum*mySmallElectronCharge;

    displayGPUMemory("#C", my_rank);
    // std::vector<double> U(nodes, 0);
    std::vector<double> Tperp(nodes, 0);
    auto U_torch = (f0_f_torch * myVolumeTorch * myVthTorch.reshape({nodes,1,1}) * myVpTorch.reshape({1, 1, myVyCount}));
    auto U_torch_sum = U_torch.sum({1, 2})/D_torch_sum;
    auto Tperp_torch = ((f0_f_torch * myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass).sum({1,2}))/D_torch_sum/mySmallElectronCharge;

    displayGPUMemory("#D", my_rank);
    std::vector<double> Tpara(nodes, 0);
    std::vector<double> Rpara(nodes, 0);
    auto en_torch = 0.5*at::pow((myVpTorch.reshape({1, myVyCount})-U_torch_sum.reshape({nodes, 1})/myVthTorch.reshape({nodes, 1})),2);
    auto Tpara_torch = 2*((f0_f_torch * myVolumeTorch * en_torch.reshape({nodes, 1, myVyCount}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass).sum({1, 2}))/D_torch_sum/mySmallElectronCharge;
    auto Rpara_torch = mySmallElectronCharge*Tpara_torch + myVth2Torch * myParticleMass * at::pow((U_torch_sum/myVthTorch), 2);

    displayGPUMemory("#E", my_rank);
    auto b_constant = torch::zeros({4,nodes}, this->myOption);
    b_constant[0] = D_torch_sum;
    b_constant[1] = U_torch_sum*D_torch_sum;
    b_constant[2] = Tperp_torch*aD_torch;
    b_constant[3] = Rpara_torch*D_torch_sum;
    // std::cout << "b_constant shape " << b_constant.sizes() << std::endl;
    b_constant = at::transpose(b_constant, 0, 1);
    if (myPrecision == 1) {
        myLagrangesTorch = b_constant.to(torch::kFloat32);
    }
    else{
        myLagrangesTorch = b_constant;
    }
    GPTLstop("compute lambdas");

    displayGPUMemory("#F", my_rank);
    // std::cout << "came here 4.1" << std::endl;
    auto recon_data = recondatain[iphi];
    auto orig_data = origdatain[iphi];
    using namespace torch::indexing;
    auto A = torch::zeros({4,nodes,myVxCount*myVyCount}, this->myOption);
    A[0] = myVolumeTorch.reshape({nodes,myVxCount*myVyCount});
    A[1] = V2_torch.reshape({nodes,myVxCount*myVyCount});
    A[2] = V3_torch.reshape({nodes,myVxCount*myVyCount});
    A[3] = V4_torch.reshape({nodes,myVxCount*myVyCount});

    displayGPUMemory("#G", my_rank);
    // std::cout << "A shape " << A.sizes() << std::endl;
    A = at::transpose(A, 0, 1);
    auto U = torch::zeros({nodes,myVxCount*myVyCount,4}, this->myOption);
    for (int index=0; index<nodes; ++index) {
        U[index] = std::get<0>(torch::linalg::svd(at::transpose(A[index], 0, 1), false));
    }
    // std::cout << "U shape" << U.sizes() << std::endl;
    // std::cout << "A shape after transpose " << A.sizes() << std::endl;

    displayGPUMemory("#H", my_rank);
    auto UT = at::transpose(U, 1, 2);
    // std::cout << "UT shape" << UT.sizes() << std::endl;
    auto rdata = recon_data.reshape({nodes, myVxCount*myVyCount, 1});
    auto odata = orig_data.reshape({nodes, myVxCount*myVyCount, 1});
    auto diff = odata - rdata;
    diff = diff.reshape({nodes, myVxCount*myVyCount, 1});
    // std::cout << "diff shape" << diff.sizes() << std::endl;
    auto outputs = rdata + at::bmm(U, at::bmm(UT, diff));
    // std::cout << "outputs shape 1" << outputs.sizes() << std::endl;
    
    displayGPUMemory("#I", my_rank);
    outputs = at::squeeze(outputs, 2);
    outputs = outputs.reshape({1, nodes, myVxCount, myVyCount});
    // std::cout << "outputs shape 2" << outputs.sizes() << std::endl;
    // std::cout << "recondatain shape " << recondatain.sizes() << " outputs shape " << outputs.sizes() << std::endl;
    displayGPUMemory("#J", my_rank);
    compareQoIs(recondatain, outputs);
    return 0;
}
