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

void LagrangeTorchL2::reconstructAndCompareErrors(int nodes, int iphi, at::Tensor &recondatain, at::Tensor &b_constant, at::Tensor &outputs)
{
    GPTLstart("Acomp");
    auto V2_torch = myVolumeTorch * myVthTorch.reshape({nodes,1,1}) * myVpTorch.reshape({1, 1, myVyCount});
    auto V3_torch = myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass;
    auto V4_torch = myVolumeTorch * at::pow(myVpTorch, at::Scalar(2)).reshape({1, myVyCount}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass;

    // std::cout << "came here 4.1" << std::endl;
    auto recon_data = recondatain[iphi];
    // using namespace torch::indexing;
    auto A = torch::zeros({4,nodes,myVxCount*myVyCount}, this->myOption);
    A[0] = myVolumeTorch.reshape({nodes,myVxCount*myVyCount});
    A[1] = V2_torch.reshape({nodes,myVxCount*myVyCount});
    A[2] = V3_torch.reshape({nodes,myVxCount*myVyCount});
    A[3] = V4_torch.reshape({nodes,myVxCount*myVyCount});
    // std::cout << "A shape " << A.sizes() << std::endl;
    A = at::transpose(A, 0, 1);
    GPTLstop("Acomp");
    // auto U = torch::zeros({nodes,myVxCount*myVyCount,4}, this->myOption);
    auto rdata = recon_data.reshape({nodes, myVxCount*myVyCount, 1});
    for (int index=0; index<nodes; ++index) {

        GPTLstart("compute A-transpose");
        auto A_idx = A[index];
        // std::cout << "A_idx shape " << A_idx.sizes() << std::endl;
        auto A_idx_T = at::transpose(A_idx, 0, 1);
        // std::cout << "A_idx_T shape " << A_idx_T.sizes() << std::endl;
        GPTLstop("compute A-transpose");
        GPTLstart("Qcomp");
        auto Q = at::matmul(A_idx_T, at::inverse(at::matmul(A_idx, A_idx_T)));
        GPTLstop("Qcomp");
        // std::cout << "Q shape " << Q.sizes() << std::endl;
        GPTLstart("Ucomp");
        auto U_idx = std::get<0>(torch::linalg::svd(A_idx_T, false));
        // std::cout << "U_idx shape " << U_idx.sizes() << std::endl;
        GPTLstop("Ucomp");
        GPTLstart("compute UUT");
        auto U_idx_T = at::transpose(U_idx, 0, 1);
        // std::cout << "U_idx_T shape " << U_idx_T.sizes() << std::endl;
        auto I = at::eye(myVxCount*myVyCount, this->myOption);
        auto temp = I - at::matmul(U_idx, U_idx_T);
        // std::cout << "temp shape " << temp.sizes() << std::endl;
        GPTLstop("compute UUT");
        GPTLstart("compute reconstruction");
        auto R = at::matmul(temp, rdata[index]);
        // std::cout << "R shape " << R.sizes() << std::endl;
        // std::cout << "b shape " << b_constant[index].sizes() << std::endl;
        auto b = b_constant[index].reshape({4, 1});
        auto o_idx = R + at::matmul(Q, b);
        // std::cout << "o_idx shape " << o_idx.sizes() << std::endl;
        outputs[index] = o_idx.reshape({myVxCount, myVyCount});
        GPTLstop("compute reconstruction");
    }
    outputs = outputs.reshape({1, nodes, myVxCount, myVyCount});
    // std::cout << "outputs shape 2" << outputs.sizes() << std::endl;
    // std::cout << "recondatain shape " << recondatain.sizes() << " outputs shape " << outputs.sizes() << std::endl;
    compareQoIs(recondatain, outputs);
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
    if (my_rank == 0) displayGPUMemory("#A", my_rank);
    GPTLstart("compute_lambdas");
    int ii, i, j, k, l, m;
    auto recondatain = torch::from_blob((void *)reconData, {1, blockCount[1], blockCount[0]*blockCount[2], blockCount[3]}, torch::kFloat64).to(torch::kCUDA)
                  .permute({0, 2, 1, 3});
    std::cout << "recondatain sizes " << recondatain.sizes() << std::endl;
    auto origdatain = myDataInTorch.reshape({1, blockCount[0]*blockCount[2], blockCount[1], blockCount[3]});
    // std::cout << "myDataInTorch sizes " << myDataInTorch.sizes() << std::endl;
    // std::cout << "origdatain sizes " << origdatain.sizes() << std::endl;
    // recondatain = at::clamp(recondatain, at::Scalar(100));
    int nodes = myNodeCount*myPlaneCount;
    c10::cuda::CUDACachingAllocator::emptyCache();

    if (my_rank == 0) displayGPUMemory("#B", my_rank);

    int breg_index = 0;
    int iphi, idx;
    std::vector <at::Tensor> tensors;
    iphi = 0;
    std::vector<double> D(nodes, 0);
    auto f0_f_torch = origdatain[iphi];
    auto D_torch = (f0_f_torch * myVolumeTorch);
    auto D_torch_sum = D_torch.sum({1, 2});
    auto aD_torch = D_torch_sum*mySmallElectronCharge;
    c10::cuda::CUDACachingAllocator::emptyCache();

    if (my_rank == 0) displayGPUMemory("#C", my_rank);

    // std::vector<double> U(nodes, 0);
    std::vector<double> Tperp(nodes, 0);
    auto U_torch = (f0_f_torch * myVolumeTorch * myVthTorch.reshape({nodes,1,1}) * myVpTorch.reshape({1, 1, myVyCount}));
    auto U_torch_sum = U_torch.sum({1, 2})/D_torch_sum;
    auto Tperp_torch = ((f0_f_torch * myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass).sum({1,2}))/D_torch_sum/mySmallElectronCharge;
    c10::cuda::CUDACachingAllocator::emptyCache();

    if (my_rank == 0) displayGPUMemory("#D", my_rank);
    std::vector<double> Tpara(nodes, 0);
    std::vector<double> Rpara(nodes, 0);
    auto en_torch = 0.5*at::pow((myVpTorch.reshape({1, myVyCount})-U_torch_sum.reshape({nodes, 1})/myVthTorch.reshape({nodes, 1})),2);
    auto Tpara_torch = 2*((f0_f_torch * myVolumeTorch * en_torch.reshape({nodes, 1, myVyCount}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass).sum({1, 2}))/D_torch_sum/mySmallElectronCharge;
    auto Rpara_torch = mySmallElectronCharge*Tpara_torch + myVth2Torch * myParticleMass * at::pow((U_torch_sum/myVthTorch), 2);
    c10::cuda::CUDACachingAllocator::emptyCache();

    if (my_rank == 0) displayGPUMemory("#E", my_rank);
    auto b_constant = torch::zeros({4,nodes}, this->myOption);
    b_constant[0] = D_torch_sum;
    b_constant[1] = U_torch_sum*D_torch_sum;
    b_constant[2] = Tperp_torch*aD_torch;
    b_constant[3] = Rpara_torch*D_torch_sum;
    // std::cout << "b_constant shape " << b_constant.sizes() << std::endl;
    b_constant = at::transpose(b_constant, 0, 1);
    if (myPrecision == 1) {
        b_constant = (b_constant.to(torch::kFloat32)).to(torch::kFloat64);
    }
    myLagrangesTorch = b_constant.detach().clone();
    GPTLstop("compute_lambdas");
    auto outputs = torch::zeros({nodes, myVxCount, myVyCount}, this->myOption);
    reconstructAndCompareErrors(nodes, iphi, recondatain, b_constant, outputs);
    return 0;
}

size_t LagrangeTorchL2::putLagrangeParameters(char* &bufferOut, size_t &bufferOutOffset, const char* precision)
{
    auto datain = myLagrangesTorch.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    myLagranges = datain_vec.data();
    int i, count = 0;
    int numObjs = myPlaneCount*myNodeCount;
    if (!strcmp(precision, "single"))
    {
        for (i=0; i<numObjs*4; i++) {
            *reinterpret_cast<float*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(float)) =
                      myLagranges[i];
        }
        return count * sizeof(float);
    }
    else {
        for (i=0; i<numObjs*4; i++) {
            *reinterpret_cast<double*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(double)) =
                      myLagranges[i];
        }
        return count * sizeof(double);
    }
}

size_t LagrangeTorchL2::putResult(char* &bufferOut, size_t &bufferOutOffset, const char* precision)
{
    // TODO: after your algorithm is done, put the result into
    // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your       first
    // double number *reinterpret_cast<double*>(bufferOut+bufferOutOff      set+8)
    // for your second double number and so on
    int intbytes = putLagrangeParameters(bufferOut, bufferOutOffset, precision);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        FILE* fp = fopen("PqMeshInfo.bin", "wb");
        int str_length = myMeshFile.length();
        printf("Mesh file %d %s\n", str_length, myMeshFile.c_str());
        fwrite(&str_length, sizeof(int), 1, fp);
        fwrite(myMeshFile.c_str(), sizeof(char), str_length, fp);
        fclose(fp);
    }
    return intbytes;
}

void LagrangeTorchL2::setDataFromCharBuffer(double* &reconData,
    const char* bufferIn, size_t sizeOut)
{
    int i, count = 0;
    // std::cout << "Step 1: Extract b_constants" << std::endl;
    myLagranges = new double[4*myNodeCount];
    for (i=0; i<4*myNodeCount; ++i) {
        myLagranges[i] = *reinterpret_cast<const float*>(
              bufferIn+(count++)*sizeof(float));
    }
    // std::cout << "Step 2: Read mesh file name" << std::endl;
    FILE* fp = fopen("PqMeshInfo.bin", "rb");
    int str_length = 0;
    fread(&str_length, sizeof(int), 1, fp);
    char meshFile[str_length];
    fread(meshFile, sizeof(char), str_length, fp);
    fclose(fp);
    // std::cout << "Step 3: Compute mesh parameters" << std::endl;
    GPTLstart("read meshfile, compute params");
    readF0Params(std::string(meshFile, 0, str_length));
    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    GPTLstop("read meshfile, compute params");
    // std::cout << "Step 4: get reconstructed data" << std::endl;
    auto recondatain = torch::from_blob((void *)reconData, {myPlaneCount, myVxCount, myNodeCount, myVyCount}, torch::kFloat64).to(torch::kCUDA)
                  .permute({0, 2, 1, 3});
    // std::cout << "recondatain shape " << recondatain.sizes() << std::endl;
    // recondatain = at::clamp(recondatain, at::Scalar(100), at::Scalar(1e+50));
    myLagrangesTorch =  torch::from_blob((void *)myLagranges, {myNodeCount, 4}, torch::kFloat64).to(torch::kCUDA);
    int iphi = 0;
    int nodes = myNodeCount*myPlaneCount;
    // std::cout << "Step 5: apply post processing and compute errors" << std::endl;
    auto outputs = torch::zeros({nodes, myVxCount, myVyCount}, this->myOption);
    reconstructAndCompareErrors(nodes, iphi, recondatain, myLagrangesTorch, outputs);
    outputs = outputs.permute({0, 2, 1, 3});
    auto datain = outputs.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    i = 0;
    for (double c: datain_vec) {
        reconData[i++] = c;
    }
    return;
}
