#include <sched.h>

#include <math.h>
#include <time.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <map>
#include "LagrangeOptimizerL2.hpp"
#include "adios2/core/Engine.h"
#include "adios2/helper/adiosFunctions.h"
#include <omp.h>
#define GET4D(d0, d1, d2, d3, i, j, k, l) ((d1 * d2 * d3) * i + (d2 * d3) * j + d3 * k + l)
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

LagrangeOptimizerL2::LagrangeOptimizerL2(const char* species, const char* precision, torch::DeviceType device)
  : LagrangeOptimizer(species, precision)
{
    this->device = device;
    this->myOption = torch::TensorOptions().dtype(torch::kFloat64).device(device);
}

LagrangeOptimizerL2::LagrangeOptimizerL2(size_t planeOffset,
    size_t nodeOffset, size_t p, size_t n, size_t vx, size_t vy,
    const uint8_t species, const uint8_t precision, torch::DeviceType device)
  : LagrangeOptimizer(planeOffset, nodeOffset, p, n, vx, vy, species, precision)
{
    this->device = device;
    this->myOption = torch::TensorOptions().dtype(torch::kFloat64).device(device);
}

LagrangeOptimizerL2::~LagrangeOptimizerL2()
{
}

void LagrangeOptimizerL2::reconstructAndCompareErrors(int nodes, int iphi, at::Tensor &recondatain, at::Tensor &b_constant, at::Tensor &outputs)
{
#if 0
    std::vector <double> A (4*myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V2 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V3 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V4 (myNodeCount*myVxCount*myVyCount, 0);
    #pragma omp parallel for default (none) \
    shared(myNodeCount, myVxCount, myVyCount, myVolume, myVth, myVp, myMuQoi, myVth2, myParticleMass, V2, V3, V4) \
    private (i, j, l, m)
    for (k=0; k<4*myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int (k%myVyCount);
        l = int(k%(myVxCount*myVyCount));
        m = int(l/myVyCount);
        A[k]
        V2[k] = myVolume[k] * myVth[i] * myVp[j];
        V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
        V4[k] = myVolume[k] * pow(myVp[j],2) * myVth2[i] * myParticleMass;
    }

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
    // auto U = torch::zeros({nodes,myVxCount*myVyCount,4}, this->myOption);
    auto rdata = recon_data.reshape({nodes, myVxCount*myVyCount, 1});
    for (int index=0; index<nodes; ++index) {
        auto A_idx = A[index];
        // std::cout << "A_idx shape " << A_idx.sizes() << std::endl;
        auto A_idx_T = at::transpose(A_idx, 0, 1);
        // std::cout << "A_idx_T shape " << A_idx_T.sizes() << std::endl;
        auto Q = at::matmul(A_idx_T, at::inverse(at::matmul(A_idx, A_idx_T)));
        // std::cout << "Q shape " << Q.sizes() << std::endl;
        auto U_idx = std::get<0>(torch::linalg::svd(A_idx_T, false));
        // std::cout << "U_idx shape " << U_idx.sizes() << std::endl;
        auto U_idx_T = at::transpose(U_idx, 0, 1);
        // std::cout << "U_idx_T shape " << U_idx_T.sizes() << std::endl;
        auto I = at::eye(myVxCount*myVyCount, this->myOption);
        auto temp = I - at::matmul(U_idx, U_idx_T);
        // std::cout << "temp shape " << temp.sizes() << std::endl;
        auto R = at::matmul(temp, rdata[index]);
        // std::cout << "R shape " << R.sizes() << std::endl;
        // std::cout << "b shape " << b_constant[index].sizes() << std::endl;
        auto b = b_constant[index].reshape({4, 1});
        auto o_idx = R + at::matmul(Q, b);
        // std::cout << "o_idx shape " << o_idx.sizes() << std::endl;
        outputs[index] = o_idx.reshape({myVxCount, myVyCount});
    }
    outputs = outputs.reshape({1, nodes, myVxCount, myVyCount});
    // std::cout << "outputs shape 2" << outputs.sizes() << std::endl;
    // std::cout << "recondatain shape " << recondatain.sizes() << " outputs shape " << outputs.sizes() << std::endl;
    // compareQoIs(recondatain, outputs);
#endif
    return;
}

int LagrangeOptimizerL2::computeLagrangeParameters(
    const double* reconData, adios2::Dims blockCount)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double start, end, start1;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    int ii, i, j, k, l, m;
    GPTLstart("compute_lambdas");
#if 0
    std::vector<double> i_g(myLocalElements);
    int lindex, rindex;
    for (int i = 0; i < myPlaneCount; i++)
    {
        #pragma omp parallel for default (none) \
        shared(i, myPlaneCount, myNodeCount, myVxCount, myVyCount, reconData, i_g) \
        private (lindex, rindex)
        for (int k = 0; k < myNodeCount; k++)
        {
            for (int j = 0; j < myVxCount; j++)
            {
                for (int l = 0; l < myVyCount; l++)
                {
                    lindex = int(GET4D(myPlaneCount, myNodeCount, myVxCount, myVyCount, i, k, j, l));
                    rindex = int(GET4D(myPlaneCount, myVxCount, myNodeCount, myVyCount, i, j, k, l));
                    i_g[lindex] = reconData[rindex];
                    // GET4D(i_g, myPlaneCount, myNodeCount, myVxCount,
                        // myVyCount, i, k, j, l) =
                        // GET4D(reconData, myPlaneCount, myVxCount,
                            // myNodeCount, myVyCount, i, j, k, l);
                }
            }
        }
    }
    reconData = i_g.data();
#endif
    myLagranges = new double[4*myNodeCount];
    std::vector <double> V2 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V3 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V4 (myNodeCount*myVxCount*myVyCount, 0);
    #pragma omp parallel for default (none) \
    shared(myNodeCount, myVxCount, myVyCount, myVolume, myVth, myVp, myMuQoi, myVth2, myParticleMass, V2, V3, V4) \
    private (i, j, l, m)
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int (k%myVyCount);
        l = int(k%(myVxCount*myVyCount));
        m = int(l/myVyCount);
        V2[k] = myVolume[k] * myVth[i] * myVp[j];
        V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
        V4[k] = myVolume[k] * pow(myVp[j],2) * myVth2[i] * myParticleMass;
    }
    int breg_index = 0;
    int iphi, idx;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        const double* f0_f = &myDataIn[iphi*myNodeCount*myVxCount*myVyCount];
        std::vector<double> D(myNodeCount, 0);
        #pragma omp parallel for default (none) \
        shared(myNodeCount, myVxCount, myVyCount, myVolume, f0_f, D) \
        private (i)
        for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
            i = int(k/(myVxCount*myVyCount));
            D[i] += f0_f[k] * myVolume[k];
            // if (i > 300) {
                // printf ("Node %d F0F[%d]=%5.3g Volume=%5.3g\n", i, k, f0_f[k], myVolume[k]);
            // }
        }
        std::vector<double> U(myNodeCount, 0);
        std::vector<double> Tperp(myNodeCount, 0);
        #pragma omp parallel for default (none) \
        shared(myNodeCount, myVxCount, myVyCount, myVolume, myVth, myVp, f0_f, myMuQoi, myVth2, myParticleMass, mySmallElectronCharge, D, U, Tperp) \
        private (i, j, l, m)
        for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
            i = int(k/(myVxCount*myVyCount));
            j = int(k%myVyCount);
            l = int(k%(myVxCount*myVyCount));
            m = int(l/myVyCount);
            U[i] += (f0_f[k] * myVolume[k] * myVth[i] * myVp[j])/D[i];
            Tperp[i] += (f0_f[k] * myVolume[k] * 0.5 * myMuQoi[m] *
                myVth2[i] * myParticleMass)/D[i]/mySmallElectronCharge;
        }
        std::vector<double> Tpara(myNodeCount, 0);
        std::vector<double> Rpara(myNodeCount, 0);
        double en;
        #pragma omp parallel for default (none) \
        shared(myNodeCount, myVxCount, myVyCount, myVolume, myVth, myVp, f0_f, myVth2, myParticleMass, mySmallElectronCharge, D, U, Tpara) \
        private (i, j, en)
        for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
            i = int(k/(myVxCount*myVyCount));
            j = int(k%myVyCount);
            en = 0.5*pow((myVp[j]-U[i]/myVth[i]),2);
            Tpara[i] += 2*(f0_f[k] * myVolume[k] * en *
                myVth2[i] * myParticleMass)/D[i]/mySmallElectronCharge;
        }
        #pragma omp parallel for default (none) \
        shared(myNodeCount, myVxCount, myVyCount, myVolume, myVth, myVth2, myParticleMass, mySmallElectronCharge, U, Tpara, Rpara) \
        private (i)
        for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
            i = int(k/(myVxCount*myVyCount));
            Rpara[i] = mySmallElectronCharge*Tpara[i] +
                myVth2[i] * myParticleMass *
                pow((U[i]/myVth[i]), 2);
        }
        double aD;
        #pragma omp parallel for default (none) \
        shared(myLagranges, myNodeCount, myVxCount, myVyCount, mySmallElectronCharge, U, D, Tperp, Rpara) \
        private (i, aD)
        for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
            i = int(k/(myVxCount*myVyCount));
            aD = D[i]*mySmallElectronCharge;
            myLagranges[i*4] = D[i];
            myLagranges[i*4 + 1] = U[i] * D[i];
            myLagranges[i*4 + 2] = Tperp[i] * aD;
            myLagranges[i*4 + 3] = Rpara[i] * D[i];
        }
    }
    c10::cuda::CUDACachingAllocator::emptyCache();

    if (my_rank == 0) displayGPUMemory("#B", my_rank);

    GPTLstop("compute_lambdas");
    // auto outputs = torch::zeros({nodes, myVxCount, myVyCount}, this->myOption);
    // reconstructAndCompareErrors(nodes, iphi, recondatain, b_constant, outputs);
    return 0;
}

size_t LagrangeOptimizerL2::putLagrangeParameters(char* &bufferOut, size_t &bufferOutOffset, const char* precision)
{
    int i, count = 0;
    int numObjs = myPlaneCount*myNodeCount;
    // commented out the following lines temporarily for compression_method=3
//     if (!strcmp(precision, "single"))
//     {
//         for (i=0; i<numObjs*4; i++) {
//             *reinterpret_cast<float*>(
//                   bufferOut+bufferOutOffset+(count++)*sizeof(float)) =
//                       myLagranges[i];
//         }
//         return count * sizeof(float);
//     }
//     else {
//         for (i=0; i<numObjs*4; i++) {
//             *reinterpret_cast<double*>(
//                   bufferOut+bufferOutOffset+(count++)*sizeof(double)) =
//                       myLagranges[i];
//         }
//         return count * sizeof(double);
//     }
}

size_t LagrangeOptimizerL2::putResult(char* &bufferOut, size_t &bufferOutOffset, const char* precision)
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

void LagrangeOptimizerL2::setDataFromCharBuffer(double* &reconData,
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
    readF0Params(std::string(meshFile, 0, str_length));
    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    // std::cout << "Step 4: get reconstructed data" << std::endl;
    auto recondatain = torch::from_blob((void *)reconData, {myPlaneCount, myVxCount, myNodeCount, myVyCount}, torch::kFloat64).to(torch::kCUDA)
                  .permute({0, 2, 1, 3});
    // std::cout << "recondatain shape " << recondatain.sizes() << std::endl;
    // recondatain = at::clamp(recondatain, at::Scalar(100), at::Scalar(1e+50));
    // myLagrangesTorch =  torch::from_blob((void *)myLagranges, {myNodeCount, 4}, torch::kFloat64).to(torch::kCUDA);
    int iphi = 0;
    int nodes = myNodeCount*myPlaneCount;
    // std::cout << "Step 5: apply post processing and compute errors" << std::endl;
    // auto outputs = torch::zeros({nodes, myVxCount, myVyCount}, this->myOption);
    // reconstructAndCompareErrors(nodes, iphi, recondatain, myLagrangesTorch, outputs);
    // outputs = outputs.permute({0, 2, 1, 3});
    // auto datain = outputs.contiguous().cpu();
    // std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    // i = 0;
    // for (double c: datain_vec) {
        // reconData[i++] = c;
    // }
    return;
}
