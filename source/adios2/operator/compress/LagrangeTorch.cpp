#include <sched.h>

#include <math.h>
#include <time.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <map>
#include "LagrangeTorch.hpp"
#include "adios2/core/Engine.h"
#include "adios2/helper/adiosFunctions.h"
#include <omp.h>
#include <string_view>
#include <c10/cuda/CUDACachingAllocator.h>

#include <gptl.h>
#include <gptlmpi.h>

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
// at::TensorOptions LagrangeTorch::ourGPUOptions = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
// at::TensorOptions LagrangeTorch::ourCPUOptions = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

LagrangeTorch::LagrangeTorch(const char* species, const char* precision, torch::DeviceType device)
  : LagrangeOptimizer(species, precision)
{
    this->device = device;
    this->myOption = torch::TensorOptions().dtype(torch::kFloat64).device(device);
}

LagrangeTorch::LagrangeTorch(size_t planeOffset,
    size_t nodeOffset, size_t p, size_t n, size_t vx, size_t vy,
    const uint8_t species, const uint8_t precision, torch::DeviceType device)
  : LagrangeOptimizer(planeOffset, nodeOffset, p, n, vx, vy, species, precision)
{
    this->device = device;
    this->myOption = torch::TensorOptions().dtype(torch::kFloat64).device(device);
}

LagrangeTorch::~LagrangeTorch()
{
}

void LagrangeTorch::computeParamsAndQoIs(const std::string meshFile,
     adios2::Dims blockStart, adios2::Dims blockCount,
     const double* dataIn)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    myMeshFile = meshFile;
    double start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    torch::NoGradGuard no_grad;
    start = MPI_Wtime();
    int planeIndex = 0;
    int nodeIndex = 2;
    int velXIndex = 1;
    int velYIndex = 3;
    int iphi = 0;
    myPlaneOffset = blockStart[planeIndex];
    myNodeOffset = blockStart[nodeIndex];
    myNodeCount = blockCount[nodeIndex];
    myPlaneCount = blockCount[planeIndex];
    myVxCount = blockCount[velXIndex];
    myVyCount = blockCount[velYIndex];
    myLocalElements = myNodeCount * myPlaneCount * myVxCount * myVyCount;
    auto datain = torch::from_blob((void *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]}, torch::kFloat64).to(this->device)
                  .permute({0, 2, 1, 3});
    myDataInTorch = datain;
    GPTLstart("read_mesh_file");
    readF0Params(meshFile);
    GPTLstop("read_mesh_file");
    // MPI_Barrier(MPI_COMM_WORLD);
    // end = MPI_Wtime();
    // if (my_rank == 0) {
        // printf ("%d Time Taken for File Reading: %f\n", mySpecies, (end-start));
    // }
    start = MPI_Wtime();
    GPTLstart("compute_params");
    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    // std::cout << "myVolumeTorch sizes " << myVolumeTorch.sizes() << std::endl;
    // std::cout << "myVpTorch sizes " << myVpTorch.sizes() << std::endl;
    // std::cout << "myMuQoiTorch sizes " << myMuQoiTorch.sizes() << std::endl;
    // std::cout << "myVthTorch sizes " << myVthTorch.sizes() << std::endl;
    // std::cout << "myVth2Torch sizes " << myVth2Torch.sizes() << std::endl;
    myMaxValue = myDataInTorch.max().item().to<double>();;
    // MPI_Barrier(MPI_COMM_WORLD);
    // end = MPI_Wtime();
    // if (my_rank == 0) {
    //     printf ("%d Time Taken for QoI param Computation: %f\n", mySpecies, (end-start));
    // }
    GPTLstop("compute_params");
}

void LagrangeTorch::getUnconvergedIndexes(at::Tensor &diff, std::vector<long>& unconvergedNodeIndex, std::map<long, long> &unconvergedMap)
{
    diff = diff.reshape({diff.numel()}).contiguous().cpu();
    // auto diff_reduced = L2_diff.index({diff});
    std::vector<long> diffvec(diff.data_ptr<long>(),
            diff.data_ptr<long>() + diff.numel());

    for (long diffindex: diffvec) {
        auto search = unconvergedMap.find(diffindex);
        if (search == unconvergedMap.end()) {
            unconvergedNodeIndex.push_back(diffindex);
            unconvergedMap[diffindex] = 1;
        }
    }
}

int LagrangeTorch::lambdaIterationsRound(int maxIter, double stepsize, at::Tensor &lambdas_torch, std::vector<long>& unconvergedNodeIndex, int nodes, at::Tensor &recon_torch, at::Tensor &orig_torch, at::Tensor &v_torch, at::Tensor &v2_torch, at::Tensor &v3_torch, at::Tensor &v4_torch, at::Tensor &d_torch, at::Tensor &u_torch, at::Tensor &t_torch, at::Tensor &r_torch, double DeB, double UeB, double TperpEB, double TparaEB, double PDeB)
{
    torch::NoGradGuard no_grad;
    auto aD_torch = d_torch*mySmallElectronCharge;
    auto gradients_torch = torch::zeros({nodes,4}, myOption);
    auto hessians_torch = torch::zeros({nodes,4,4}, myOption);
    auto L2_den = torch::zeros({nodes, maxIter+1}, myOption);
    auto L2_upara = torch::zeros({nodes, maxIter+1}, myOption);
    auto L2_tperp = torch::zeros({nodes, maxIter+1}, myOption);
    auto L2_rpara = torch::zeros({nodes, maxIter+1}, myOption);
    auto L2_pd = torch::zeros({nodes, maxIter+1}, myOption);

    auto K = torch::zeros({nodes,myVxCount,myVyCount}, myOption);
    int count = 0;
    int converged = 0;

    using namespace torch::indexing;
    while (count < maxIter)
    {
        gradients_torch.index_put_({Slice(None), 0}, (-((recon_torch*v_torch*at::exp(-K)).sum({1,2})) + d_torch));
        gradients_torch.index_put_({Slice(None), 1}, (-((recon_torch*v2_torch*at::exp(-K)).sum({1,2})) + u_torch*d_torch));
        gradients_torch.index_put_({Slice(None), 2}, (-((recon_torch*v3_torch*at::exp(-K)).sum({1,2})) + t_torch*aD_torch));
        gradients_torch.index_put_({Slice(None), 3}, (-((recon_torch*v4_torch*at::exp(-K)).sum({1,2})) + r_torch*d_torch));
        hessians_torch.index_put_({Slice(None), 0, 0}, (recon_torch*at::pow(v_torch,2)*at::exp(-K)).sum({1,2}));
        hessians_torch.index_put_({Slice(None), 0, 1}, (recon_torch*v_torch*v2_torch*at::exp(-K)).sum({1,2}));
        hessians_torch.index_put_({Slice(None), 0, 2}, (recon_torch*v_torch*v3_torch*at::exp(-K)).sum({1,2}));
        hessians_torch.index_put_({Slice(None), 0, 3}, (recon_torch*v_torch*v4_torch*at::exp(-K)).sum({1,2}));
        hessians_torch.index_put_({Slice(None), 1, 0}, hessians_torch.index({Slice(None), 0, 1}));
        hessians_torch.index_put_({Slice(None), 1, 1}, (recon_torch*at::pow(v2_torch, 2)*at::exp(-K)).sum({1,2}));
        hessians_torch.index_put_({Slice(None), 1, 2}, (recon_torch*v2_torch*v3_torch*at::exp(-K)).sum({1,2}));
        hessians_torch.index_put_({Slice(None), 1, 3}, (recon_torch*v2_torch*v4_torch*at::exp(-K)).sum({1,2}));
        hessians_torch.index_put_({Slice(None), 2, 0}, hessians_torch.index({Slice(None), 0, 2}));
        hessians_torch.index_put_({Slice(None), 2, 1}, hessians_torch.index({Slice(None), 1, 2}));
        hessians_torch.index_put_({Slice(None), 2, 2}, (recon_torch*at::pow(v3_torch, 2)*at::exp(-K)).sum({1,2}));
        hessians_torch.index_put_({Slice(None), 2, 3}, (recon_torch*v3_torch*v4_torch*at::exp(-K)).sum({1,2}));
        hessians_torch.index_put_({Slice(None), 3, 0}, hessians_torch.index({Slice(None), 0, 3}));
        hessians_torch.index_put_({Slice(None), 3, 1}, hessians_torch.index({Slice(None), 1, 3}));
        hessians_torch.index_put_({Slice(None), 3, 2}, hessians_torch.index({Slice(None), 2, 3}));
        hessians_torch.index_put_({Slice(None), 3, 3}, (recon_torch*at::pow(v4_torch, 2)*at::exp(-K)).sum({1,2}));
        try {
            lambdas_torch = lambdas_torch - at::squeeze(stepsize*at::bmm(at::inverse(hessians_torch), gradients_torch.reshape({nodes, 4, 1})));
        }
        catch (const c10::Error& e) {
            std::cout << "Need to compute pseudoinverse for hessians_torch" << std::endl;
            // std::cout << hessians_torch << std::endl;
            lambdas_torch = lambdas_torch - at::squeeze(at::bmm(torch::linalg::pinv(hessians_torch), gradients_torch.reshape({nodes, 4, 1})));
        }
        auto l1 = lambdas_torch.index({Slice(None), 0}).reshape({nodes, 1, 1}) * v_torch;
        auto l2 = lambdas_torch.index({Slice(None), 1}).reshape({nodes, 1, 1}) * v2_torch;
        auto l3 = lambdas_torch.index({Slice(None), 2}).reshape({nodes, 1, 1}) * v3_torch;
        auto l4 = lambdas_torch.index({Slice(None), 3}).reshape({nodes, 1, 1}) * v4_torch;
        K = l1 + l2 + l3 + l4;
        count = count + 1;
        if (count % 1 == 0) {
            auto breg_result = recon_torch*at::exp(-K);
            auto update_D = (breg_result * v_torch).sum({1,2});
            auto update_U = (breg_result * v2_torch).sum({1,2})/d_torch;
            auto update_Tperp = (breg_result * v3_torch).sum({1,2})/aD_torch;
            auto update_Rpara = (breg_result * v4_torch).sum({1,2})/d_torch;
            auto rmse_pd = at::pow(breg_result - orig_torch, 2).sum({1,2});
            L2_den.index_put_({Slice(None), count}, at::pow(update_D-d_torch, 2));
            L2_upara.index_put_({Slice(None), count}, at::pow(update_U-u_torch, 2));
            L2_tperp.index_put_({Slice(None), count}, at::pow(update_Tperp-t_torch, 2));
            L2_rpara.index_put_({Slice(None), count}, at::pow(update_Rpara-r_torch, 2));
            L2_pd.index_put_({Slice(None), count}, at::sqrt(rmse_pd));
            if (count >= 2) {
                auto L2_den_diff = (L2_den.index({Slice(None), count}) - L2_den.index({Slice(None), count - 1})).abs();
                auto L2_upara_diff = (L2_upara.index({Slice(None), count}) - L2_upara.index({Slice(None), count - 1})).abs();
                auto L2_tperp_diff = (L2_tperp.index({Slice(None), count}) - L2_tperp.index({Slice(None), count - 1})).abs();
                auto L2_rpara_diff = (L2_rpara.index({Slice(None), count}) - L2_rpara.index({Slice(None), count - 1})).abs();
                auto L2_pd_diff = (L2_pd.index({Slice(None), count}) - L2_pd.index({Slice(None), count - 1})).abs();
                auto den_diff = torch::argwhere(L2_den_diff > DeB);
                auto upara_diff = torch::argwhere(L2_upara_diff > UeB);
                auto tperp_diff = torch::argwhere(L2_tperp_diff > TperpEB);
                auto rpara_diff = torch::argwhere(L2_rpara_diff > TparaEB);
                auto pd_diff = torch::argwhere(L2_pd_diff > PDeB);
                if (den_diff.numel() == 0 && upara_diff.numel()==0 && tperp_diff.numel() == 0 && rpara_diff.numel() == 0 && pd_diff.numel() == 0)
                {
                    // std::cout << "All nodes converged at iteration " << count << std::endl;
                    converged = 1;
                }
                else if (count == maxIter - 1) {
                    std::map<long, long> unconvergedMap;
                    if (den_diff.numel() > 0) {
                        getUnconvergedIndexes(den_diff, unconvergedNodeIndex, unconvergedMap);
                    }
                    if (upara_diff.numel() > 0) {
                        getUnconvergedIndexes(upara_diff, unconvergedNodeIndex, unconvergedMap);
                    }
                    if (tperp_diff.numel() > 0) {
                        getUnconvergedIndexes(tperp_diff, unconvergedNodeIndex, unconvergedMap);
                    }
                    if (rpara_diff.numel() > 0) {
                        getUnconvergedIndexes(rpara_diff, unconvergedNodeIndex, unconvergedMap);
                    }
                    if (pd_diff.numel() > 0) {
                        getUnconvergedIndexes(pd_diff, unconvergedNodeIndex, unconvergedMap);
                    }
                }
            }
        }
        if (converged == 1) {
            break;
        }
    }
    return converged;
}

int LagrangeTorch::computeLagrangeParameters(
    const double* reconData, adios2::Dims blockCount)
{
    int unconverged_size = 0;
    int unconverged_images = 0;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    torch::NoGradGuard no_grad;
    double start, end;
    // MPI_Barrier(MPI_COMM_WORLD);
    // start = MPI_Wtime();
    // std::cout << "came here 4.0" << std::endl;
    // displayGPUMemory("#A", my_rank);
    GPTLstart("compute_lambdas");
    int ii, i, j, k, l, m;
    auto recondatain = torch::from_blob((void *)reconData, {1, blockCount[1], blockCount[0]*blockCount[2], blockCount[3]}, torch::kFloat64).to(this->device)
                  .permute({0, 2, 1, 3});
    // std::cout << "recondatain sizes " << recondatain.sizes() << std::endl;
    auto origdatain = myDataInTorch.reshape({1, blockCount[0]*blockCount[2], blockCount[1], blockCount[3]});
    // std::cout << "myDataInTorch sizes " << myDataInTorch.sizes() << std::endl;
    // std::cout << "origdatain sizes " << origdatain.sizes() << std::endl;
    double maxDataValue = myDataInTorch.max().item().to<double>();
    double minDataValue = myDataInTorch.min().item().to<double>();
    // recondatain = at::clamp(recondatain, at::Scalar(100), at::Scalar(maxDataValue));
    recondatain = at::clamp(recondatain, at::Scalar(100));
    int nodes = myNodeCount*myPlaneCount;
    auto V2_torch = myVolumeTorch * myVthTorch.reshape({nodes,1,1}) * myVpTorch.reshape({1, 1, myVyCount});
    auto V3_torch = myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass;
    auto V4_torch = myVolumeTorch * at::pow(myVpTorch, at::Scalar(2)).reshape({1, myVyCount}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass;

    // displayGPUMemory("#A1", my_rank);
    int breg_index = 0;
    int iphi, idx;
    std::vector <at::Tensor> tensors;
    iphi = 0;
    std::vector<double> D(nodes, 0);
    auto f0_f_torch = origdatain[iphi];
    auto D_torch = (f0_f_torch * myVolumeTorch);
    auto D_torch_sum = D_torch.sum({1, 2});

    // displayGPUMemory("#A2", my_rank);
    std::vector<double> U(nodes, 0);
    std::vector<double> Tperp(nodes, 0);
    auto U_torch = (f0_f_torch * myVolumeTorch * myVthTorch.reshape({nodes,1,1}) * myVpTorch.reshape({1, 1, myVyCount}));
    auto U_torch_sum = U_torch.sum({1, 2})/D_torch_sum;
    auto Tperp_torch = ((f0_f_torch * myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass).sum({1,2}))/D_torch_sum/mySmallElectronCharge;

    // displayGPUMemory("#A3", my_rank);
    std::vector<double> Tpara(nodes, 0);
    std::vector<double> Rpara(nodes, 0);
    auto en_torch = 0.5*at::pow((myVpTorch.reshape({1, myVyCount})-U_torch_sum.reshape({nodes, 1})/myVthTorch.reshape({nodes, 1})),2);
    auto Tpara_torch = 2*((f0_f_torch * myVolumeTorch * en_torch.reshape({nodes, 1, myVyCount}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass).sum({1, 2}))/D_torch_sum/mySmallElectronCharge;
    auto Rpara_torch = mySmallElectronCharge*Tpara_torch + myVth2Torch * myParticleMass * at::pow((U_torch_sum/myVthTorch), 2);

    // displayGPUMemory("#A4", my_rank);
    // std::cout << "came here 4.1" << std::endl;
    int count_unLag = 0;
    std::vector <int> node_unconv;
    double maxD = D_torch_sum.max().item().to<double>();
    double maxU = U_torch_sum.max().item().to<double>();
    double maxTperp = Tperp_torch.max().item().to<double>();
    double maxTpara = Rpara_torch.max().item().to<double>();

    double DeB = pow(maxD*1e-05, 2);
    double UeB = pow(maxU*1e-05, 2);
    double TperpEB = pow(maxTperp*1e-05, 2);
    double TparaEB = pow(maxTpara*1e-05, 2);
    double PDeB = pow(myMaxValue*1e-05, 2);

    // displayGPUMemory("#A5", my_rank);
    std::vector<long> unconvergedNodeIndex;
    std::map<long, long> unconvergedMap;
    auto lambdas_torch = torch::zeros({nodes,4}, myOption);
    auto recon_data = recondatain[iphi];
    auto orig_data = origdatain[iphi];
    int converged = lambdaIterationsRound(50, 1.0, lambdas_torch, unconvergedNodeIndex, nodes, recon_data, orig_data, myVolumeTorch, V2_torch, V3_torch, V4_torch, D_torch_sum, U_torch_sum, Tperp_torch, Rpara_torch, DeB, UeB, TperpEB, TparaEB, PDeB);
    // std::cout << "came here 4.2" << std::endl;
    int round = 1;
    int maxRound = 3;
    int maxIterArray[maxRound] {800, 1600};
    double stepSizeArray[maxRound] {0.1, 0.01};
    // displayGPUMemory("#B", my_rank);
    while (converged == 0 && round <= maxRound)
    {
        std::cout << "All nodes did not converge on rank " << my_rank << " on round " << round << std::endl;
        size_t vecIndex = unconvergedNodeIndex.size();
        /*
        for (vecIndex=0; vecIndex<unconvergedNodeIndex.size(); ++vecIndex) {
            std::cout << "Unconverged node (" << round << "): " << iphi << ", " << unconvergedNodeIndex[vecIndex] << std::endl;
        }
        */
        auto unodes = torch::from_blob((void*)unconvergedNodeIndex.data(), vecIndex, torch::kInt64).to(this->device);
        auto ltorch = torch::zeros({unconvergedNodeIndex.size(),4}, myOption);

        using namespace torch::indexing;
        auto recon_plane = recondatain[iphi];
        auto recon_torch = recon_plane.index({unodes, Slice(None), Slice(None)});
        auto orig_plane = origdatain[iphi];
        auto orig_torch = orig_plane.index({unodes, Slice(None), Slice(None)});
        auto v_torch = myVolumeTorch.index({unodes, Slice(None), Slice(None)});
        auto v2_torch = V2_torch.index({unodes, Slice(None), Slice(None)});
        auto v3_torch = V3_torch.index({unodes, Slice(None), Slice(None)});
        auto v4_torch = V4_torch.index({unodes, Slice(None), Slice(None)});
        auto d_torch = D_torch_sum.index({unodes});
        auto u_torch = U_torch_sum.index({unodes});
        auto t_torch = Tperp_torch.index({unodes});
        auto r_torch = Rpara_torch.index({unodes});
        int newnodes = unconvergedNodeIndex.size();
        unconvergedNodeIndex.clear();
        converged = lambdaIterationsRound(maxIterArray[round-1], stepSizeArray[round-1], ltorch, unconvergedNodeIndex, newnodes, recon_torch, orig_torch, v_torch, v2_torch, v3_torch, v4_torch, d_torch, u_torch, t_torch, r_torch, DeB, UeB, TperpEB, TparaEB, PDeB);
        for (vecIndex=0; vecIndex<unconvergedNodeIndex.size(); ++vecIndex) {
            // std::cout << "Unconverged node (" << round << "): " << iphi << ", " << unconvergedNodeIndex[vecIndex] << std::endl;
            unconvergedNodeIndex[vecIndex] = unodes[unconvergedNodeIndex[vecIndex]].item<long>();
        }
        lambdas_torch.index_put_({unodes, Slice(None)}, ltorch);
        round += 1;
    }
    
    // displayGPUMemory("#C", my_rank);
    // std::cout << "came here 4.3" << std::endl;
    using namespace torch::indexing;
    auto K = torch::zeros({nodes,myVxCount,myVyCount}, myOption);
    if (myPrecision == 0) {
        auto l1 = lambdas_torch.index({Slice(None), 0}).reshape({nodes, 1, 1}) * myVolumeTorch;
        auto l2 = lambdas_torch.index({Slice(None), 1}).reshape({nodes, 1, 1}) * V2_torch;
        auto l3 = lambdas_torch.index({Slice(None), 2}).reshape({nodes, 1, 1}) * V3_torch;
        auto l4 = lambdas_torch.index({Slice(None), 3}).reshape({nodes, 1, 1}) * V4_torch;
        K = l1 + l2 + l3 + l4;
    }
    else if (myPrecision == 1) {
        auto lambdas_torch_32 = lambdas_torch.to(torch::kFloat32);
        auto l1 = lambdas_torch_32.index({Slice(None), 0}).reshape({nodes, 1, 1}) * myVolumeTorch;
        auto l2 = lambdas_torch_32.index({Slice(None), 1}).reshape({nodes, 1, 1}) * V2_torch;
        auto l3 = lambdas_torch_32.index({Slice(None), 2}).reshape({nodes, 1, 1}) * V3_torch;
        auto l4 = lambdas_torch_32.index({Slice(None), 3}).reshape({nodes, 1, 1}) * V4_torch;
        K = l1 + l2 + l3 + l4;
    }
    else if (myPrecision == 2) {
        auto lambdas_torch_32 = lambdas_torch.to(torch::kFloat32);
        auto lambdas_torch_16 = lambdas_torch_32.to(torch::kFloat16);
        // if (my_rank == 0) {
            // std::cout << "Lambdas 32" << lambdas_torch_32 << std::endl;
            // std::cout << "Lambdas 16" << lambdas_torch_16 << std::endl;
            // std::cout << "Lambdas 64" << lambdas_torch << std::endl;
        // }
        auto l1 = lambdas_torch_16.index({Slice(None), 0}).reshape({nodes, 1, 1}) * myVolumeTorch;
        auto l2 = lambdas_torch_16.index({Slice(None), 1}).reshape({nodes, 1, 1}) * V2_torch;
        auto l3 = lambdas_torch_16.index({Slice(None), 2}).reshape({nodes, 1, 1}) * V3_torch;
        auto l4 = lambdas_torch_16.index({Slice(None), 3}).reshape({nodes, 1, 1}) * V4_torch;
        K = l1 + l2 + l3 + l4;
    }
    // displayGPUMemory("#D", my_rank);
    auto outputs = recondatain[iphi]*at::exp(-K);
    // Check for any nans
    auto pda_isnan = at::isnan(outputs);
    bool isPDnan = pda_isnan.any().item<bool>();
    // std::cout << "came here 4.4" << std::endl;
    if (isPDnan) {
        auto loc = torch::argwhere(pda_isnan == true);
        // auto elems = at::_unique(loc.slice(1, 0, 1));
        auto elems = loc.slice(1, 0, 1).contiguous().to(torch::kCPU);
        auto unique_elems = at::_unique(elems);
        auto elem = std::get<0>(unique_elems);
        std::cout << "x loc " << elem << std::endl;
        long* unique_e = elem.data_ptr<long>();
        int numelements = elem.numel();
        unconverged_images += numelements;
        for (int ii=0; ii<numelements; ++ii) {
            outputs.index_put_({unique_e[ii], Slice(None), Slice(None)}, origdatain.index({iphi, unique_e[ii], Slice(None), Slice(None)}));
            unconvergedMap[unique_e[ii]] = 1;
        }
    }

    // displayGPUMemory("#E", my_rank);
    // Check for any infs
    pda_isnan = at::isinf(outputs);
    isPDnan = pda_isnan.any().item<bool>();
    if (isPDnan) {
        auto loc = torch::argwhere(pda_isnan == true);
        auto elems = loc.slice(1, 0, 1).contiguous().to(torch::kCPU);
        auto unique_elems = at::_unique(elems);
        auto elem = std::get<0>(unique_elems);
        std::cout << "y loc " << elem << std::endl;
        long* unique_e = elem.data_ptr<long>();
        int numelements = elem.numel();
        unconverged_images += numelements;
        for (int ii=0; ii<numelements; ++ii) {
            outputs.index_put_({unique_e[ii], Slice(None), Slice(None)}, origdatain.index({iphi, unique_e[ii], Slice(None), Slice(None)}));
            unconvergedMap[unique_e[ii]] = 1;
        }
    }

    // displayGPUMemory("#F", my_rank);
    pda_isnan = at::isneginf(outputs);
    isPDnan = pda_isnan.any().item<bool>();
    if (isPDnan) {
        auto loc = torch::argwhere(pda_isnan == true);
        auto elems = loc.slice(1, 0, 1).contiguous().to(torch::kCPU);
        auto unique_elems = at::_unique(elems);
        auto elem = std::get<0>(unique_elems);
        std::cout << "n loc " << elem << std::endl;
        long* unique_e = elem.data_ptr<long>();
        int numelements = elem.numel();
        unconverged_images += numelements;
        for (int ii=0; ii<numelements; ++ii) {
            outputs.index_put_({unique_e[ii], Slice(None), Slice(None)}, origdatain.index({iphi, unique_e[ii], Slice(None), Slice(None)}));
            unconvergedMap[unique_e[ii]] = 1;
        }
    }

    // displayGPUMemory("#G", my_rank);
    // Check for any very high values
    auto pos_outputs = at::abs(outputs);
    auto high = torch::argwhere(pos_outputs > 1e25);
    isPDnan = high.numel() > 0;
    if (isPDnan) {
        auto elems = high.slice(1, 0, 1).contiguous().to(torch::kCPU);
        auto unique_elems = at::_unique(elems);
        auto elem = std::get<0>(unique_elems);
        std::cout << "z loc " << elem << std::endl;
        long* unique_e = elem.data_ptr<long>();
        int numelements = elem.numel();
        unconverged_images += numelements;
        for (int ii=0; ii<numelements; ++ii) {
            outputs.index_put_({unique_e[ii], Slice(None), Slice(None)}, origdatain.index({iphi, unique_e[ii], Slice(None), Slice(None)}));
            unconvergedMap[unique_e[ii]] = 1;
        }
    }

    // displayGPUMemory("#H", my_rank);
    size_t vecIndex = 0;
    for (vecIndex=0; vecIndex<unconvergedNodeIndex.size(); ++vecIndex) {
        auto search = unconvergedMap.find(unconvergedNodeIndex[vecIndex]);
        if (search == unconvergedMap.end()) {
            outputs.index_put_({unconvergedNodeIndex[vecIndex], Slice(None), Slice(None)}, origdatain.index({iphi, unconvergedNodeIndex[vecIndex], Slice(None), Slice(None)}));
            unconverged_images += 1;
        }
    }
    tensors.push_back(outputs);
    // std::cout << "came here 4.5" << std::endl;
    myLagrangesTorch = lambdas_torch;
    unconverged_size += unconverged_images*sizeof(double)*myVxCount*myVyCount; // size of images
    unconverged_size += unconverged_images*sizeof(int)*2; // plane and node indexes
    unconverged_size += 1; // how many images are represented as is

    GPTLstop("compute_lambdas");
    // displayGPUMemory("#I", my_rank);
    at::Tensor combined = at::concat(tensors).reshape({1, nodes, myVxCount, myVyCount});
    compareQoIs(recondatain, combined);
    // std::vector <double> combinedVec(combined.data_ptr<double>(), combined.data_ptr<double>() + combined.numel());
    // writeOutput("i_f", combinedVec);
    // std::cout << "came here 4.9" << std::endl;
    if (unconverged_images > 256) {
        std::cout << "Error: number of unconverged images > 256. Please lower allowed error bound." << std::endl;
    }
    else if (unconverged_images > 0) {
        std::cout << "Unconverged images " << unconverged_images << " and sizes " << unconverged_size << std::endl;
    }
    // displayGPUMemory("#J", my_rank);
    return unconverged_size;
}

size_t LagrangeTorch::putLagrangeParameters(char* &bufferOut, size_t &bufferOutOffset, const char* precision)
{
    auto datain = myLagrangesTorch.contiguous().cpu();
    std::cout << "datain.type:" << datain.type() << std::endl;
    // (2023/03) FIXME: LagrangeTorchL2 returns float32
    datain = datain.to(torch::kFloat64);
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    myLagranges = datain_vec.data();
    if (!strcmp(precision, "single"))
    {
        int i, count = 0;
        int numObjs = myPlaneCount*myNodeCount;
        for (i=0; i<numObjs*4; i+=4) {
            *reinterpret_cast<float*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(float)) =
                      myLagranges[i];
        }
        for (i=0; i<numObjs; ++i) {
            *reinterpret_cast<float*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(float)) =
                      myLagranges[i+1];
        }
        for (i=0; i<numObjs; ++i) {
            *reinterpret_cast<float*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(float)) =
                      myLagranges[i+2];
        }
        for (i=0; i<numObjs; ++i) {
            *reinterpret_cast<float*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(float)) =
                      myLagranges[i+3];
        }
        return count * sizeof(float);
    }
    else {
        int i, count = 0;
        int numObjs = myPlaneCount*myNodeCount;
        for (i=0; i<numObjs*4; i+=4) {
            *reinterpret_cast<double*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(double)) =
                      myLagranges[i];
        }
        for (i=0; i<numObjs; ++i) {
            *reinterpret_cast<double*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(double)) =
                      myLagranges[i+1];
        }
        for (i=0; i<numObjs; ++i) {
            *reinterpret_cast<double*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(double)) =
                      myLagranges[i+2];
        }
        for (i=0; i<numObjs; ++i) {
            *reinterpret_cast<double*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(double)) =
                      myLagranges[i+3];
        }
        return count * sizeof(double);
    }
}

size_t LagrangeTorch::putResult(char* &bufferOut, size_t &bufferOutOffset, const char* precision)
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

char* LagrangeTorch::setDataFromCharBuffer(double* &reconData,
    const char* bufferIn, size_t sizeOut)
{
    int i, count = 0;
    for (i=0; i<4*myNodeCount; ++i) {
          myLagranges[i] = *reinterpret_cast<const double*>(
              bufferIn+(count++)*sizeof(double));
    }
    FILE* fp = fopen("PqMeshInfo.bin", "rb");
    int str_length = 0;
    fread(&str_length, sizeof(int), 1, fp);
    char meshFile[str_length];
    fread(meshFile, sizeof(char), str_length, fp);
    fclose(fp);
    readF0Params(std::string(meshFile, 0, str_length));
    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    auto recondatain = torch::from_blob((void *)reconData, {myPlaneCount, myVxCount, myNodeCount, myVyCount}, torch::kFloat64).to(this->device)
                  .permute({0, 2, 1, 3});
    recondatain = at::clamp(recondatain, at::Scalar(100), at::Scalar(1e+50));
    myLagrangesTorch =  torch::from_blob((void *)myLagranges, {myNodeCount, 4}, torch::kFloat64).to(this->device);
    auto V2_torch = myVolumeTorch * myVthTorch.reshape({myNodeCount,1,1}) * myVpTorch.reshape({1, 1, myVyCount});
    auto V3_torch = myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass;
    auto V4_torch = myVolumeTorch * at::pow(myVpTorch, at::Scalar(2)).reshape({1, myVyCount}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass;
    using namespace torch::indexing;
    auto l1 = myLagrangesTorch.index({Slice(None), 0}).reshape({myNodeCount, 1, 1}) * myVolumeTorch;
    auto l2 = myLagrangesTorch.index({Slice(None), 1}).reshape({myNodeCount, 1, 1}) * V2_torch;
    auto l3 = myLagrangesTorch.index({Slice(None), 2}).reshape({myNodeCount, 1, 1}) * V3_torch;
    auto l4 = myLagrangesTorch.index({Slice(None), 3}).reshape({myNodeCount, 1, 1}) * V4_torch;
    auto K = l1 + l2 + l3 + l4;
    auto outputs = recondatain[0]*at::exp(-K);
    auto datain = outputs.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    reconData = datain_vec.data();
    return reinterpret_cast<char*>(reconData);
}

// Get all variables from mesh file pertaining to ions and electrons
void LagrangeTorch::readF0Params(const std::string meshFile)
{
    // Read ADIOS2 files from here
    adios2::core::ADIOS adios("C++");
    auto &io = adios.DeclareIO("SubIO");
    auto *engine = &io.Open(meshFile, adios2::Mode::Read);

    // Get grid_vol
    auto var = io.InquireVariable<double>("f0_grid_vol_vonly");
    std::vector<std::size_t> volumeShape = var->Shape();

    var->SetSelection(adios2::Box<adios2::Dims>({0, myNodeOffset}, {volumeShape[0], myNodeCount}));
    engine->Get(*var, myGridVolume);

    // Get myF0Nvp
    auto var_nvp = io.InquireVariable<int>("f0_nvp");
    engine->Get(*var_nvp, myF0Nvp);

    // Get f0_nmu
    auto var_nmu = io.InquireVariable<int>("f0_nmu");
    engine->Get(*var_nmu, myF0Nmu);

    // Get f0_dvp
    auto var_dvp = io.InquireVariable<double>("f0_dvp");
    engine->Get(*var_dvp, myF0Dvp);

    // Get f0_dsmu
    auto var_dsmu = io.InquireVariable<double>("f0_dsmu");
    engine->Get(*var_dsmu, myF0Dsmu);

    // Get f0_T_ev
    auto var_ev = io.InquireVariable<double>("f0_T_ev");
    std::vector<std::size_t> evShape = var_ev->Shape();

    var_ev->SetSelection(adios2::Box<adios2::Dims>({0, myNodeOffset}, {evShape[0], myNodeCount}));
    engine->Get(*var_ev, myF0TEv);
    engine->Close();
    myGridVolumeTorch = torch::from_blob((void *)myGridVolume.data(), {volumeShape[0], myNodeCount}, torch::kFloat64).to(this->device);;
    // std::cout << "myGridVolumeTorch sizes " << myGridVolumeTorch.sizes() << std::endl;
    myF0TEvTorch = torch::from_blob((void *)myF0TEv.data(), {evShape[0], myNodeCount}, torch::kFloat64).to(this->device);;
    // std::cout << "myF0TEvTorch sizes " << myF0TEvTorch.sizes() << std::endl;
}

void LagrangeTorch::setVolume()
{
#if 0
    auto vp_vol_torch = torch::ones({myF0Nvp[0]*2+1}, myOption);
    vp_vol_torch[0] = 0.5;
    vp_vol_torch[-1] = 0.5;

    auto mu_vol_torch = torch::ones({myF0Nmu[0]+1}, myOption);
    mu_vol_torch[0] = 0.5;
    mu_vol_torch[-1] = 0.5;

    c10::string_view indexing{"ij"};
    std::vector<at::Tensor> args = torch::meshgrid({mu_vol_torch, vp_vol_torch}, indexing);
    at::Tensor cx = args[0].contiguous();
    at::Tensor cy = args[1].contiguous();
    at::Tensor mu_vp_vol_torch = at::transpose(cx, 0, 1) * at::transpose(cy, 0, 1);
    auto f0_grid_vol = myGridVolumeTorch[mySpecies];
#else
    std::vector<double> vp_vol;
    vp_vol.push_back(0.5);
    for (int ii = 1; ii<myF0Nvp[0]*2; ++ii) {
        vp_vol.push_back(1.0);
    }
    vp_vol.push_back(0.5);

    std::vector<double> mu_vol;
    mu_vol.push_back(0.5);
    for (int ii = 1; ii<myF0Nmu[0]; ++ii) {
        mu_vol.push_back(1.0);
    }
    mu_vol.push_back(0.5);

    std::vector<double> mu_vp_vol;
    for (int ii=0; ii<mu_vol.size(); ++ii) {
        for (int jj=0; jj<vp_vol.size(); ++jj) {
            mu_vp_vol.push_back(mu_vol[ii] * vp_vol[jj]);
        }
    }
    auto mu_vp_vol_torch=torch::from_blob((void *)mu_vp_vol.data(), {myVxCount, myVyCount}, torch::kFloat64).to(this->device);
    auto f0_grid_vol = myGridVolumeTorch[mySpecies];
    myVolumeTorch = f0_grid_vol.reshape({myNodeCount,1,1}) * mu_vp_vol_torch.reshape({1,myVxCount,myVyCount});
    myVolumeTorch = at::tile(myVolumeTorch, {myPlaneCount, 1, 1});
#endif
    return;
}

void LagrangeTorch::setVp()
{
    for (int ii = -myF0Nvp[0]; ii<myF0Nvp[0]+1; ++ii) {
        myVp.push_back(ii*myF0Dvp[0]);
    }
    myVpTorch = torch::from_blob((void *)myVp.data(), {myVyCount}, torch::kFloat64).to(this->device);
    return;
}

void LagrangeTorch::setMuQoi()
{
    auto mu = at::multiply(at::arange(myF0Nmu[0]+1, myOption), at::Scalar(myF0Dsmu[0]));
    myMuQoiTorch = at::pow(mu, 2);
    return;
}

void LagrangeTorch::setVth2()
{
    auto f0_T_ev_torch = myF0TEvTorch[mySpecies];
    myVth2Torch = at::multiply(f0_T_ev_torch, at::Scalar(mySmallElectronCharge/myParticleMass));
    myVthTorch = at::sqrt(myVth2Torch);
    myVth2Torch = at::tile(myVth2Torch, {myPlaneCount});
    myVthTorch = at::tile(myVthTorch, {myPlaneCount});
}

void LagrangeTorch::compute_C_qois(int iphi, at::Tensor &density, at::Tensor &upara, at::Tensor &tperp, at::Tensor &tpara, at::Tensor &n0, at::Tensor &t0, at::Tensor &dataInTorch)
{
    int i, j, k, nodes=myPlaneCount*myNodeCount;
    auto f0_f = dataInTorch[iphi];
    auto den = f0_f * myVolumeTorch;
    density = den.sum({1, 2});
    // std::cout << "Density compute_C_qois " << density.sizes() << std::endl;
    auto upar = f0_f * myVolumeTorch * myVthTorch.reshape({nodes,1,1}) * myVpTorch.reshape({1, 1, myVyCount});
    upara = upar.sum({1, 2})/density;
    // std::cout << "Upara compute_C_qois " << upara.sizes() << std::endl;
    auto upar_ = upara/myVthTorch;
    auto tper = f0_f * myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass;
    tperp = tper.sum({1, 2})/density/mySmallElectronCharge;
    // std::cout << "Tperp compute_C_qois " << tperp.sizes() << std::endl;
    auto en = 0.5*at::pow((myVpTorch.reshape({1, myVyCount})-upar_.reshape({nodes, 1})/myVthTorch.reshape({nodes, 1})),2);
    auto T_par = ((f0_f * myVolumeTorch * en.reshape({nodes, 1, myVyCount}) * myVth2Torch.reshape({nodes,1,1}) * myParticleMass));
    tpara = 2*T_par.sum({1, 2})/density/mySmallElectronCharge;
    // std::cout << "Tpara compute_C_qois " << tpara.sizes() << std::endl;
    n0 = density;
    t0 = (2.0*tper.sum({1, 2}) + T_par.sum({1, 2}))/3.0;
    return;
}

void dump(torch::Tensor ten, char* vname, int rank)
{
    char fname[255];
    sprintf(fname, "qoi-%s-%d.pt", vname, rank);
    torch::save(ten, fname);
    std::cout << ten << std::endl;
}

void LagrangeTorch::compareQoIs(at::Tensor& reconData, at::Tensor& bregData)
{
    int iphi = 0;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    torch::NoGradGuard no_grad;
    at::Tensor rdensity;
    at::Tensor rupara;
    at::Tensor rtperp;
    at::Tensor rtpara;
    at::Tensor rn0;
    at::Tensor rt0;
    compute_C_qois(iphi, rdensity, rupara, rtperp, rtpara, rn0, rt0, reconData);
#ifdef UF_DEBUG
        if (my_rank == 0) {
            std::cout << "Reconstructed density" << std::endl;
            dump(rdensity, "rdensity", my_rank);
            std::cout << "Reconstructed upara" << std::endl;
            dump(rupara, "rupara", my_rank);
            std::cout << "Reconstructed tperp" << std::endl;
            dump(rtperp, "rtperp", my_rank);
            std::cout << "Reconstructed tpara" << std::endl;
            dump(rtpara, "rtpara", my_rank);
        }
#endif
    // std::cout << "came here 4.6" << std::endl;
    at::Tensor bdensity;
    at::Tensor bupara;
    at::Tensor btperp;
    at::Tensor btpara;
    at::Tensor bn0;
    at::Tensor bt0;
    // std::cout << "Breg data " << bregData.sizes() << std::endl;
    compute_C_qois(iphi, bdensity, bupara, btperp, btpara, bn0, bt0, bregData);
    auto origdatain = myDataInTorch.reshape({1, myPlaneCount*myNodeCount, myVxCount, myVyCount});
    // std::cout << "came here 4.7" << std::endl;
    at::Tensor refdensity;
    at::Tensor refupara;
    at::Tensor reftperp;
    at::Tensor reftpara;
    at::Tensor refn0;
    at::Tensor reft0;
    compute_C_qois(iphi, refdensity, refupara, reftperp, reftpara, refn0, reft0, origdatain);
#ifdef UF_DEBUG
        if (my_rank == 0) {
            std::cout << "Reconstructed density" << std::endl;
            dump(refdensity, "refdensity", my_rank);
            std::cout << "Reconstructed upara" << std::endl;
            dump(refupara, "refupara", my_rank);
            std::cout << "Reconstructed tperp" << std::endl;
            dump(reftperp, "reftperp", my_rank);
            std::cout << "Reconstructed tpara" << std::endl;
            dump(reftpara, "reftpara", my_rank);
        }
#endif
    // std::cout << "came here 4.8" << std::endl;
    compareErrorsPD(origdatain, reconData, bregData, "PD", my_rank);
    compareErrorsPD(refdensity, rdensity, bdensity, "density", my_rank);
    compareErrorsPD(refupara, rupara, bupara, "upara", my_rank);
    compareErrorsPD(reftperp, rtperp, btperp, "tperp", my_rank);
    compareErrorsPD(reftpara, rtpara, btpara, "tpara", my_rank);
    compareErrorsPD(refn0, rn0, bn0, "n0", my_rank);
    compareErrorsPD(reft0, rt0, bt0, "T0", my_rank);
    // std::cout << "came here 4.85" << std::endl;
    return;
}

void LagrangeTorch::compareErrorsPD(at::Tensor& dataIn, at::Tensor& reconData, at::Tensor& bregData, const char* etype, int rank)
{
    double pd_b = at::pow((dataIn-reconData),2).sum().item().to<double>();
    double pd_a = at::pow((dataIn-bregData),2).sum().item().to<double>();
    double pd_size_b, pd_size_a;
    double pd_min_b, pd_min_a;
    double pd_max_b, pd_max_a;
    pd_max_a = pd_max_b = dataIn.max().item().to<double>();
    pd_min_a = pd_min_b = dataIn.min().item().to<double>();
    pd_size_a = pd_size_b = reconData.numel();
    // get total error for recon
    double pd_e_b;
    double pd_s_b;
    double pd_omin_b;
    double pd_omax_b;
    MPI_Allreduce(&pd_b, &pd_e_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_size_b, &pd_s_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_min_b, &pd_omin_b, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_max_b, &pd_omax_b, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double pd_e_a;
    double pd_s_a;
    double pd_omin_a;
    double pd_omax_a;
    MPI_Allreduce(&pd_a, &pd_e_a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_size_a, &pd_s_a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_min_a, &pd_omin_a, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_max_a, &pd_omax_a, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << ((int) mySpecies) << " Overall " << etype << " Error: " << sqrt(pd_e_b/pd_s_b)/(pd_omax_b-pd_omin_b) << " " << sqrt(pd_e_a/pd_s_a)/(pd_omax_a-pd_omin_a) << std::endl;
    }
}
