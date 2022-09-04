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

// Define static class members
at::TensorOptions LagrangeTorch::ourGPUOptions = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
at::TensorOptions LagrangeTorch::ourCPUOptions = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

LagrangeTorch::LagrangeTorch(const char* species)
  : LagrangeOptimizer(species)
{
}

LagrangeTorch::LagrangeTorch(size_t planeOffset,
    size_t nodeOffset, size_t p, size_t n, size_t vx, size_t vy,
    const uint8_t species)
  : LagrangeOptimizer(planeOffset, nodeOffset, p, n, vx, vy, species)
{
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
    auto datain = torch::from_blob((void *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]}, torch::kFloat64).to(torch::kCUDA)
                  .permute({0, 2, 1, 3});
    myDataInTorch = datain;
    readF0Params(meshFile);
    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    myMaxValue = myDataInTorch.max().item().to<double>();;
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (my_rank == 0) {
        printf ("%d Time Taken for QoI Computation: %f\n", mySpecies, (end-start));
    }
}

void LagrangeTorch::computeLagrangeParameters(
    const double* reconData, adios2::Dims blockCount)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    int ii, i, j, k, l, m;
    auto recondatain = torch::from_blob((void *)reconData, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]}, torch::kFloat64).to(torch::kCUDA)
                  .permute({0, 2, 1, 3});
    recondatain = at::clamp(recondatain, at::Scalar(100), at::Scalar(1e+50));
    auto V2_torch = myVolumeTorch * myVthTorch.reshape({myNodeCount,1,1}) * myVpTorch.reshape({1, 1, myVyCount});
    auto V3_torch = myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass;
    auto V4_torch = myVolumeTorch * at::pow(myVpTorch, at::Scalar(2)).reshape({1, myVyCount}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass;

    int breg_index = 0;
    int iphi, idx;
    std::vector <at::Tensor> tensors;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        std::vector<double> D(myNodeCount, 0);
        auto f0_f_torch = myDataInTorch[iphi];
        auto D_torch = (f0_f_torch * myVolumeTorch).sum({1, 2});

        std::vector<double> U(myNodeCount, 0);
        std::vector<double> Tperp(myNodeCount, 0);
        auto U_torch = ((f0_f_torch * myVolumeTorch * myVthTorch.reshape({myNodeCount,1,1}) * myVpTorch.reshape({1, 1, myVyCount})).sum({1, 2}))/D_torch;
        auto Tperp_torch = ((f0_f_torch * myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass).sum({1,2}))/D_torch/mySmallElectronCharge;

        std::vector<double> Tpara(myNodeCount, 0);
        std::vector<double> Rpara(myNodeCount, 0);
        auto en_torch = 0.5*at::pow((myVpTorch.reshape({1, myVyCount})-U_torch.reshape({myNodeCount, 1})/myVthTorch.reshape({myNodeCount, 1})),2);
        auto Tpara_torch = 2*((f0_f_torch * myVolumeTorch * en_torch.reshape({myNodeCount, 1, myVyCount}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass).sum({1, 2}))/D_torch/mySmallElectronCharge;
        auto Rpara_torch = mySmallElectronCharge*Tpara_torch + myVth2Torch * myParticleMass * at::pow((U_torch/myVthTorch), 2);

        int count_unLag = 0;
        std::vector <int> node_unconv;
        double maxD = D_torch.max().item().to<double>();
        double maxU = U_torch.max().item().to<double>();
        double maxTperp = Tperp_torch.max().item().to<double>();
        double maxTpara = Rpara_torch.max().item().to<double>();
        auto aD_torch = D_torch*mySmallElectronCharge;

        double DeB = pow(maxD*1e-05, 2);
        double UeB = pow(maxU*1e-05, 2);
        double TperpEB = pow(maxTperp*1e-05, 2);
        double TparaEB = pow(maxTpara*1e-05, 2);
        double PDeB = pow(myMaxValue*1e-05, 2);

        int maxIter = 50;
        auto lambdas_torch = torch::zeros({myNodeCount,4}, ourGPUOptions);
        auto gradients_torch = torch::zeros({myNodeCount,4}, ourGPUOptions);
        auto hessians_torch = torch::zeros({myNodeCount,4,4}, ourGPUOptions);
        auto L2_den = torch::zeros({myNodeCount, maxIter+1}, ourGPUOptions);
        auto L2_upara = torch::zeros({myNodeCount, maxIter+1}, ourGPUOptions);
        auto L2_tperp = torch::zeros({myNodeCount, maxIter+1}, ourGPUOptions);
        auto L2_rpara = torch::zeros({myNodeCount, maxIter+1}, ourGPUOptions);
        auto L2_pd = torch::zeros({myNodeCount, maxIter+1}, ourGPUOptions);

        auto K = torch::zeros({myNodeCount,myVxCount,myVyCount}, ourGPUOptions);
        int count = 0;
        int converged = 0;

        using namespace torch::indexing;
        while (count < maxIter)
        {
            gradients_torch.index_put_({Slice(None), 0}, (-((recondatain[iphi]*myVolumeTorch*at::exp(-K)).sum({1,2})) + D_torch));
            gradients_torch.index_put_({Slice(None), 1}, (-((recondatain[iphi]*V2_torch*at::exp(-K)).sum({1,2})) + U_torch*D_torch));
            gradients_torch.index_put_({Slice(None), 2}, (-((recondatain[iphi]*V3_torch*at::exp(-K)).sum({1,2})) + Tperp_torch*aD_torch));
            gradients_torch.index_put_({Slice(None), 3}, (-((recondatain[iphi]*V4_torch*at::exp(-K)).sum({1,2})) + Rpara_torch*D_torch));
            hessians_torch.index_put_({Slice(None), 0, 0}, (recondatain[iphi]*at::pow(myVolumeTorch,2)*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 0, 1}, (recondatain[iphi]*myVolumeTorch*V2_torch*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 0, 2}, (recondatain[iphi]*myVolumeTorch*V3_torch*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 0, 3}, (recondatain[iphi]*myVolumeTorch*V4_torch*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 1, 0}, hessians_torch.index({Slice(None), 0, 1}));
            hessians_torch.index_put_({Slice(None), 1, 1}, (recondatain[iphi]*at::pow(V2_torch, 2)*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 1, 2}, (recondatain[iphi]*V2_torch*V3_torch*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 1, 3}, (recondatain[iphi]*V2_torch*V4_torch*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 2, 0}, hessians_torch.index({Slice(None), 0, 2}));
            hessians_torch.index_put_({Slice(None), 2, 1}, hessians_torch.index({Slice(None), 1, 2}));
            hessians_torch.index_put_({Slice(None), 2, 2}, (recondatain[iphi]*at::pow(V3_torch, 2)*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 2, 3}, (recondatain[iphi]*V3_torch*V4_torch*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 3, 0}, hessians_torch.index({Slice(None), 0, 3}));
            hessians_torch.index_put_({Slice(None), 3, 1}, hessians_torch.index({Slice(None), 1, 3}));
            hessians_torch.index_put_({Slice(None), 3, 2}, hessians_torch.index({Slice(None), 2, 3}));
            hessians_torch.index_put_({Slice(None), 3, 3}, (recondatain[iphi]*at::pow(V4_torch, 2)*at::exp(-K)).sum({1,2}));
            lambdas_torch = lambdas_torch - at::squeeze(at::bmm(hessians_torch.inverse(), gradients_torch.reshape({myNodeCount, 4, 1})));
            auto l1 = lambdas_torch.index({Slice(None), 0}).reshape({myNodeCount, 1, 1}) * myVolumeTorch;
            auto l2 = lambdas_torch.index({Slice(None), 1}).reshape({myNodeCount, 1, 1}) * V2_torch;
            auto l3 = lambdas_torch.index({Slice(None), 2}).reshape({myNodeCount, 1, 1}) * V3_torch;
            auto l4 = lambdas_torch.index({Slice(None), 3}).reshape({myNodeCount, 1, 1}) * V4_torch;
            K = l1 + l2 + l3 + l4;
            count = count + 1;
            if (count % 10 == 0) {
                auto breg_result = recondatain[iphi]*at::exp(-K);
                auto update_D = (breg_result * myVolumeTorch).sum({1,2});
                auto update_U = (breg_result * V2_torch).sum({1,2})/D_torch;
                auto update_Tperp = (breg_result * V3_torch).sum({1,2})/aD_torch;
                auto update_Rpara = (breg_result * V4_torch).sum({1,2})/D_torch;
                auto rmse_pd = at::pow(breg_result - myDataInTorch[iphi], 2).sum({1,2});
                L2_den.index_put_({Slice(None), count}, at::pow(update_D-D_torch, 2));
                L2_upara.index_put_({Slice(None), count}, at::pow(update_U-U_torch, 2));
                L2_tperp.index_put_({Slice(None), count}, at::pow(update_Tperp-Tperp_torch, 2));
                L2_rpara.index_put_({Slice(None), count}, at::pow(update_Rpara-Rpara_torch, 2));
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
                        break;
                    }
                }
            }
        }
        if (count == maxIter && converged == 0)
        {
            std::cout << "All nodes did not converge on rank " << my_rank << std::endl;
        }
        auto outputs = recondatain[iphi]*at::exp(-K);
        tensors.push_back(outputs);
        myLagrangesTorch = lambdas_torch;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (my_rank == 0) {
        printf ("%d Time Taken for Lagrange Computations: %f\n", mySpecies, end-start);
    }
    at::Tensor combined = at::concat(tensors).reshape({myPlaneCount, myNodeCount, myVxCount, myVyCount});
    compareQoIs(recondatain, combined);
    return;
}

size_t LagrangeTorch::putLagrangeParameters(char* &bufferOut, size_t &bufferOutOffset, const char* precision)
{
    auto datain = myLagrangesTorch.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    myLagranges = datain_vec.data();
    if (!strcmp(precision, "float"))
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
    auto recondatain = torch::from_blob((void *)reconData, {myPlaneCount, myVxCount, myNodeCount, myVyCount}, torch::kFloat64).to(torch::kCUDA)
                  .permute({0, 2, 1, 3});
    recondatain = at::clamp(recondatain, at::Scalar(100), at::Scalar(1e+50));
    myLagrangesTorch =  torch::from_blob((void *)myLagranges, {myNodeCount, 4}, torch::kFloat64).to(torch::kCUDA);
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
    myGridVolumeTorch = torch::from_blob((void *)myGridVolume.data(), {volumeShape[0], myNodeCount}, torch::kFloat64).to(torch::kCUDA);;
    myF0TEvTorch = torch::from_blob((void *)myF0TEv.data(), {evShape[0], myNodeCount}, torch::kFloat64).to(torch::kCUDA);;
}

void LagrangeTorch::setVolume()
{
    auto vp_vol_torch = torch::ones({myF0Nvp[0]*2+1}, ourGPUOptions);
    vp_vol_torch[0] = 0.5;
    vp_vol_torch[-1] = 0.5;

    auto mu_vol_torch = torch::ones({myF0Nmu[0]+1}, ourGPUOptions);
    mu_vol_torch[0] = 0.5;
    mu_vol_torch[-1] = 0.5;

    c10::string_view indexing{"ij"};
    std::vector<at::Tensor> args = torch::meshgrid({mu_vol_torch, vp_vol_torch}, indexing);
    at::Tensor cx = args[0].contiguous();
    at::Tensor cy = args[1].contiguous();
    at::Tensor mu_vp_vol_torch = at::transpose(cx, 0, 1) * at::transpose(cy, 0, 1);
    auto f0_grid_vol = myGridVolumeTorch[mySpecies];
    myVolumeTorch = at::multiply(f0_grid_vol.reshape({myNodeCount,1}), mu_vp_vol_torch.reshape({1,myVxCount*myVyCount}));
    myVolumeTorch = myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount});
    return;
}

void LagrangeTorch::setVp()
{
    myVpTorch = at::multiply(at::arange(-myF0Nvp[0], myF0Nvp[0]+1, ourGPUOptions), at::Scalar(myF0Dvp[0]));
    return;
}

void LagrangeTorch::setMuQoi()
{
    auto mu = at::multiply(at::arange(myF0Nmu[0]+1, ourGPUOptions), at::Scalar(myF0Dsmu[0]));
    myMuQoiTorch = at::pow(mu, at::Scalar(2));

    return;
}

void LagrangeTorch::setVth2()
{
    auto f0_T_ev_torch = myF0TEvTorch[mySpecies];
    myVth2Torch = at::multiply(f0_T_ev_torch, at::Scalar(mySmallElectronCharge/myParticleMass));
    myVthTorch = at::sqrt(myVth2Torch);
}

void LagrangeTorch::compute_C_qois(int iphi, at::Tensor &density, at::Tensor &upara, at::Tensor &tperp, at::Tensor &tpara, at::Tensor &n0, at::Tensor &t0, at::Tensor &dataInTorch)
{
    int i, j, k;
    auto f0_f = dataInTorch[iphi];
    auto den = f0_f * myVolumeTorch;
    density = den.sum({1, 2});
    auto upar = f0_f * myVolumeTorch * myVthTorch.reshape({myNodeCount,1,1}) * myVpTorch.reshape({1, 1, myVyCount});
    upara = upar.sum({1, 2})/density;
    auto upar_ = upara/myVthTorch;
    auto tper = f0_f * myVolumeTorch * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass;
    tperp = tper.sum({1, 2})/density/mySmallElectronCharge;
    auto en = 0.5*at::pow((myVpTorch.reshape({1, myVyCount})-upar_.reshape({myNodeCount, 1})/myVthTorch.reshape({myNodeCount, 1})),2);
    auto T_par = ((f0_f * myVolumeTorch * en.reshape({myNodeCount, 1, myVyCount}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass));
    tpara = 2*T_par.sum({1, 2})/density/mySmallElectronCharge;
    n0 = density;
    t0 = (2.0*tper.sum({1, 2}) + T_par.sum({1, 2}))/3.0;
    return;
}

void LagrangeTorch::compareQoIs(at::Tensor& reconData, at::Tensor& bregData)
{
    int iphi;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    at::Tensor rdensity;
    at::Tensor rupara;
    at::Tensor rtperp;
    at::Tensor rtpara;
    at::Tensor rn0;
    at::Tensor rt0;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        compute_C_qois(iphi, rdensity, rupara, rtperp, rtpara, rn0, rt0, reconData);
    }
    at::Tensor bdensity;
    at::Tensor bupara;
    at::Tensor btperp;
    at::Tensor btpara;
    at::Tensor bn0;
    at::Tensor bt0;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        compute_C_qois(iphi, bdensity, bupara, btperp, btpara, bn0, bt0, bregData);
    }
    at::Tensor refdensity;
    at::Tensor refupara;
    at::Tensor reftperp;
    at::Tensor reftpara;
    at::Tensor refn0;
    at::Tensor reft0;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        compute_C_qois(iphi, refdensity, refupara, reftperp, reftpara, refn0, reft0, myDataInTorch);
    }
    compareErrorsPD(myDataInTorch, reconData, bregData, "PD", my_rank);
    compareErrorsPD(refdensity, rdensity, bdensity, "density", my_rank);
    compareErrorsPD(refupara, rupara, bupara, "upara", my_rank);
    compareErrorsPD(reftperp, rtperp, btperp, "tperp", my_rank);
    compareErrorsPD(reftpara, rtpara, btpara, "tpara", my_rank);
    compareErrorsPD(refn0, rn0, bn0, "n0", my_rank);
    compareErrorsPD(reft0, rt0, bt0, "T0", my_rank);
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
