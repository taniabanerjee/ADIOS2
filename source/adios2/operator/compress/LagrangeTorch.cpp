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
#include "KmeansMPI.h"
#include <omp.h>
#include <string_view>
#define GET4D(d0, d1, d2, d3, i, j, k, l) ((d1 * d2 * d3) * i + (d2 * d3) * j + d3 * k + l)


// int mpi_kmeans(double*, int, int, int, float, int*&, double*&);

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
#ifdef UF_DEBUG
    printf("#planes: %d, #nodes: %d, #vx: %d, #vy: %d\n", myPlaneCount, myNodeCount, myVxCount, myVyCount);
#endif
    myLocalElements = myNodeCount * myPlaneCount * myVxCount * myVyCount;
    auto datain = torch::from_blob((void *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]}, torch::kFloat64).to(torch::kCUDA)
                  .permute({0, 2, 1, 3});
    myDataInTorch = datain;
    datain = datain.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    myDataIn = datain_vec;
    readF0Params(meshFile);
    setVolume();
    setVp();
    setMuQoi();
    setVth2();
#ifdef UF_DEBUG
    printf ("volume: gv %d f0_nvp %d f0_nmu %d, vp: %d, vth: %d, vth2: %d, mu_qoi: %d\n", myGridVolume.size(), myF0Nvp.size(), myF0Nmu.size(), myVp.size(), myVth.size(), myVth2.size(), myMuQoi.size());
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        compute_C_qois(iphi, myDensity, myUpara, myTperp, myTpara, myN0, myT0, myDataIn.data());
    }
#endif
    myMaxValue = 0;
    for (size_t i = 0; i < myLocalElements; ++i) {
        myMaxValue = (myMaxValue > myDataIn[i]) ? myMaxValue : myDataIn[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (my_rank == 0) {
        printf ("%d Time Taken for QoI Computation: %f\n", mySpecies, (end-start));
    }
}

void LagrangeTorch::computeLagrangeParameters(
    const double* reconData, const int applyPQ)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double start, end, start1;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    int ii, i, j, k, l, m;
    // pass adios2::Dims blockCount
    // auto datain = torch::from_blob((void *)reconData, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]}, torch::kFloat64).to(torch::kCUDA)
    auto recondatain = torch::from_blob((void *)reconData, {myPlaneCount, myVxCount, myNodeCount, myVyCount}, torch::kFloat64).to(torch::kCUDA)
                  .permute({0, 2, 1, 3});
    recondatain = at::clamp(recondatain, at::Scalar(100), at::Scalar(1e+50));
    auto datain = recondatain.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    reconData = datain_vec.data();
    std::vector <double> V2 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V3 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V4 (myNodeCount*myVxCount*myVyCount, 0);
    auto V2_torch = myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount}) * myVthTorch.reshape({myNodeCount,1,1}) * myVpTorch.reshape({1, 1, myVyCount});
    auto V3_torch = myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount}) * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass;
    auto V4_torch = myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount}) * at::pow(myVpTorch, at::Scalar(2)).reshape({1, myVyCount}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass;
    // std::cout << "myVp sizes" << at::pow(myVpTorch, at::Scalar(2)).reshape({1, myVyCount}) << std::endl;
    // std::cout << "myVth2 sizes" << myVth2Torch.reshape({myNodeCount, 1, 1}).sizes() << std::endl;
    // std::cout << "V4_torch sizes" << V4_torch.sizes() << std::endl;

    auto datain2 = V2_torch.contiguous().cpu();
    std::vector<double> datain_vec2(datain2.data_ptr<double>(), datain2.data_ptr<double>() + datain2.numel());
    V2 = datain_vec2;

    auto datain3 = V3_torch.contiguous().cpu();
    std::vector<double> datain_vec3(datain3.data_ptr<double>(), datain3.data_ptr<double>() + datain3.numel());
    V3 = datain_vec3;

    auto datain4 = V4_torch.contiguous().cpu();
    std::vector<double> datain_vec4(datain4.data_ptr<double>(), datain4.data_ptr<double>() + datain4.numel());
    V4 = datain_vec4;

    int breg_index = 0;
    int iphi, idx;
    std::vector <double> new_recon_vec;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        const double* f0_f = &myDataIn[iphi*myNodeCount*myVxCount*myVyCount];
        std::vector<double> D(myNodeCount, 0);
        auto f0_f_torch = myDataInTorch[iphi];
        std::cout << "f0_f_torch sizes" << f0_f_torch.sizes() << std::endl;
        auto D_torch = (f0_f_torch *  myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount})).sum({1, 2});
        std::cout << "D sizes" << D_torch.sizes() << std::endl;

        auto datain5 = D_torch.contiguous().cpu();
        std::vector<double> datain_vec5(datain5.data_ptr<double>(), datain5.data_ptr<double>() + datain5.numel());
        D = datain_vec5;

        std::vector<double> U(myNodeCount, 0);
        std::vector<double> Tperp(myNodeCount, 0);
        auto U_torch = ((f0_f_torch * myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount}) * myVthTorch.reshape({myNodeCount,1,1}) * myVpTorch.reshape({1, 1, myVyCount})).sum({1, 2}))/D_torch;
        auto Tperp_torch = ((f0_f_torch * myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount}) * 0.5 * myMuQoiTorch.reshape({1,myVxCount,1}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass).sum({1,2}))/D_torch/mySmallElectronCharge;

        auto datain6 = U_torch.contiguous().cpu();
        std::vector<double> datain_vec6(datain6.data_ptr<double>(), datain6.data_ptr<double>() + datain6.numel());
        U = datain_vec6;

        auto datain7 = Tperp_torch.contiguous().cpu();
        std::vector<double> datain_vec7(datain7.data_ptr<double>(), datain7.data_ptr<double>() + datain7.numel());
        Tperp = datain_vec7;

#if 0
        if (U == datain_vec6) {
            std::cout << "vectors U and U_torch are equal" << std::endl;
        }
        else {
            std::cout << "vectors U and U_torch are not equal" << std::endl;
            for (i=0; i<myNodeCount; ++i) {
              if (abs(U[i] - datain_vec6[i]) > 0.1) {
                std::cout << i << " " << U[i] << " " << datain_vec6[i] << std::endl;
              }
            }
        }
#endif
        std::vector<double> Tpara(myNodeCount, 0);
        std::vector<double> Rpara(myNodeCount, 0);
        auto en_torch = 0.5*at::pow((myVpTorch.reshape({1, myVyCount})-U_torch.reshape({myNodeCount, 1})/myVthTorch.reshape({myNodeCount, 1})),2);
        std::cout << "en_torch sizes" << en_torch.sizes() << std::endl;
        auto Tpara_torch = 2*((f0_f_torch * myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount}) * en_torch.reshape({myNodeCount, 1, myVyCount}) * myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass).sum({1, 2}))/D_torch/mySmallElectronCharge;
        // auto Rpara_torch = mySmallElectronCharge*Tpara_torch + (myVth2Torch.reshape({myNodeCount,1,1}) * myParticleMass at::pow((U_torch.reshape({myNodeCount, 1})/myVthTorch.reshape({myNodeCount, 1})), 2);
        auto Rpara_torch = mySmallElectronCharge*Tpara_torch + myVth2Torch * myParticleMass * at::pow((U_torch/myVthTorch), 2);
        std::cout << "Rpara_torch sizes" << Rpara_torch.sizes() << std::endl;

        auto datain8 = Rpara_torch.contiguous().cpu();
        std::vector<double> datain_vec8(datain8.data_ptr<double>(), datain8.data_ptr<double>() + datain8.numel());
        Rpara = datain_vec8;

        int count_unLag = 0;
        std::vector <int> node_unconv;
        double maxD = D_torch.max().item().to<double>();
        double maxU = U_torch.max().item().to<double>();
        double maxTperp = Tperp_torch.max().item().to<double>();
        double maxTpara = Rpara_torch.max().item().to<double>();
        auto aD_torch = D_torch*mySmallElectronCharge;

        double DeB = pow(maxD*1e-09, 2);
        double UeB = pow(maxU*1e-09, 2);
        double TperpEB = pow(maxTperp*1e-09, 2);
        double TparaEB = pow(maxTpara*1e-09, 2);
        double PDeB = pow(myMaxValue*1e-09, 2);

        int maxIter = 5;
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
        auto lambdas_torch = torch::zeros({myNodeCount,4}, options);
        auto gradients_torch = torch::zeros({myNodeCount,4}, options);
        auto hessians_torch = torch::zeros({myNodeCount,4,4}, options);

        auto K = torch::zeros({myNodeCount,39,39}, options);
        int count = 0;

#if 1
        using namespace torch::indexing;
        while (count < maxIter)
        {
            gradients_torch.index_put_({Slice(None), 0}, (-((recondatain[iphi]*myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount})*at::exp(-K)).sum({1,2})) + D_torch));
            gradients_torch.index_put_({Slice(None), 1}, (-((recondatain[iphi]*V2_torch*at::exp(-K)).sum({1,2})) + U_torch*D_torch));
            gradients_torch.index_put_({Slice(None), 2}, (-((recondatain[iphi]*V3_torch*at::exp(-K)).sum({1,2})) + Tperp_torch*aD_torch));
            gradients_torch.index_put_({Slice(None), 3}, (-((recondatain[iphi]*V4_torch*at::exp(-K)).sum({1,2})) + Rpara_torch*D_torch));
            hessians_torch.index_put_({Slice(None), 0, 0}, (recondatain[iphi]*at::pow(myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount}),2)*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 0, 1}, (recondatain[iphi]*myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount})*V2_torch*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 0, 2}, (recondatain[iphi]*myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount})*V3_torch*at::exp(-K)).sum({1,2}));
            hessians_torch.index_put_({Slice(None), 0, 3}, (recondatain[iphi]*myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount})*V4_torch*at::exp(-K)).sum({1,2}));
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
#if 0
            if (my_rank == 0) {
                std::cout << "Count " << count << std::endl;
                std::cout << "Hessians size " << hessians_torch.sizes() << std::endl;
                std::cout << "Hessians " << hessians_torch << std::endl;
                std::cout << "Hessians inverse " << hessians_torch.inverse() << std::endl;
                std::cout << "Gradients " << gradients_torch << std::endl;
                std::cout << "MXM shape" << at::bmm(hessians_torch.inverse(), gradients_torch.reshape({myNodeCount, 4, 1})).sizes() << std::endl;
                std::cout << "MXM " << at::bmm(hessians_torch.inverse(), gradients_torch.reshape({myNodeCount, 4, 1})) << std::endl;
                std::cout << "Final shape " << at::squeeze(at::bmm(hessians_torch.inverse(), gradients_torch.reshape({myNodeCount, 4, 1}))).sizes() << std::endl;
                std::cout << "Final result " << at::squeeze(at::bmm(hessians_torch.inverse(), gradients_torch.reshape({myNodeCount, 4, 1}))) << std::endl;
            }
#endif
            auto l1 = lambdas_torch.index({Slice(None), 0}).reshape({myNodeCount, 1, 1}) * myVolumeTorch.reshape({myNodeCount, myVxCount, myVyCount});
            auto l2 = lambdas_torch.index({Slice(None), 1}).reshape({myNodeCount, 1, 1}) * V2_torch;
            auto l3 = lambdas_torch.index({Slice(None), 2}).reshape({myNodeCount, 1, 1}) * V3_torch;
            auto l4 = lambdas_torch.index({Slice(None), 3}).reshape({myNodeCount, 1, 1}) * V4_torch;
            K = l1 + l2 + l3 + l4;
            // std::cout << "Count: " << count << std::endl;
            // std::cout << "Gradient: " << gradients_torch << std::endl;
            // std::cout << "Hessians: " << hessians_torch << std::endl;
            // std::cout << "Lagrange parameters: " << lambdas_torch << std::endl;
            // std::cout << "K: " << K << std::endl;
            // std::cout << "exp(-K): " << at::exp(-K) << std::endl;
            count = count + 1;
        }
        std::cout << "K shape " << K.sizes() << std::endl;
        std::cout << "exp(-K) shape " << at::exp(-K).sizes() << std::endl;
        auto outputs = recondatain[iphi]*at::exp(-K);
        auto datain9 = outputs.contiguous().cpu();
        std::vector<double> datain_vec9(datain9.data_ptr<double>(), datain9.data_ptr<double>() + datain9.numel());
        new_recon_vec.insert(std::end(new_recon_vec), std::begin(datain_vec9), std::end(datain_vec9));

        auto datain11 = lambdas_torch.contiguous().cpu();
        std::vector<double> datain_vec11(datain11.data_ptr<double>(), datain11.data_ptr<double>() + datain11.numel());
        double* myGradients = datain_vec11.data();

        auto datain10 = lambdas_torch.contiguous().cpu();
        std::vector<double> datain_vec10(datain10.data_ptr<double>(), datain10.data_ptr<double>() + datain10.numel());
        myLagranges = datain_vec10.data();
#else
        myLagranges = new double[4*myNodeCount];
        for (idx=0; idx<myNodeCount; ++idx) {
            int count = 0;
            double gradients[4] = {0.0, 0.0, 0.0, 0.0};
            double hessians[4][4] = {0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0};
            double K[myVxCount*myVyCount];
            double breg_result[myVxCount*myVyCount];
            memset(K, 0, myVxCount*myVyCount*sizeof(double));
            std::vector <double> L2_den (maxIter, 0);
            std::vector <double> L2_upara (maxIter, 0);
            std::vector <double> L2_tperp (maxIter, 0);
            std::vector <double> L2_tpara (maxIter, 0);
            std::vector <double> L2_PD (maxIter, 0);
            std::fill(L2_den.begin(), L2_den.end(), 0);
            std::fill(L2_upara.begin(), L2_upara.end(), 0);
            std::fill(L2_tperp.begin(), L2_tperp.end(), 0);
            std::fill(L2_tpara.begin(), L2_tpara.end(), 0);
            std::fill(L2_PD.begin(), L2_PD.end(), 0);
            const double* recon_one = &reconData[myNodeCount*myVxCount*myVyCount*iphi + myVxCount*myVyCount*idx];
            double lambdas[4] = {0.0, 0.0, 0.0, 0.0};
            count = 0;
            double aD = D[idx]*mySmallElectronCharge;
            int i;
            while (1) {
                for (i=0; i<myVxCount*myVyCount; ++i) {
                    K[i] = lambdas[0]*myVolume[myVxCount*myVyCount*idx + i] +
                           lambdas[1]*V2[myVxCount*myVyCount*idx + i] +
                           lambdas[2]*V3[myVxCount*myVyCount*idx + i] +
                           lambdas[3]*V4[myVxCount*myVyCount*idx + i];
                }
                double update_D=0, update_U=0, update_Tperp=0, update_Tpara=0, rmse_pd=0;
                if (count > 0) {
                    for (i=0; i<myVxCount*myVyCount; ++i) {
                        breg_result[i] = recon_one[i]*
                            exp(-K[i]);
                        update_D += breg_result[i]*myVolume[
                            myVxCount*myVyCount*idx + i];
                        update_U += breg_result[i]*V2[
                            myVxCount*myVyCount*idx + i]/D[idx];
                        update_Tperp += breg_result[i]*V3[
                            myVxCount*myVyCount*idx + i]/aD;
                        update_Tpara += breg_result[i]*V4[
                            myVxCount*myVyCount*idx + i]/D[idx];
                        rmse_pd += pow((breg_result[i] - f0_f[myVxCount*myVyCount*idx + i]), 2);
                    }
                    L2_den[count] = pow((update_D - D[idx]), 2);
                    L2_upara[count] = pow((update_U - U[idx]), 2);
                    L2_tperp[count] = pow((update_Tperp-Tperp[idx]), 2);
                    L2_tpara[count] = pow((update_Tpara-Rpara[idx]), 2);
                    L2_PD[count] = sqrt(rmse_pd);
                    bool c1, c2, c3, c4;
                    bool converged = (isConverged(L2_den, DeB, count)
                        && isConverged(L2_upara, UeB, count)
                        && isConverged(L2_tpara, TparaEB, count)
                        && isConverged(L2_tperp, TperpEB, count))
                        && isConverged(L2_PD, PDeB, count);
                    if (converged) {
                        // for (i=0; i<myVxCount*myVyCount; ++i) {
                            // breg_recon[breg_index++] = breg_result[i];
                        // }
                        /*
                        double mytperp[1521];
                        for (i=0; i<myVxCount*myVyCount; ++i) {
                            mytperp[i] = breg_result[i]*V3[
                                myVxCount*myVyCount*idx + i]/aD;
                        }
                        printf ("Mytperp %d\n", mytperp[0]);
                        */
                        myLagranges[idx*4] = lambdas[0];
                        myLagranges[idx*4 + 1] = lambdas[1];
                        myLagranges[idx*4 + 2] = lambdas[2];
                        myLagranges[idx*4 + 3] = lambdas[3];
                        break;
                    }
                    else if (count == maxIter && !converged) {
                        // for (i=0; i<myVxCount*myVyCount; ++i) {
                            // breg_recon[breg_index++] = recon_one[i];
                        // }
                        converged = (isConverged(L2_den, DeB, count)
                        && isConverged(L2_upara, UeB, count)
                        && isConverged(L2_tpara, TparaEB, count)
                        && isConverged(L2_tperp, TperpEB, count)
                        && isConverged(L2_PD, PDeB, count));
                        myLagranges[idx*4] = 0;
                        myLagranges[idx*4 + 1] = 0;
                        myLagranges[idx*4 + 2] = 0;
                        myLagranges[idx*4 + 3] = 0;
                        printf ("Node %d did not converge\n", idx);
                        count_unLag = count_unLag + 1;
                        {
                            node_unconv.push_back(idx);
                        }
                        break;
                    }
                }
                double gvalue1 = D[idx], gvalue2 = U[idx]*D[idx];
                double gvalue3 = Tperp[idx]*aD, gvalue4 = Rpara[idx]*D[idx];
                double hvalue1 = 0, hvalue2 = 0, hvalue3 = 0, hvalue4 = 0;
                double hvalue5 = 0, hvalue6 = 0, hvalue7 = 0;
                double hvalue8 = 0, hvalue9 = 0, hvalue10 = 0;

                for (i=0; i<myVxCount*myVyCount; ++i) {
                    gvalue1 += recon_one[i]*myVolume[myVxCount*myVyCount*idx + i]*
                      exp(-K[i])*-1.0;
                    gvalue2 += recon_one[i]*
                      V2[myVxCount*myVyCount*idx + i]*exp(-K[i])*-1.0;
                    gvalue3 += recon_one[i]*
                      V3[myVxCount*myVyCount*idx + i]*exp(-K[i])*-1.0;
                    gvalue4 += recon_one[i]*
                      V4[myVxCount*myVyCount*idx + i]*exp(-K[i])*-1.0;

                    hvalue1 += recon_one[i]*pow(
                        myVolume[myVxCount*myVyCount*idx + i], 2)*exp(-K[i]);
                    hvalue2 += recon_one[i]* myVolume[
                        myVxCount*myVyCount*idx + i]*V2[myVxCount*myVyCount*idx + i]*
                        exp(-K[i]);
                    hvalue3 += recon_one[i]* myVolume[
                        myVxCount*myVyCount*idx + i]*V3[myVxCount*myVyCount*idx + i]*
                        exp(-K[i]);
                    hvalue4 += recon_one[i]* myVolume[
                        myVxCount*myVyCount*idx + i]*V4[myVxCount*myVyCount*idx + i]*
                        exp(-K[i]);
                    hvalue5 += recon_one[i]*pow(
                        V2[myVxCount*myVyCount*idx + i],2)*exp(-K[i]);
                    hvalue6 += recon_one[i]*V2[
                        myVxCount*myVyCount*idx + i]*V3[myVxCount*myVyCount*idx + i]*
                        exp(-K[i]);
                    hvalue7 += recon_one[i]*V2[
                        myVxCount*myVyCount*idx + i]*V4[myVxCount*myVyCount*idx + i]*
                        exp(-K[i]);
                    hvalue8 += recon_one[i]*pow(
                        V3[myVxCount*myVyCount*idx + i],2)*exp(-K[i]);
                    hvalue9 += recon_one[i]*V3[
                        myVxCount*myVyCount*idx + i]*V4[myVxCount*myVyCount*idx + i]*
                        exp(-K[i]);
                    hvalue10 += recon_one[i]*pow(
                        V4[myVxCount*myVyCount*idx + i],2)*exp(-K[i]);
                }
                gradients[0] = gvalue1;
                gradients[1] = gvalue2;
                gradients[2] = gvalue3;
                gradients[3] = gvalue4;
                if (my_rank == 0) {
                    std::cout << "myrank 0 idx " << idx << std::endl;
                    std::cout << "torch grad " << myGradients[idx*4] << " c++ grad " << gradients[0] << std::endl;
                    std::cout << "torch grad " << myGradients[idx*4+1] << " c++ grad " << gradients[1] << std::endl;
                    std::cout << "torch grad " << myGradients[idx*4+2] << " c++ grad " << gradients[2] << std::endl;
                    std::cout << "torch grad " << myGradients[idx*4+3] << " c++ grad " << gradients[3] << std::endl;
                    std::cout << "torch recon_one[0]" << reconData[0] << " c++ recon_one[0] " << recon_one[0] << std::endl;
                }
                hessians[0][0] = hvalue1;
                hessians[0][1] = hvalue2;
                hessians[0][2] = hvalue3;
                hessians[0][3] = hvalue4;
                hessians[1][0] = hvalue2;
                hessians[1][1] = hvalue5;
                hessians[1][2] = hvalue6;
                hessians[1][3] = hvalue7;
                hessians[2][0] = hvalue3;
                hessians[2][1] = hvalue6;
                hessians[2][2] = hvalue8;
                hessians[2][3] = hvalue9;
                hessians[3][0] = hvalue4;
                hessians[3][1] = hvalue7;
                hessians[3][2] = hvalue9;
                hessians[3][3] = hvalue10;
                // compute lambdas
                int order = 4;
                int k;
                double d = determinant(hessians, order);
                if (d == 0) {
                    printf ("Need to define pesudoinverse for matrix in node %d\n", idx);
                    printf ("%5.3g %5.3g %5.3g %5.3g\n", hessians[0][0],
                        hessians[0][1], hessians[0][2], hessians[0][3]);
                    printf ("%5.3g %5.3g %5.3g %5.3g\n", hessians[1][0],
                        hessians[1][1], hessians[1][2], hessians[1][3]);
                    printf ("%5.3g %5.3g %5.3g %5.3g\n", hessians[2][0],
                        hessians[2][1], hessians[2][2], hessians[2][3]);
                    printf ("%5.3g %5.3g %5.3g %5.3g\n", hessians[3][0],
                        hessians[3][1], hessians[3][2], hessians[3][3]);
                    break;
                }
                else{
                    printf ("%5.3g %5.3g %5.3g %5.3g\n", hessians[0][0],
                        hessians[0][1], hessians[0][2], hessians[0][3]);
                    printf ("%5.3g %5.3g %5.3g %5.3g\n", hessians[1][0],
                        hessians[1][1], hessians[1][2], hessians[1][3]);
                    printf ("%5.3g %5.3g %5.3g %5.3g\n", hessians[2][0],
                        hessians[2][1], hessians[2][2], hessians[2][3]);
                    printf ("%5.3g %5.3g %5.3g %5.3g\n", hessians[3][0],
                        hessians[3][1], hessians[3][2], hessians[3][3]);
                    double** inverse = cofactor(hessians, order);
                    double matmul[4] = {0, 0, 0, 0};
                    for (i=0; i<4; ++i) {
                        matmul[i] = 0;
                        for (k=0; k<4; ++k) {
                            matmul[i] += inverse[i][k] * gradients[k];
                        }
                    }
                    lambdas[0] = lambdas[0] - matmul[0];
                    lambdas[1] = lambdas[1] - matmul[1];
                    lambdas[2] = lambdas[2] - matmul[2];
                    lambdas[3] = lambdas[3] - matmul[3];
                }
                count = count + 1;
            }
            if (my_rank == 0) {
                std::cout << "myrank 0 idx " << idx << std::endl;
                std::cout << "torch Lag " << myLagrangesTorch[idx*4] << " c++ Lag " << lambdas[0] << std::endl;
                std::cout << "torch Lag " << myLagrangesTorch[idx*4+1] << " c++ Lag " << lambdas[1] << std::endl;
                std::cout << "torch Lag " << myLagrangesTorch[idx*4+2] << " c++ Lag " << lambdas[2] << std::endl;
                std::cout << "torch Lag " << myLagrangesTorch[idx*4+3] << " c++ Lag " << lambdas[3] << std::endl;
            }
        }
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (my_rank == 0) {
        printf ("%d Time Taken for Lagrange Computations: %f %f\n", mySpecies, end-start, end-start1);
    }
#if 0
    double* breg_recon = new double[myLocalElements];
    memset(breg_recon, 0, myLocalElements*sizeof(double));
    double* new_recon = breg_recon;
    if (applyPQ) {
        double nK[myVxCount*myVyCount];
        myLagrangeIndexesDensity = new int[myPlaneCount*myNodeCount];
        myLagrangeIndexesUpara = new int[myPlaneCount*myNodeCount];
        myLagrangeIndexesTperp = new int[myPlaneCount*myNodeCount];
        myLagrangeIndexesRpara = new int[myPlaneCount*myNodeCount];
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        if (useKMeansMPI == 0) {
            quantizeLagranges(0, myLagrangeIndexesDensity, myDensityTable);
            quantizeLagranges(1, myLagrangeIndexesUpara, myUparaTable);
            quantizeLagranges(2, myLagrangeIndexesTperp, myTperpTable);
            quantizeLagranges(3, myLagrangeIndexesRpara, myRparaTable);
        }
        else {
            quantizeLagrangesMPI(0, myLagrangeIndexesDensity, myDensityTable);
            quantizeLagrangesMPI(1, myLagrangeIndexesUpara, myUparaTable);
            quantizeLagrangesMPI(2, myLagrangeIndexesTperp, myTperpTable);
            quantizeLagrangesMPI(3, myLagrangeIndexesRpara, myRparaTable);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();
        if (my_rank == 0) {
            printf ("Time Taken for Quantization: %f\n", end-start);
        }
        for (iphi=0; iphi<myPlaneCount; ++iphi) {
            for (idx = 0; idx<myNodeCount; ++idx) {
                const double* recon_one = &reconData[myNodeCount*myVxCount*
                      myVyCount*iphi + myVxCount*myVyCount*idx];
                double* new_recon_one = &new_recon[myNodeCount*myVxCount*
                      myVyCount*iphi + myVxCount*myVyCount*idx];
                int x = 4*idx;
                int m1 = myLagrangeIndexesDensity[iphi*myNodeCount + idx];
                int m2 = myLagrangeIndexesUpara[iphi*myNodeCount + idx];
                int m3 = myLagrangeIndexesTperp[iphi*myNodeCount + idx];
                int m4 = myLagrangeIndexesRpara[iphi*myNodeCount + idx];
                double c1 = myDensityTable[m1];
                double c2 = myUparaTable[m2];
                double c3 = myTperpTable[m3];
                double c4 = myRparaTable[m4];
                for (i=0; i<myVxCount * myVyCount; ++i) {
                    nK[i] = (c1)*myVolume[myVxCount*myVyCount*idx+i]+
                           (c2)*V2[myVxCount*myVyCount*idx+i] +
                           (c3)*V3[myVxCount*myVyCount*idx+i] +
                           (c4)*V4[myVxCount*myVyCount*idx+i];
                    new_recon_one[i] = recon_one[i] * exp(-nK[i]);
                }
            }
        }
    }
    else {
      const char* precision = "double";
        double nK[myVxCount*myVyCount];
        for (iphi=0; iphi<myPlaneCount; ++iphi) {
            for (idx = 0; idx<myNodeCount; ++idx) {
                const double* recon_one = &reconData[myNodeCount*myVxCount*
                      myVyCount*iphi + myVxCount*myVyCount*idx];
                double* new_recon_one = &new_recon[myNodeCount*myVxCount*
                      myVyCount*iphi + myVxCount*myVyCount*idx];
                int x = 4*idx;
                for (i=0; i<myVxCount * myVyCount; ++i) {
                    if (!strcmp(precision, "float"))
                    {
                        nK[i] = float(myLagranges[x])*myVolume[myVxCount*myVyCount*idx+i]+
                           float(myLagranges[x+1])*V2[myVxCount*myVyCount*idx+i] +
                           float(myLagranges[x+2])*V3[myVxCount*myVyCount*idx+i] +
                           float(myLagranges[x+3])*V4[myVxCount*myVyCount*idx+i];
                    }
                    else if (!strcmp(precision, "double"))
                    {
                        nK[i] = (myLagranges[x])*myVolume[myVxCount*myVyCount*idx+i]+
                           (myLagranges[x+1])*V2[myVxCount*myVyCount*idx+i] +
                           (myLagranges[x+2])*V3[myVxCount*myVyCount*idx+i] +
                           (myLagranges[x+3])*V4[myVxCount*myVyCount*idx+i];
                    }
                    new_recon_one[i] = recon_one[i] * exp(-nK[i]);
                }
            }
        }
    }
    compareQoIs(reconData, new_recon);
#else
    std::cout << "Recon data size " << new_recon_vec.size() << std::endl;
    double* new_recon = new_recon_vec.data();
    for (int kk = 0; kk < 100; ++kk) {
        std::cout << "recon " << reconData[kk] << " recon pp " << new_recon[kk] << std::endl;
    }
    compareQoIs(reconData, new_recon);
#endif
    return;
}

void LagrangeTorch::initializeClusterCenters(double* &clusters, double* lagarray, int numObjs)
{
    clusters = new double[myNumClusters];
    assert(clusters != NULL);

    srand(time(NULL));
    double* myNumbers = new double[myNumClusters];
    std::map <int, int> mymap;
    for (int i=0; i<myNumClusters; ++i) {
        int index = rand() % numObjs;
        while (mymap.find(index) != mymap.end()) {
            index = rand() % numObjs;
        }
        clusters[i] = lagarray[index];
        mymap[index] = i;
    }
}

void LagrangeTorch::quantizeLagranges(int offset, int* &membership, double* &clusters)
{
    int numObjs = myPlaneCount*myNodeCount;
    float threshold = 0.0001;
    double* lagarray = new double[myNodeCount];
    for (int iphi = 0; iphi<myPlaneCount; ++iphi) {
        for (int idx = 0; idx<myNodeCount; ++idx) {
            lagarray[iphi*myNodeCount + idx] = myLagranges[iphi*myNodeCount + 4*idx + offset];
        }
    }

    initializeClusterCenters(clusters, lagarray, numObjs);
    membership = new int [numObjs];
    memset (membership, 0, numObjs*sizeof(int));
    kmeans(lagarray, numObjs, myNumClusters, threshold, membership, clusters);
    return;
}

void LagrangeTorch::initializeClusterCentersMPI(double* &clusters, int numP, int myRank, double* lagarray, int numObjs)
{
    clusters = new double[myNumClusters];
    assert(clusters != NULL);
    int* counts = new int[numP];
    int* disps = new int[numP];

    int pertask = myNumClusters/numP;
    for (int i=0; i<numP-1; i++) {
        counts[i] = pertask;
    }
    counts[numP-1] = myNumClusters - pertask*(numP-1);

    disps[0] = 0;
    for (int i=1; i<numP; i++) {
        disps[i] = disps[i-1] + counts[i-1];
    }

    srand(time(NULL));
    int myNumClusters = counts[myRank];
    double* myNumbers = new double[myNumClusters];
    std::map <int, int> mymap;
    for (int i=0; i<myNumClusters; ++i) {
        int index = rand() % numObjs;
        while (mymap.find(index) != mymap.end()) {
            index = rand() % numObjs;
        }
        myNumbers[i] = lagarray[index];
        mymap[index] = i;
    }
    MPI_Allgatherv(myNumbers, myNumClusters, MPI_DOUBLE, clusters, counts, disps, MPI_DOUBLE, MPI_COMM_WORLD);
}

void LagrangeTorch::quantizeLagrangesMPI(int offset, int* &membership, double* &clusters)
{
    int numObjs = myPlaneCount*myNodeCount;
    float threshold = 0.01;
    int num_procs;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double* lagarray = new double[myNodeCount];
    for (int iphi = 0; iphi<myPlaneCount; ++iphi) {
        for (int idx = 0; idx<myNodeCount; ++idx) {
            lagarray[iphi*myNodeCount + idx] = myLagranges[iphi*myNodeCount + 4*idx + offset];
        }
    }

    initializeClusterCentersMPI(clusters, num_procs, my_rank, lagarray, numObjs);
    membership = new int [numObjs];
    memset (membership, 0, numObjs*sizeof(int));
    mpi_kmeans(lagarray, numObjs, myNumClusters,
        threshold, membership, clusters);
    return;
}

// Get the number of bytes needed to store the PQ table
size_t LagrangeTorch::getTableSize()
{
    return 0;
}

size_t LagrangeTorch::putLagrangeParameters(char* &bufferOut, size_t &bufferOutOffset)
{
#if 0
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
#else
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
#endif
}

size_t LagrangeTorch::putResultV2(char* &bufferOut, size_t &bufferOutOffset)
{
    // TODO: after your algorithm is done, put the result into
    // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your       first
    // double number *reinterpret_cast<double*>(bufferOut+bufferOutOff      set+8)
    // for your second double number and so on
    int intbytes = putLagrangeParameters(bufferOut, bufferOutOffset);
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

size_t LagrangeTorch::putPQIndexes(char* &bufferOut, size_t &bufferOutOffset)
{
    int i, intcount = 0, count = 0, singlecount = 0, bufferbytes = 0;
    int numObjs = myPlaneCount*myNodeCount;
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<int*>(
              bufferOut+bufferOutOffset+(intcount++)*sizeof(uint8_t)) =
                  uint8_t(myLagrangeIndexesDensity[i]);
    }
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<int*>(
              bufferOut+bufferOutOffset+(intcount++)*sizeof(uint8_t)) =
                  uint8_t(myLagrangeIndexesUpara[i]);
    }
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<int*>(
              bufferOut+bufferOutOffset+(intcount++)*sizeof(uint8_t)) =
                  uint8_t(myLagrangeIndexesTperp[i]);
    }
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<int*>(
              bufferOut+bufferOutOffset+(intcount++)*sizeof(uint8_t)) =
                  uint8_t(myLagrangeIndexesRpara[i]);
    }
    bufferbytes = intcount * sizeof(uint8_t);
    if (useKMeansMPI == 0) {
        singlecount = 1;
        int intbytes = intcount * sizeof(uint8_t);
        *reinterpret_cast<int*>(
              bufferOut+bufferOutOffset+intbytes) = myNumClusters;
        intbytes += singlecount * sizeof(int);
        for (i=0; i<myNumClusters; ++i) {
            *reinterpret_cast<int*>(bufferOut+bufferOutOffset+
                intbytes+(count++)*sizeof(double)) =
                  myDensityTable[i];
        }
        for (i=0; i<myNumClusters; ++i) {
            *reinterpret_cast<int*>(bufferOut+bufferOutOffset+
                intbytes+(count++)*sizeof(double)) =
                  myUparaTable[i];
        }
        for (i=0; i<myNumClusters; ++i) {
            *reinterpret_cast<int*>(bufferOut+bufferOutOffset+
                intbytes+(count++)*sizeof(double)) =
                  myTperpTable[i];
        }
        for (i=0; i<myNumClusters; ++i) {
            *reinterpret_cast<int*>(bufferOut+bufferOutOffset+
                intbytes+(count++)*sizeof(double)) =
                  myRparaTable[i];
        }
        bufferbytes = intbytes + count * sizeof(double);
    }
    return bufferbytes;
}

size_t LagrangeTorch::getPQIndexes(const char* bufferIn)
{
    int i, intcount = 0;
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesDensity[i] = *(reinterpret_cast<const uint8_t*>(bufferIn+(intcount++)*sizeof(uint8_t)));
    }
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesUpara[i] = (*reinterpret_cast<const uint8_t*>(bufferIn+(intcount++)*sizeof(uint8_t)));
    }
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesTperp[i] = (*reinterpret_cast<const uint8_t*>(bufferIn+(intcount++)*sizeof(uint8_t)));
    }
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesRpara[i] = (*reinterpret_cast<const uint8_t*>(bufferIn+(intcount++)*sizeof(uint8_t)));
    }
    return intcount*sizeof(uint8_t);
}

size_t LagrangeTorch::putResultV1(char* &bufferOut, size_t &bufferOutOffset)
{
    // TODO: after your algorithm is done, put the result into
    // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your       first
    // double number *reinterpret_cast<double*>(bufferOut+bufferOutOff      set+8)
    // for your second double number and so on
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int intbytes = putPQIndexes(bufferOut, bufferOutOffset);
    // printf ("Rank %d numObjs %d numBytes %d\n", my_rank, myPlaneCount*myNodeCount, intbytes);
    if (my_rank == 0 && useKMeansMPI == 1) {
        FILE* fp = fopen("PqMeshInfo.bin", "wb");
        // write out the PQ table and the mesh parameters
        fwrite(&myNumClusters, sizeof(int), 1, fp);
        fwrite(myDensityTable, sizeof(double), myNumClusters, fp);
        fwrite(myUparaTable, sizeof(double), myNumClusters, fp);
        fwrite(myTperpTable, sizeof(double), myNumClusters, fp);
        fwrite(myRparaTable, sizeof(double), myNumClusters, fp);
        int str_length = myMeshFile.length();
        printf("Mesh file %d %s\n", str_length, myMeshFile.c_str());
        fwrite(&str_length, sizeof(int), 1, fp);
        fwrite(myMeshFile.c_str(), sizeof(char), str_length, fp);
        fclose(fp);
    }
    return intbytes;
}

char* LagrangeTorch::setDataFromCharBufferV1(double* &reconData,
    const char* bufferIn, size_t sizeOut)
{
    size_t bufferOffset = getPQIndexes(bufferIn);
    FILE* fp = fopen("PqMeshInfo.bin", "rb");
    fread(&myNumClusters, sizeof(int), 1, fp);
    fread(myDensityTable, sizeof(double), myNumClusters, fp);
    fread(myUparaTable, sizeof(double), myNumClusters, fp);
    fread(myTperpTable, sizeof(double), myNumClusters, fp);
    fread(myRparaTable, sizeof(double), myNumClusters, fp);
    int str_length = 0;
    fread(&str_length, sizeof(int), 1, fp);
    char meshFile[str_length];
    fread(meshFile, sizeof(char), str_length, fp);
    fclose(fp);
    readF0Params(std::string(meshFile, 0, str_length));
    std::vector <double> V2 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V3 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V4 (myNodeCount*myVxCount*myVyCount, 0);
    double nK[myVxCount*myVyCount];
    int i, j, k, l, m;
    myLocalElements = myNodeCount*myPlaneCount*myVxCount*myVyCount;
    for (i=0; i<myLocalElements; ++i) {
        if (!(reconData[i] > 0)) {
            ((double*)reconData)[i] = myEpsilon;
        }
    }
    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int (k%myVxCount);
        l = int(k%(myVxCount*myVyCount));
        m = int(l/myVyCount);
        V2[k] = myVolume[k] * myVth[i] * myVp[j];
        V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
        V4[k] = myVolume[k] * pow(myVp[j],2) * myVth2[i] * myParticleMass;
    }
    int iphi, idx;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        for (idx = 0; idx<myNodeCount; ++idx) {
            const double* recon_one = &reconData[myNodeCount*myVxCount*
                  myVyCount*iphi + myVxCount*myVyCount*idx];
            int m1 = myLagrangeIndexesDensity[iphi*myNodeCount + idx];
            int m2 = myLagrangeIndexesUpara[iphi*myNodeCount + idx];
            int m3 = myLagrangeIndexesTperp[iphi*myNodeCount + idx];
            int m4 = myLagrangeIndexesRpara[iphi*myNodeCount + idx];
            double c1 = myDensityTable[m1];
            double c2 = myUparaTable[m2];
            double c3 = myTperpTable[m3];
            double c4 = myRparaTable[m4];
            for (i=0; i<myVxCount * myVyCount; ++i) {
                nK[i] = (c1)*myVolume[myVxCount*myVyCount*idx+i]+
                       (c2)*V2[myVxCount*myVyCount*idx+i] +
                       (c3)*V3[myVxCount*myVyCount*idx+i] +
                       (c4)*V4[myVxCount*myVyCount*idx+i];
                ((double*)recon_one)[i] = recon_one[i] * exp(-nK[i]);
            }
        }
    }
    return reinterpret_cast<char*>(reconData);
}

size_t LagrangeTorch::putResult(char* &bufferOut, size_t &bufferOutOffset)
{
    // TODO: after your algorithm is done, put the result into
    // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your       first
    // double number *reinterpret_cast<double*>(bufferOut+bufferOutOff      set+8)
    // for your second double number and so on
    int i, intcount = 0, count = 0;
    int numObjs = myPlaneCount*myNodeCount;
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<int*>(
              bufferOut+bufferOutOffset+(intcount++)*sizeof(int)) =
                  myLagrangeIndexesDensity[i];
    }
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<int*>(
              bufferOut+bufferOutOffset+(intcount++)*sizeof(int)) =
                  myLagrangeIndexesUpara[i];
    }
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<int*>(
              bufferOut+bufferOutOffset+(intcount++)*sizeof(int)) =
                  myLagrangeIndexesTperp[i];
    }
    for (i=0; i<numObjs; ++i) {
        *reinterpret_cast<int*>(
              bufferOut+bufferOutOffset+(intcount++)*sizeof(int)) =
                  myLagrangeIndexesRpara[i];
    }
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        FILE* fp = fopen("PqMeshInfo.bin", "wb");
        // write out the PQ table and the mesh parameters
        fwrite(&myNumClusters, sizeof(int), 1, fp);
        fwrite(myDensityTable, sizeof(double), myNumClusters, fp);
        fwrite(myUparaTable, sizeof(double), myNumClusters, fp);
        fwrite(myTperpTable, sizeof(double), myNumClusters, fp);
        fwrite(myRparaTable, sizeof(double), myNumClusters, fp);
        fwrite(&myNodeCount, sizeof(int), 1, fp);
        int indexOffset = mySpecies==1 ? myNodeCount : 0;
        fwrite(myGridVolume.data()+indexOffset, sizeof(double), myNodeCount, fp);
        fwrite(myF0TEv.data()+indexOffset, sizeof(double), myNodeCount, fp);
        fwrite(&myF0Dvp, sizeof(double), 1, fp);
        fwrite(&myF0Dsmu, sizeof(double), 1, fp);
        fwrite(&myF0Nvp, sizeof(int), 1, fp);
        fwrite(&myF0Nmu, sizeof(int), 1, fp);
        fclose(fp);
#if 0
        // for (i=0; i<myNumClusters; ++i) {
            //*reinterpret_cast<double*>(
               // bufferOut+bufferOutOffset+(count++)*sizeof(double)) =
               // myDensityTable[i];
            // fwrite(myDensityTable[i], sizeof(double), 1, fp);
        // }
        double* gridVolume = new double[myNodeCount];
        double* f0tev = new double[myNodeCount];
        int elements = 0;
        i = 0;
        for (double d : myGridVolume) {
            if (elements < myNodeCount) {
                elements++;
                continue;
            }
            // *reinterpret_cast<double*>(
                  // bufferOut+bufferOutOffset+(count++)*sizeof(double)) = d;
            gridVolume[i++] = d;
        }
        // Access f0_t_ev with an offset of nodes to get to the electrons
        elements = 0;
        for (double d : myF0TEv) {
            if (elements < myNodeCount) {
                elements++;
                continue;
            }
            *reinterpret_cast<double*>(
                  bufferOut+bufferOutOffset+(count++)*sizeof(double)) = d;
        }
        *reinterpret_cast<double*>(
            bufferOut+bufferOutOffset+(count++)*sizeof(double)) = myF0Dvp[0];
        *reinterpret_cast<double*>(
            bufferOut+bufferOutOffset+(count++)*sizeof(double)) = myF0Dsmu[0];

        int offset = count*sizeof(double) + intcount*sizeof(int);
        *reinterpret_cast<int*>(
            bufferOut+bufferOutOffset+offset) = myF0Nvp[0];
        *reinterpret_cast<int*>(
            bufferOut+bufferOutOffset+offset+sizeof(int)) = myF0Nmu[0];
        intcount += 2;
#endif
    }
    return intcount*sizeof(int);
}

void LagrangeTorch::setDataFromCharBuffer(double* &reconData,
    const char* bufferIn, size_t bufferTotalSize)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int i, j, k, l, m, intcount = 0, doublecount = 0;
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesDensity[i] = *(reinterpret_cast<const int*>(bufferIn+(intcount++)*sizeof(int)));
    }
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesUpara[i] = (*reinterpret_cast<const int*>(bufferIn+(intcount++)*sizeof(int)));
    }
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesTperp[i] = (*reinterpret_cast<const int*>(bufferIn+(intcount++)*sizeof(int)));
    }
    for (i=0; i<myNodeCount; ++i) {
        myLagrangeIndexesRpara[i] = (*reinterpret_cast<const int*>(bufferIn+(intcount++)*sizeof(int)));
    }
    FILE* fp = fopen("PqMeshInfo.bin", "rb");
    fread(&myNumClusters, sizeof(int), 1, fp);
    fread(myDensityTable, sizeof(double), myNumClusters, fp);
    fread(myUparaTable, sizeof(double), myNumClusters, fp);
    fread(myTperpTable, sizeof(double), myNumClusters, fp);
    fread(myRparaTable, sizeof(double), myNumClusters, fp);
    fread(&myNodeCount, sizeof(int), 1, fp);
    double* gridVolume = new double[myNodeCount];
    double* f0TEv = new double[myNodeCount];
    fread(gridVolume, sizeof(double), myNodeCount, fp);
    fread(f0TEv, sizeof(double), myNodeCount, fp);
    fread(&myF0Dvp, sizeof(double), 1, fp);
    fread(&myF0Dsmu, sizeof(double), 1, fp);
    fread(&myF0Nvp, sizeof(int), 1, fp);
    fread(&myF0Nmu, sizeof(int), 1, fp);
    fclose(fp);
    for (i=0; i<myNodeCount; ++i) {
        myGridVolume.push_back(0.0);
        myF0TEv.push_back(0.0);
    }
    for (i=0; i<myNodeCount; ++i) {
        myGridVolume.push_back(gridVolume[i]);
        myF0TEv.push_back(f0TEv[i]);
    }
#if 0
    double* gridVolume = new double[myNodeCount];
    double* f0TEv = new double[myNodeCount];
    int* nvp = new int [1];
    int* nmu = new int [1];
    double* dvp = new double[1];
    double* dsmu = new double[1];
    int bufferOffset = intcount*sizeof(int);
    int bcast_rank = 0;
    int local_rank = 0;
    // if ((bufferTotalSize-bufferOffset) > 0) {
        for (i=0; i<myNumClusters; ++i) {
            myDensityTable[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
        // local_rank = my_rank;
    // }
    // MPI_Allreduce(&local_rank, &bcast_rank, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Bcast(myDensityTable, myNumClusters, MPI_DOUBLE, bcast_rank, MPI_COMM_WORLD);
        for (i=0; i<myNumClusters; ++i) {
            myUparaTable[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
        for (i=0; i<myNumClusters; ++i) {
            myTperpTable[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
        for (i=0; i<myNumClusters; ++i) {
            myRparaTable[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
    // printf ("My rank %d density %5.3g upara %5.3g tperp %5.3g rpara %5.3g\n", my_rank, myDensityTable[0], myUparaTable[0], myTperpTable[0], myRparaTable[0]);
        bufferOffset += doublecount*sizeof(double);
        doublecount = 0;
        for (i=0; i<myNodeCount; ++i) {
            gridVolume[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
        bufferOffset += i*sizeof(double);
        for (i=0; i<myNodeCount; ++i) {
            f0TEv[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+(doublecount++)*sizeof(double)));
        }
        bufferOffset += doublecount*sizeof(double);
        dvp[0] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset));
        dsmu[0] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+sizeof(double)));
        nvp[0] = (*reinterpret_cast<const int*>(bufferIn+bufferOffset+2*sizeof(double)));
        nmu[0] = (*reinterpret_cast<const int*>(bufferIn+bufferOffset+2*sizeof(double)+sizeof(int)));
    MPI_Bcast(myDensityTable, myNumClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(myUparaTable, myNumClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(myTperpTable, myNumClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(myRparaTable, myNumClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gridVolume, myNodeCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    myGridVolume.insert(myGridVolume.begin(), gridVolume, gridVolume+myNodeCount);
    MPI_Bcast(f0TEv, myNodeCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    myF0TEv.insert(myF0TEv.begin(), f0TEv, f0TEv+myNodeCount);
    MPI_Bcast(dvp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    myF0Dvp.push_back(dvp[0]);
    MPI_Bcast(dsmu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    myF0Dsmu.push_back(dsmu[0]);
    MPI_Bcast(nvp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    myF0Nvp.push_back(nvp[0]);
    MPI_Bcast(nmu, 1, MPI_INT, 0, MPI_COMM_WORLD);
    myF0Nmu.push_back(nmu[0]);

    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    std::vector <double> V2 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V3 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V4 (myNodeCount*myVxCount*myVyCount, 0);
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int (k%myVxCount);
        V2[k] = myVolume[k] * myVth[i] * myVp[j];
        V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
        V4[k] = myVolume[k] * pow(myVp[j],2) * myVth2[i] * myParticleMass;
    }
    double K[myVxCount*myVyCount];
    int iphi, idx;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        for (idx = 0; idx<myNodeCount; ++idx) {
            double* recon_one = &reconData[myNodeCount*myVxCount*
                  myVyCount*iphi + myVxCount*myVyCount*idx];
            int x = 4*idx;
            for (i=0; i<myVxCount * myVyCount; ++i) {
                K[i] = myDensityTable[myLagrangeIndexesDensity[idx]]*myVolume[myVxCount*myVyCount*idx+i]+
                       myUparaTable[myLagrangeIndexesUpara[idx]]*V2[myVxCount*myVyCount*idx+i] +
                       myTperpTable[myLagrangeIndexesTperp[idx]]*V3[myVxCount*myVyCount*idx+i] +
                       myRparaTable[myLagrangeIndexesRpara[idx]]*V4[myVxCount*myVyCount*idx+i];
                recon_one[i] = recon_one[i] * exp(-K[i]);
            }
        }
    }
#endif
}

size_t LagrangeTorch::putResultNoPQ(char* &bufferOut, size_t &bufferOutOffset)
{
    // TODO: after your algorithm is done, put the result into
    // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your       first
    // double number *reinterpret_cast<double*>(bufferOut+bufferOutOff      set+8)
    // for your second double number and so on

    int i, count = 0;
    for (i=0; i<4*myNodeCount; ++i) {
          *reinterpret_cast<double*>(
              bufferOut+bufferOutOffset+(count++)*sizeof(double)) =
                  myLagranges[i];
    }
    int lagrangeCount = count;
    // Access grid_vol with an offset of nodes to get to the electrons
    int elements = 0;
    for (double d : myGridVolume) {
        if (elements < myNodeCount) {
            elements++;
            continue;
        }
        *reinterpret_cast<double*>(
              bufferOut+bufferOutOffset+(count++)*sizeof(double)) = d;
#ifdef UF_DEBUG
        if (count == lagrangeCount+myNodeCount) {
             printf("Grid vol element %f\n", d);
        }
#endif
    }
    // Access f0_t_ev with an offset of nodes to get to the electrons
    elements = 0;
    for (double d : myF0TEv) {
        if (elements < myNodeCount) {
            elements++;
            continue;
        }
        *reinterpret_cast<double*>(
              bufferOut+bufferOutOffset+(count++)*sizeof(double)) = d;
    }
    *reinterpret_cast<double*>(
        bufferOut+bufferOutOffset+(count++)*sizeof(double)) = myF0Dvp[0];
    *reinterpret_cast<double*>(
        bufferOut+bufferOutOffset+(count++)*sizeof(double)) = myF0Dsmu[0];
    int offset = count*sizeof(double);
    *reinterpret_cast<int*>(
        bufferOut+bufferOutOffset+offset) = myF0Nvp[0];
    *reinterpret_cast<int*>(
        bufferOut+bufferOutOffset+offset+sizeof(int)) = myF0Nmu[0];
    return count*sizeof(double) + 2*sizeof(int);
}

void LagrangeTorch::setDataFromCharBuffer2(double* &reconData,
    const char* bufferIn, size_t bufferOffset, size_t totalSize)
{
    size_t bufferSize = totalSize - bufferOffset;
    int i,j,k,l,m;
    // Assuming the Lagrange parameters are stored as double numbers
    // This will change as we add quantization
    // Find node size
    myNodeCount = int((bufferSize-2*sizeof(double)-2*sizeof(int))/(6*sizeof(double)));
    int numLagrangeParameters = myNodeCount*4;
    for (i=0; i<numLagrangeParameters; ++i) {
        myLagranges[i] = (*reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
    }
    bufferOffset += i*sizeof(double);
    for (i=0; i<myNodeCount; ++i) {
        myGridVolume.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
    }
    bufferOffset += i*sizeof(double);
    for (i=0; i<myNodeCount; ++i) {
        myF0TEv.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
    }
    bufferOffset += i*sizeof(double);
    myF0Dvp.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset));
    myF0Dsmu.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset+sizeof(double)));
    myF0Nvp.push_back(*reinterpret_cast<const int*>(bufferIn+bufferOffset+2*sizeof(double)));
    myF0Nmu.push_back(*reinterpret_cast<const int*>(bufferIn+bufferOffset+2*sizeof(double)+sizeof(int)));
    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    std::vector <double> V2 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V3 (myNodeCount*myVxCount*myVyCount, 0);
    std::vector <double> V4 (myNodeCount*myVxCount*myVyCount, 0);
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int (k%myVxCount);
        V2[k] = myVolume[k] * myVth[i] * myVp[j];
        V3[k] = myVolume[k] * 0.5 * myMuQoi[m] * myVth2[i] * myParticleMass;
        V4[k] = myVolume[k] * pow(myVp[j],2) * myVth2[i] * myParticleMass;
    }
    double K[myVxCount*myVyCount];
    int iphi, idx;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        for (idx = 0; idx<myNodeCount; ++idx) {
            double* recon_one = &reconData[myNodeCount*myVxCount*
                  myVyCount*iphi + myVxCount*myVyCount*idx];
            int x = 4*idx;
            for (i=0; i<myVxCount * myVyCount; ++i) {
                K[i] = myLagranges[x]*myVolume[myVxCount*myVyCount*idx+i]+
                       myLagranges[x+1]*V2[myVxCount*myVyCount*idx+i] +
                       myLagranges[x+2]*V3[myVxCount*myVyCount*idx+i] +
                       myLagranges[x+3]*V4[myVxCount*myVyCount*idx+i];
                recon_one[i] = recon_one[i] * exp(-K[i]);
            }
        }
    }
}

#if 0
void LagrangeTorch::readCharBuffer(const char* bufferIn, size_t bufferOffset, size_t bufferSize)
{
    int i;
    // Assuming the Lagrange parameters are stored as double numbers
    // This will change as we add quantization
    // Find node size
    myNodeCount = int((bufferSize-2*sizeof(double)-2*sizeof(int))/(6*sizeof(double)));
    int numLagrangeParameters = myNodeCount*4;
    std::vector <double> lagranges;
    std::vector <double> gridVol;
    std::vector <double> f0TEv;
    double nvp, dvp, nmu, dsmu;
    for (i=0; i<numLagrangeParameters; ++i) {
        lagranges.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
    }
    bufferOffset += i*sizeof(double);
    for (i=0; i<myNodeCount; ++i) {
        gridVol.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
        if (i == myNodeCount-1) {
             printf("Grid vol element %f\n", *reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
        }
    }
    bufferOffset += i*sizeof(double);
    for (i=0; i<myNodeCount; ++i) {
        f0TEv.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
    }
    bufferOffset += i*sizeof(double);
    dvp = *reinterpret_cast<const double*>(bufferIn+bufferOffset);
    dsmu = *reinterpret_cast<const double*>(bufferIn+bufferOffset+sizeof(double));
    nvp = *reinterpret_cast<const int*>(bufferIn+bufferOffset+2*sizeof(double));
    nmu = *reinterpret_cast<const int*>(bufferIn+bufferOffset+2*sizeof(double)+sizeof(int));
    double error = rmseError(myLagranges, lagranges);
    printf ("Lagrange error %f\n", error);
    error = rmseError2(myGridVolume, gridVol, myNodeCount);
    printf ("Grid volume error %f\n", error);
    error = rmseError2(myF0TEv, f0TEv, myNodeCount);
    printf ("f0_T_ev error %f\n", error);
    printf ("Nvp error %f\n", myF0Nvp[0] - nvp);
    printf ("Dvp error %f\n", myF0Dvp[0] - dvp);
    printf ("Nmu error %f\n", myF0Nmu[0] - nmu);
    printf ("Dsmu error %f\n", myF0Dsmu[0] - dsmu);
}
#endif

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
    auto datain = myVolumeTorch.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    myVolume = datain_vec;
    return;
}

void LagrangeTorch::setVp()
{
    myVpTorch = at::multiply(at::arange(-myF0Nvp[0], myF0Nvp[0]+1, ourGPUOptions), at::Scalar(myF0Dvp[0]));
    auto datain = myVpTorch.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    myVp = datain_vec;
    return;
}

void LagrangeTorch::setMuQoi()
{
    auto mu = at::multiply(at::arange(myF0Nmu[0]+1, ourGPUOptions), at::Scalar(myF0Dsmu[0]));
    myMuQoiTorch = at::pow(mu, at::Scalar(2));

    auto datain = myMuQoiTorch.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    myMuQoi = datain_vec;
    return;
}

void LagrangeTorch::setVth2()
{
    auto f0_T_ev_torch = myF0TEvTorch[mySpecies];
    myVth2Torch = at::multiply(f0_T_ev_torch, at::Scalar(mySmallElectronCharge/myParticleMass));
    myVthTorch = at::sqrt(myVth2Torch);

    auto datain = myVth2Torch.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    myVth2 = datain_vec;

    auto datain2 = myVthTorch.contiguous().cpu();
    std::vector<double> datain_vec2(datain2.data_ptr<double>(), datain2.data_ptr<double>() + datain2.numel());
    myVth = datain_vec2;
}

void LagrangeTorch::compute_C_qois(int iphi,
      std::vector <double> &density, std::vector <double> &upara,
      std::vector <double> &tperp, std::vector <double> &tpara,
      std::vector <double> &n0, std::vector <double> &t0,
      const double* dataIn)
{
    std::vector <double> den;
    std::vector <double> upar;
    std::vector <double> upar_;
    std::vector <double> tper;
    std::vector <double> en;
    std::vector <double> T_par;
    int i, j, k;
    std::cout << "data in" << myDataInTorch.sizes() << std::endl;
    auto f0_f_torch = myDataInTorch[iphi];
    auto datain = f0_f_torch.contiguous().cpu();
    std::vector<double> datain_vec(datain.data_ptr<double>(), datain.data_ptr<double>() + datain.numel());
    const double* f0_f_from_torch = datain_vec.data();
    const double* f0_f = &dataIn[iphi*myNodeCount*myVxCount*myVyCount];
#if 0
    std::cout << "f0_f data";
    for (i = 0; i<100; ++i) {
        std::cout << " " << f0_f[i] << " " << f0_f_from_torch[i];
    }
    std::cout << std::endl;
    for (i = 0; i<myNodeCount*myVxCount*myVyCount; ++i) {
        if (f0_f[i] != f0_f_from_torch[i]) {
            std::cout << "non-matching index " << i << " " << f0_f[i] << " " << f0_f_from_torch[i] << std::endl;
        }
    }
#endif

    int den_index = iphi*myNodeCount;

    for (i=0; i<myNodeCount*myVxCount*myVyCount; ++i) {
        den.push_back(f0_f[i] * myVolume[i]);
    }

    double value = 0;
    for (i=0; i<myNodeCount; ++i) {
        value = 0;
        for (j=0; j<myVxCount*myVyCount; ++j) {
            value += den[myVxCount*myVyCount*i + j];
        }
        density.push_back(value);
    }
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            for (k=0; k<myVyCount; ++k) {
                upar.push_back(f0_f[myVxCount*myVyCount*i + myVyCount*j + k] *
                    myVolume[myVxCount*myVyCount*i + myVyCount*j + k] *
                    myVth[i]*myVp[k]);
                }
    }
    for (i=0; i<myNodeCount; ++i) {
        double value = 0;
        for (j=0; j<myVxCount*myVyCount; ++j) {
            value += upar[myVxCount*myVyCount*i + j];
        }
        upara.push_back(value/density[den_index + i]);
    }
    for (i=0; i<myNodeCount; ++i) {
        upar_.push_back(upara[den_index + i]/myVth[i]);
    }
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            for (k=0; k<myVyCount; ++k) {
                tper.push_back(f0_f[myVxCount*myVyCount*i + myVyCount*j + k] *
                    myVolume[myVxCount*myVyCount*i + myVyCount*j + k] * 0.5 *
                    myMuQoi[j] * myVth2[i] * myParticleMass);
                }
    }
    for (i=0; i<myNodeCount; ++i) {
        double value = 0;
        for (j=0; j<myVxCount*myVyCount; ++j) {
            value += tper[myVxCount*myVyCount*i + j];
        }
        tperp.push_back(value/density[den_index + i]/mySmallElectronCharge);
        // printf ("Tperp %g, %g, %g, %g\n", value/density[den_index + i]/mySmallElectronCharge, value, myParticleMass, mySmallElectronCharge);
    }
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVyCount; ++j)
            en.push_back(0.5*pow((myVp[j]-upar_[i]),2));
    }
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            for (k=0; k<myVyCount; ++k)
                T_par.push_back(f0_f[myVxCount*myVyCount*i + myVyCount*j + k] *
                    myVolume[myVxCount*myVyCount*i + myVyCount*j + k] *
                    en[myVyCount*i+k] * myVth2[i] * myParticleMass);
    }
    for (i=0; i<myNodeCount; ++i) {
        double value = 0;
        for (j=0; j<myVxCount*myVyCount; ++j) {
            value += T_par[myVxCount*myVyCount*i + j];
        }
        tpara.push_back(2.0*value/density[den_index + i]/mySmallElectronCharge);
    }
    for (i=0; i<myNodeCount; ++i) {
        n0.push_back(density[i]);
        t0.push_back((2.0*tperp[i]+tpara[i])/3.0);
    }
    return;
}

#if 0
void LagrangeTorch::compute_C_qois(int iphi,
      std::vector <double> &density, std::vector <double> &upara,
      std::vector <double> &tperp, std::vector <double> &tpara,
      std::vector <double> &n0, std::vector <double> &t0,
      const double* dataIn)
{
    std::vector <double> den;
    std::vector <double> upar;
    std::vector <double> upar_;
    std::vector <double> tper;
    std::vector <double> en;
    std::vector <double> T_par;
    int i, j, k;
    const double* f0_f = &dataIn[iphi*myNodeCount*myVxCount*myVyCount];
    int den_index = iphi*myNodeCount;

    double value = 0;
    for (i=0; i<myNodeCount; ++i) {
        value = 0;
        for (k=0; k<myVxCount; ++k) {
            int offset = k*myNodeCount*myVyCount + i*myVyCount;
            for (j=offset; j < myVyCount+offset; j++) {
                value += f0_f[j] * myVolume[i*myVxCount*myVyCount + k*myVyCount + (j-offset)];
            }
        }
        density.push_back(value);
    }
    for (i=0; i<myNodeCount; ++i) {
        value = 0;
        for (k=0; k<myVxCount; ++k) {
            int offset = k*myNodeCount*myVyCount + i*myVyCount;
            for (j=offset; j < myVyCount+offset; j++) {
                value +=f0_f[j] * myVolume[i*myVxCount*myVyCount + k*myVyCount + (j-offset)] * myVth[i]*myVp[(j-offset)];
            }
        }
        upara.push_back(value/density[den_index + i]);
    }
    for (i=0; i<myNodeCount; ++i) {
        value = 0;
        for (k=0; k<myVxCount; ++k) {
            int offset = k*myNodeCount*myVyCount + i*myVyCount;
            for (j=offset; j < myVyCount+offset; j++) {
                value +=f0_f[j] * myVolume[i*myVxCount*myVyCount + k*myVyCount + (j-offset)] * 0.5 * myMuQoi[k] * myVth2[i] * myParticleMass;
            }
        }
        tperp.push_back(value/density[den_index + i]/mySmallElectronCharge);
    }
    for (i=0; i<myNodeCount; ++i) {
        upar_.push_back(upara[den_index + i]/myVth[i]);
    }
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVyCount; ++j)
            en.push_back(0.5*pow((myVp[j]-upar_[i]),2));
    }
    for (i=0; i<myNodeCount; ++i) {
        value = 0;
        for (k=0; k<myVxCount; ++k) {
            int offset = k*myNodeCount*myVyCount + i*myVyCount;
            for (j=offset; j < myVyCount+offset; j++) {
                value += f0_f[j] * myVolume[i*myVxCount*myVyCount + k*myVyCount + (j-offset)] * en[myVyCount*i+(j-offset)] * myVth2[i] * myParticleMass;
            }
        }
        tpara.push_back(2.0*value/density[den_index + i]/mySmallElectronCharge);
    }
#if 0
    if (myPlaneOffset == 0 && myNodeOffset == 0) {
        FILE* fp = fopen ("density.txt", "w");
        for (i=0; i<myNodeCount; ++i) {
            fprintf(fp, "%f\n", density[i]);
        }
        fclose(fp);
        fp = fopen ("upara.txt", "w");
        for (i=0; i<myNodeCount; ++i) {
            fprintf(fp, "%f\n", upara[i]);
        }
        fclose(fp);
        fp = fopen ("tperp.txt", "w");
        for (i=0; i<myNodeCount; ++i) {
            fprintf(fp, "%f\n", tperp[i]);
        }
        fclose(fp);
        fp = fopen ("tpara.txt", "w");
        for (i=0; i<myNodeCount; ++i) {
            fprintf(fp, "%f\n", tpara[i]);
        }
        fclose(fp);
    }
#endif
    for (i=0; i<myNodeCount; ++i) {
        n0.push_back(density[i]);
        t0.push_back((2.0*tperp[i]+tpara[i])/3.0);
    }
    return;
}
#endif

#if 0
void compute_C_qois_new(int iphi,
      std::vector <double> &density, std::vector <double> &upara,
      std::vector <double> &tperp, std::vector <double> &tpara,
      std::vector <double> &n0, std::vector <double> &t0,
      const double* dataIn)
{
    std::vector <double> myden(myNodeCount, 0);
    std::vector <double> myupara(myNodeCount, 0);
    std::vector <double> mytperp(myNodeCount, 0);
    std::vector <double> upar_;
    std::vector <double> tper;
    std::vector <double> en;
    std::vector <double> T_par;
    // std::vector <double> upar_(myNodeCount, 0);
    // std::vector <double> tper(myNodeCount*myVxCount*myVyCount, 0);
    // std::vector <double> en (myNodeCount*myVxCount, 0);
    // std::vector <double> T_par(myNodeCount*myVxCount*myVyCount, 0);
    int i, j, k, l;

    const double* f0_f = &dataIn[iphi*myNodeCount*myVxCount*myVyCount];
    int den_index = iphi*myNodeCount;

#pragma omp parallel for
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        myden[i] += f0_f[k] * myVolume[k];
    }
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int(k%myVyCount);
        l = int(k/myVyCount);
        myupara[i] += (f0_f[k] * myVolume[k] * myVth[i] * myVp[j])/myden[i];
        // mytperp[i] = (f0_f[k] * myVolume[k] * 0.5 * myMuQoi[l] * myVth2[i] * myParticleMass)/myden[i]/mySmallElectronCharge;
    }
    density.assign(myden.begin(), myden.end());
    upara.assign(myupara.begin(), myupara.end());
    // tperp.assign(mytperp.begin(), mytperp.end());
    for (i=0; i<myNodeCount; ++i) {
        upar_.push_back(upara[den_index + i]/myVth[i]);
    }
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            for (k=0; k<myVyCount; ++k) {
                tper.push_back(f0_f[myVxCount*myVyCount*i + myVyCount*j + k] *
                    myVolume[myVxCount*myVyCount*i + myVyCount*j + k] * 0.5 *
                    myMuQoi[j] * myVth2[i] * myParticleMass);
            }
    }
    for (i=0; i<myNodeCount; ++i) {
        double value = 0;
        for (j=0; j<myVxCount*myVyCount; ++j) {
            value += tper[myVxCount*myVyCount*i + j];
        }
        tperp.push_back(value/density[den_index + i]/mySmallElectronCharge);
        // printf ("Tperp %g, %g, %g, %g\n", value/density[den_index + i]/mySmallElectronCharge, value, myParticleMass, mySmallElectronCharge);
    }
#if 0
    for (k=0; k<myNodeCount*myVxCount*myVyCount; ++k) {
        i = int(k/(myVxCount*myVyCount));
        j = int(k%myVyCount);
        l = int(k/myVyCount);
        myupara[i] += (f0_f[k] * myVolume[k] * myVth[i] * myVp[j])/myden[i];
        mytperp[i] = (f0_f[k] * myVolume[k] * 0.5 * myMuQoi[l] * myVth2[i] * myParticleMass)/myden[i]/mySmallElectronCharge;
    }
#endif
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            en.push_back(0.5*pow((myVp[j]-upar_[i]),2));
    }
    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            for (k=0; k<myVyCount; ++k)
                T_par.push_back(f0_f[myVxCount*myVyCount*i + myVyCount*j + k] *
                    myVolume[myVxCount*myVyCount*i + myVyCount*j + k] *
                    en[myVxCount*i+k] * myVth2[i] * myParticleMass);
    }
    for (i=0; i<myNodeCount; ++i) {
        double value = 0;
        for (j=0; j<myVxCount*myVyCount; ++j) {
            value += T_par[myVxCount*myVyCount*i + j];
        }
        tpara.push_back(2.0*value/density[den_index + i]/mySmallElectronCharge);
    }
    for (i=0; i<myNodeCount; ++i) {
        n0.push_back(myDensity[i]);
        t0.push_back((2.0*myTperp[i]+myTpara[i])/3.0);
    }
    return;
}
#endif

bool LagrangeTorch::isConverged(std::vector <double> difflist, double eB, int count)
{
    bool status = false;
    if (count < 2) {
         return status;
    }
    double last2Val = difflist[count-2];
    double last1Val = difflist[count-1];
    if (abs(last2Val - last1Val) < eB) {
        status = true;
    }
    return status;
}

#if 0
void LagrangeTorch::compareErrorsPD(const double* reconData, const double* bregData, int rank)
{
    double pd_b;
    size_t pd_size_b;
    double pd_min_b;
    double pd_max_b;
    double pd_a;
    size_t pd_size_a;
    double pd_min_a;
    double pd_max_a;
    double pd_error_b = rmseErrorPD(reconData, pd_b, pd_max_b, pd_min_b, pd_size_b);
    double pd_error_a = rmseErrorPD(bregData, pd_a, pd_max_a, pd_min_a, pd_size_a);
    // get total error for recon
    double pd_e_b;
    size_t pd_s_b;
    double pd_omin_b;
    double pd_omax_b;
    MPI_Allreduce(&pd_b, &pd_e_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_size_b, &pd_s_b, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_min_b, &pd_omin_b, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_max_b, &pd_omax_b, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double pd_e_a;
    size_t pd_s_a;
    double pd_omin_a;
    double pd_omax_a;
    MPI_Allreduce(&pd_a, &pd_e_a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    printf ("Rank %d Num elements %d\n", rank, pd_size_a);
    MPI_Allreduce(&pd_size_a, &pd_s_a, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_min_a, &pd_omin_a, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_max_a, &pd_omax_a, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf ("%d Overall PD Error: %f %f\n", mySpecies, sqrt(pd_e_b/pd_s_b)/(pd_omax_b-pd_omin_b), sqrt(pd_e_a/pd_s_a)/(pd_omax_a-pd_omin_a));
        // printf ("PD Error stats: %f %d %f %f %f %d %f %f\n", pd_e_b, pd_s_b,pd_omax_b,pd_omin_b, pd_e_a,pd_s_a,pd_omax_a,pd_omin_a);
    }
}
#endif

void LagrangeTorch::compareErrorsPD(const double* reconData, const double* bregData, int rank)
{
    double pd_b;
    double pd_size_b;
    double pd_min_b;
    double pd_max_b;
    double pd_a;
    double pd_size_a;
    double pd_min_a;
    double pd_max_a;
    double pd_error_b = rmseErrorPD(reconData, pd_b, pd_max_b, pd_min_b, pd_size_b);
    double pd_error_a = rmseErrorPD(bregData, pd_a, pd_max_a, pd_min_a, pd_size_a);
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
        printf ("%d Overall PD Error: %f %f\n", mySpecies, sqrt(pd_e_b/pd_s_b)/(pd_omax_b-pd_omin_b), sqrt(pd_e_a/pd_s_a)/(pd_omax_a-pd_omin_a));
        // printf ("PD Error stats: %f %f %f %f %f %f %f %f\n", pd_e_b, pd_s_b,pd_omax_b,pd_omin_b, pd_e_a,pd_s_a,pd_omax_a,pd_omin_a);
    }
}

void LagrangeTorch::compareErrorsQoI(std::vector <double> &refqoi, std::vector <double> &rqoi, std::vector <double> &bqoi, const char* qoi, int rank)
{
    double pd_b;
    int pd_size_b;
    double pd_min_b;
    double pd_max_b;
    double pd_a;
    int pd_size_a;
    double pd_min_a;
    double pd_max_a;
    double pd_error_b = rmseError(refqoi, rqoi, pd_b, pd_max_b, pd_min_b, pd_size_b);
    double pd_error_a = rmseError(refqoi, bqoi, pd_a, pd_max_a, pd_min_a, pd_size_a);
    // get total error for recon
    double pd_e_b;
    int pd_s_b;
    double pd_omin_b;
    double pd_omax_b;
    MPI_Allreduce(&pd_b, &pd_e_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_size_b, &pd_s_b, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_min_b, &pd_omin_b, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_max_b, &pd_omax_b, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double pd_e_a;
    int pd_s_a;
    double pd_omin_a;
    double pd_omax_a;
    MPI_Allreduce(&pd_a, &pd_e_a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_size_a, &pd_s_a, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_min_a, &pd_omin_a, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&pd_max_a, &pd_omax_a, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf ("%d Overall %s Error: %f %10.5g\n", mySpecies, qoi, sqrt(pd_e_b/pd_s_b)/(pd_omax_b-pd_omin_b), sqrt(pd_e_a/pd_s_a)/(pd_omax_a-pd_omin_a));
    }
}

void LagrangeTorch::compareQoIs(const double* reconData,
        const double* bregData)
{
    int iphi;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    std::vector <double> rdensity;
    std::vector <double> rupara;
    std::vector <double> rtperp;
    std::vector <double> rtpara;
    std::vector <double> rn0;
    std::vector <double> rt0;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        compute_C_qois(iphi, rdensity, rupara, rtperp, rtpara, rn0, rt0, reconData);
#if 0
        if (my_rank == 0) {
            FILE* fp = fopen("PartialOrigQoI.txt", "w");
            for (int i=0; i<rdensity.size(); ++i) {
                fprintf (fp, "(%d %d) density %5.3g upara %5.3g tperp %5.3g tpara %5.3g n0 %5.3g t0 %5.3g vth %5.3g vp %5.3g\n fof %5.3g", myPlaneOffset, i, rdensity[i], rupara[i], rtperp[i], rtpara[i], rn0[i], rt0[0], myVth[i], myVp[i%39], reconData[i]);
                if (i==0) {
                    for (int j=0; j<1521; ++j) {
                        // fprintf (fp, "(%d %d) f0f %5.3g volume %5.3g\n", myPlaneOffset, j, reconData[j], myVolume[j]);
                        fprintf (fp, "(%d %d) volume %5.3g grid volume %5.3g nvp %d nmu %d\n", myPlaneOffset, j, myVolume[j], myGridVolume[j+myNodeCount], myF0Nvp[0], myF0Nmu[0]);
                    }
                }
            }
            fclose (fp);
            fp = fopen("PartialOrigF0F.txt", "w");
            for (int i=0; i<10*39*39; ++i) {
                fprintf (fp, "(%d %d) %5.3g\n", i/1521, i%1521, reconData[i]);
            }
            fclose (fp);
        }
#endif
    }
    std::vector <double> bdensity;
    std::vector <double> bupara;
    std::vector <double> btperp;
    std::vector <double> btpara;
    std::vector <double> bn0;
    std::vector <double> bt0;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        compute_C_qois(iphi, bdensity, bupara, btperp, btpara, bn0, bt0, bregData);
#if 0
        if (my_rank == 0) {
            FILE* fp = fopen("PartialBregQoI.txt", "w");
            for (int i=0; i<bdensity.size(); ++i) {
                fprintf (fp, "(%d %d) density %5.3g upara %5.3g tperp %5.3g tpara %5.3g n0 %5.3g t0 %5.3g vth %5.3g vp %5.3g fof %5.3g\n", myPlaneOffset, i, bdensity[i], bupara[i], btperp[i], btpara[i], bn0[i], bt0[0], myVth[i], myVp[i%39], bregData[i]);
                if (i==0) {
                    for (int j=0; j<1521; ++j) {
                        // fprintf (fp, "(%d %d) f0f %5.3g volume %5.3g\n", myPlaneOffset, j, bregData[j], myVolume[j]);
                        fprintf (fp, "(%d %d) volume %5.3g grid volume %5.3g nvp %d nmu %d\n", myPlaneOffset, j, myVolume[j], myGridVolume[j+myNodeCount], myF0Nvp[0], myF0Nmu[0]);
                    }
                }
            }
            fclose (fp);
            fp = fopen("PartialBregF0F.txt", "w");
            for (int i=0; i<10*39*39; ++i) {
                fprintf (fp, "(%d %d) %5.3g\n", i/1521, i%1521, bregData[i]);
            }
            fclose (fp);
        }
#endif
    }
    std::vector <double> refdensity;
    std::vector <double> refupara;
    std::vector <double> reftperp;
    std::vector <double> reftpara;
    std::vector <double> refn0;
    std::vector <double> reft0;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        compute_C_qois(iphi, refdensity, refupara, reftperp, reftpara, refn0, reft0, myDataIn.data());
#if 0
        if (my_rank == 0) {
            FILE* fp = fopen("PartialRefQoI.txt", "w");
            for (int i=0; i<bdensity.size(); ++i) {
                fprintf (fp, "(%d %d) density %5.3g upara %5.3g tperp %5.3g tpara %5.3g n0 %5.3g t0 %5.3g vth %5.3g vp %5.3g fof %5.3g\n", myPlaneOffset, i, bdensity[i], bupara[i], btperp[i], btpara[i], bn0[i], bt0[0], myVth[i], myVp[i%39], bregData[i]);
                if (i==0) {
                    for (int j=0; j<1521; ++j) {
                        // fprintf (fp, "(%d %d) f0f %5.3g volume %5.3g\n", myPlaneOffset, j, bregData[j], myVolume[j]);
                        fprintf (fp, "(%d %d) volume %5.3g grid volume %5.3g nvp %d nmu %d\n", myPlaneOffset, j, myVolume[j], myGridVolume[j+myNodeCount], myF0Nvp[0], myF0Nmu[0]);
                    }
                }
            }
            fclose (fp);
            fp = fopen("PartialRefF0F.txt", "w");
            for (int i=0; i<10*39*39; ++i) {
                fprintf (fp, "(%d %d) %5.3g\n", i/1521, i%1521, bregData[i]);
            }
            fclose (fp);
        }
#endif
    }
    compareErrorsPD(reconData, bregData, my_rank);
    compareErrorsQoI(refdensity, rdensity, bdensity, "density", my_rank);
    compareErrorsQoI(refupara, rupara, bupara, "upara", my_rank);
    compareErrorsQoI(reftperp, rtperp, btperp, "tperp", my_rank);
    compareErrorsQoI(reftpara, rtpara, btpara, "tpara", my_rank);
    compareErrorsQoI(refn0, rn0, bn0, "n0", my_rank);
    compareErrorsQoI(reft0, rt0, bt0, "T0", my_rank);
#if 0
    if (isnan(pd_error_a)) {
        for (int i=0; i<myLocalElements; ++i) {
            printf ("Breg data: %d %f\n", i, bregData[i]);
        }
    }
#endif
    return;
}

double LagrangeTorch::rmseErrorPD(const double* y, double &e, double &maxv, double &minv, double &nsize)
{
    e = 0;
    maxv = -99999;
    minv = 99999;
    const double* x = myDataIn.data();
    nsize = double(myLocalElements);
    for (int i=0; i<nsize; ++i) {
        e += pow((x[i] - y[i]), 2);
        if (x[i] < minv) {
            minv = x[i];
        }
        if (x[i] > maxv) {
            maxv = x[i];
        }
    }
    return sqrt(e/nsize)/(maxv-minv);
}

double LagrangeTorch::rmseError(std::vector <double> &x, std::vector <double> &y, double &e, double &maxv, double &minv, int &ysize)
{
    int xsize = x.size();
    ysize = y.size();
    assert(xsize == ysize);
    e = 0;
    maxv = -99999;
    minv = 99999;
    for (int i=0; i<xsize; ++i) {
        e += pow((x[i] - y[i]), 2);
        if (x[i] < minv) {
            minv = x[i];
        }
        if (x[i] > maxv) {
            maxv = x[i];
        }
    }
    return sqrt(e/xsize)/(maxv-minv);
}

double LagrangeTorch::determinant(double a[4][4], double k)
{
  double s = 1, det = 0;
  double b[4][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
  int i, j, m, n, c;
  if (k == 1)
    {
     return (a[0][0]);
    }
  else
    {
     det = 0;
     for (c = 0; c < k; c++)
       {
        m = 0;
        n = 0;
        for (i = 0;i < k; i++)
          {
            for (j = 0 ;j < k; j++)
              {
                b[i][j] = 0;
                if (i != 0 && j != c)
                 {
                   b[m][n] = a[i][j];
                   if (n < (k - 2))
                    n++;
                   else
                    {
                     n = 0;
                     m++;
                     }
                   }
               }
             }
          det = det + s * (a[0][c] * determinant(b, k - 1));
          s = -1 * s;
          }
    }

    return (det);
}

double** LagrangeTorch::cofactor(double num[4][4], double f)
{
 double b[4][4], fac[4][4];
 int p, q, m, n, i, j;
 for (q = 0;q < f; q++)
 {
   for (p = 0;p < f; p++)
    {
     m = 0;
     n = 0;
     for (i = 0;i < f; i++)
     {
       for (j = 0;j < f; j++)
        {
          if (i != q && j != p)
          {
            b[m][n] = num[i][j];
            if (n < (f - 2))
             n++;
            else
             {
               n = 0;
               m++;
               }
            }
        }
      }
      fac[q][p] = pow(-1, q + p) * determinant(b, f - 1);
    }
  }
  return transpose(num, fac, f);
}
/*Finding transpose of matrix*/
double** LagrangeTorch::transpose(double num[4][4], double fac[4][4], double r)
{
  int i, j;
  double b[4][4], d;
  double** inverse = new double* [4];
  inverse[0] = new double[4];
  inverse[1] = new double[4];
  inverse[2] = new double[4];
  inverse[3] = new double[4];

  for (i = 0;i < r; i++)
    {
     for (j = 0;j < r; j++)
       {
         b[i][j] = fac[j][i];
        }
    }
  d = determinant(num, r);
  for (i = 0;i < r; i++)
    {
     for (j = 0;j < r; j++)
       {
        inverse[i][j] = b[i][j] / d;
        }
    }
    return inverse;
}
