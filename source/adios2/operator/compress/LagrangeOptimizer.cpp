#include <time.h>
#include <assert.h>
#include "LagrangeOptimizer.hpp"
#include "adios2/core/Engine.h"
#include "adios2/helper/adiosFunctions.h"

LagrangeOptimizer::LagrangeOptimizer()
{
    // Initialize charge and mass variables
    mySmallElectronCharge = 1.6022e-19;
    myParticleMass = 3.344e-27;
}

LagrangeOptimizer::~LagrangeOptimizer()
{
}

void LagrangeOptimizer::computeParamsAndQoIs(const std::string meshFile,
     adios2::Dims blockStart, adios2::Dims blockCount,
     const double* dataIn)
{
    clock_t start = clock();
    int planeIndex = 0;
    int nodeIndex = 2;
    int velXIndex = 1;
    int velYIndex = 3;
    int iphi = 0;
    myNodeOffset = blockStart[nodeIndex];
    myNodeCount = blockCount[nodeIndex];
    myPlaneCount = blockCount[planeIndex];
    myVxCount = blockCount[velXIndex];
    myVyCount = blockCount[velYIndex];
    printf("#planes: %d, #nodes: %d, #vx: %d, #vy: %d\n", myPlaneCount, myNodeCount, myVxCount, myVyCount);
    myLocalElements = myNodeCount * myPlaneCount * myVxCount * myVyCount;
    myDataIn = dataIn;
    readF0Params(meshFile);
    setVolume();
    setVp();
    setMuQoi();
    setVth2();
    printf ("volume: gv %d f0_nvp %d f0_nmu %d, vp: %d, vth: %d, vth2: %d, mu_qoi: %d\n", myGridVolume.size(), myF0Nvp.size(), myF0Nmu.size(), myVp.size(), myVth.size(), myVth2.size(), myMuQoi.size());
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        compute_C_qois(iphi, myDensity, myUpara, myTperp, myTpara, myN0, myT0, myDataIn);
    }
    myMaxValue = 0;
    for (size_t i = 0; i < myLocalElements; ++i) {
        myMaxValue = (myMaxValue > myDataIn[i]) ? myMaxValue : myDataIn[i];
    }
    printf ("Time Taken for QoI Computation: %5.3g\n", ((double)(clock()-start))/CLOCKS_PER_SEC);
}

std::vector <double> LagrangeOptimizer::computeLagrangeParameters(
    const double* reconData)
{
    clock_t start = clock();
    int ii, i, j, k;
    for (ii=0; ii<myLocalElements; ++ii) {
        if (!(reconData[ii] > 0)) {
            ((double*)reconData)[ii] = 100;
        }
    }
    double* breg_recon = new double[myLocalElements];
    int count = 0;
    double gradients[4] = {0.0, 0.0, 0.0, 0.0};
    double hessians[4][4] = {0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0};
    double K[myVxCount*myVyCount];
    double breg_result[myVxCount*myVyCount];
    memset(K, 0, myVxCount*myVyCount*sizeof(double));
    std::vector <double> V2 = qoi_V2();
    std::vector <double> V3 = qoi_V3();
    std::vector <double> V4 = qoi_V4();
    std::vector <double> tpara_data;
    int idx, iphi;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        for (i=0; i<myNodeCount; ++i) {
            tpara_data.push_back(mySmallElectronCharge *
                myTpara[myNodeCount*iphi + i] +
                myVth2[i] * myParticleMass *
                pow((myUpara[myNodeCount*iphi + i]/myVth[i]), 2));
        }
    }
    int breg_index = 0;
    // start = clock();
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        double* D = &myDensity[iphi*myNodeCount];
        double* U = &myUpara[iphi*myNodeCount];
        double* Tperp = &myTperp[iphi*myNodeCount];
        double* Tpara = &tpara_data[iphi*myNodeCount];
        int count_unLag = 0;
        std::vector <int> node_unconv;
        double maxD = -99999;
        double maxU = -99999;
        double maxTperp = -99999;
        double maxTpara = -99999;
        for (i=0; i<myNodeCount; ++i) {
            if (D[i] > maxD) {
                maxD = D[i];
            }
            if (U[i] > maxU) {
                maxU = U[i];
            }
            if (Tperp[i] > maxTperp) {
                maxTperp = Tperp[i];
            }
            if (Tpara[i] > maxTpara) {
                maxTpara = Tpara[i];
            }
        }
        double DeB = pow(maxD*1e-09, 2);
        double UeB = pow(maxU*1e-09, 2);
        double TperpEB = pow(maxTperp*1e-09, 2);
        double TparaEB = pow(maxTpara*1e-09, 2);
        double PDeB = pow(myMaxValue*1e-09, 2);
        const double* f0_f = &myDataIn[iphi*myNodeCount*myVxCount*myVyCount];
        for (idx=0; idx<myNodeCount; ++idx) {
            const double* recon_one = &reconData[myNodeCount*myVxCount*myVyCount*iphi + myVxCount*myVyCount*idx];
            double lambdas[4] = {0.0, 0.0, 0.0, 0.0};
            std::vector <double> L2_den;
            std::vector <double> L2_upara;
            std::vector <double> L2_tperp;
            std::vector <double> L2_tpara;
            std::vector <double> L2_PD;
            count = 0;
            double aD = D[idx]*mySmallElectronCharge;
            while (1) {
                for (i=0; i<myVxCount*myVyCount; ++i) {
                    K[i] = lambdas[0]*myVolume[myVxCount*myVyCount*idx + i] +
                           lambdas[1]*V2[myVxCount*myVyCount*idx + i] +
                           lambdas[2]*V3[myVxCount*myVyCount*idx + i] +
                           lambdas[3]*V4[myVxCount*myVyCount*idx + i];
                }
#ifdef UF_DEBUG
                printf("L1 %g, L2 %g L3 %g, L4 %g K[0] %g\n", lambdas[0], lambdas[1], lambdas[2], lambdas[3], exp(-K[0]));
#endif
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
                        rmse_pd += pow((breg_result[i] - f0_f[myVxCount*myVyCount*idx]), 2);
                    }
                    L2_den.push_back(pow((update_D - D[idx]), 2));
                    L2_upara.push_back(pow((update_U - U[idx]), 2));
                    L2_tperp.push_back(pow((update_Tperp-myTperp[idx]), 2));
                    L2_tpara.push_back(pow((update_Tpara-Tpara[idx]), 2));
                    L2_PD.push_back(sqrt(rmse_pd));
                    bool c1, c2, c3, c4;
                    bool converged = (isConverged(L2_den, DeB)
                        && isConverged(L2_upara, UeB)
                        && isConverged(L2_tpara, TparaEB)
                        && isConverged(L2_tperp, TperpEB))
                        && isConverged(L2_PD, PDeB);
                    if (converged) {
                        for (i=0; i<myVxCount*myVyCount; ++i) {
                            breg_recon[breg_index++] = breg_result[i];
                        }
                        /*
                        double mytperp[1521];
                        for (i=0; i<myVxCount*myVyCount; ++i) {
                            mytperp[i] = breg_result[i]*V3[
                                myVxCount*myVyCount*idx + i]/aD;
                        }
                        printf ("Mytperp %d\n", mytperp[0]);
                        */
                        myLagranges.push_back(lambdas[0]);
                        myLagranges.push_back(lambdas[1]);
                        myLagranges.push_back(lambdas[2]);
                        myLagranges.push_back(lambdas[3]);
#ifdef UF_DEBUG
                        if (idx % 2000 == 0) {
                            printf ("node %d finished\n", idx);
                        }
#endif
                        break;
                    }
                    else if (count == 20 && !converged) {
                        for (i=0; i<myVxCount*myVyCount; ++i) {
                            breg_recon[breg_index++] = recon_one[i];
                        }
                        myLagranges.push_back(lambdas[0]);
                        myLagranges.push_back(lambdas[1]);
                        myLagranges.push_back(lambdas[2]);
                        myLagranges.push_back(lambdas[3]);
                        printf ("Node %d did not converge\n", idx);
                        count_unLag = count_unLag + 1;
                        node_unconv.push_back(idx);
                        break;
                    }
                }
                double gvalue1 = D[idx], gvalue2 = U[idx]*D[idx];
                double gvalue3 = Tperp[idx]*aD, gvalue4 = Tpara[idx]*D[idx];
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
                }
                else{
                    double** inverse = cofactor(hessians, order);
#if UF_DEBUG
                    printf("Hessians: %g, %g, %g, %g\n", hessians[0][0], hessians[0][1], hessians[0][2], hessians[0][3]);
                    printf("Inverse: %g, %g, %g, %g\n", inverse[0][0], inverse[0][1], inverse[0][2], inverse[0][3]);
#endif
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
        }
    }
    printf ("Time Taken for Optimization Computation: %5.3g\n", ((double)(clock()-start))/CLOCKS_PER_SEC);
    compareQoIs(reconData, breg_recon);
    return myLagranges;
}

long unsigned int LagrangeOptimizer::getParameterSize()
{
    return myNodeCount * 4 * sizeof(double);
}

// Get the number of bytes needed to store the PQ table
long unsigned int LagrangeOptimizer::getTableSize()
{
    return 0;
}

size_t LagrangeOptimizer::putResult(char* &bufferOut, size_t &bufferOutOffset)
{
    // TODO: after your algorithm is done, put the result into
    // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your       first
    // double number *reinterpret_cast<double*>(bufferOut+bufferOutOff      set+8)
    // for your second double number and so on

    int count = 0;
    for (double d : myLagranges) {
          *reinterpret_cast<double*>(
              bufferOut+bufferOutOffset+(count++)*sizeof(double)) = d;
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
        if (count == lagrangeCount+myNodeCount) {
             printf("Grid vol element %f\n", d);
        }
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
    int doubleCount = count*sizeof(double);
    *reinterpret_cast<int*>(
        bufferOut+bufferOutOffset+doubleCount) = myF0Nvp[0];
    *reinterpret_cast<int*>(
        bufferOut+bufferOutOffset+doubleCount+sizeof(int)) = myF0Nmu[0];
    return doubleCount + 2*sizeof(int);
}

void LagrangeOptimizer::setDataFromCharBuffer(double* &reconData,
    const char* bufferIn, size_t bufferOffset, size_t totalSize)
{
    size_t bufferSize = totalSize - bufferOffset;
    int i;
    // Assuming the Lagrange parameters are stored as double numbers
    // This will change as we add quantization
    // Find node size
    myNodeCount = int((bufferSize-2*sizeof(double)-2*sizeof(int))/(6*sizeof(double)));
    int numLagrangeParameters = myNodeCount*4;
    for (i=0; i<numLagrangeParameters; ++i) {
        myLagranges.push_back(*reinterpret_cast<const double*>(bufferIn+bufferOffset+i*sizeof(double)));
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
}

void LagrangeOptimizer::readCharBuffer(const char* bufferIn, size_t bufferOffset, size_t bufferSize)
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

void LagrangeOptimizer::readF0Params(const std::string meshFile)
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
}

void LagrangeOptimizer::setVolume()
{
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
        for (int jj=0; jj<mu_vol.size(); ++jj) {
            mu_vp_vol.push_back(mu_vol[ii] * vp_vol[jj]);
        }
    }
    if (myGridVolume.size() == 2*myNodeCount) {
        for (int ii=0; ii<myNodeCount; ++ii) {
            for (int jj=0; jj<mu_vp_vol.size(); ++jj) {
            // Access myGridVolume with an offset of nnodes to get to the electrons
                myVolume.push_back(myGridVolume[myNodeCount+ii] * mu_vp_vol[jj]);
            }
        }
    }
    else {
        for (int ii=0; ii<myNodeCount; ++ii) {
            for (int jj=0; jj<mu_vp_vol.size(); ++jj) {
            // Access myGridVolume with an offset of nnodes to get to the electrons
                myVolume.push_back(myGridVolume[ii] * mu_vp_vol[jj]);
            }
        }
    }
    return;
}

void LagrangeOptimizer::setVp()
{
    for (int ii = -myF0Nvp[0]; ii<myF0Nvp[0]+1; ++ii) {
        myVp.push_back(ii*myF0Dvp[0]);
    }
    return;
}

void LagrangeOptimizer::setMuQoi()
{
    for (int ii = 0; ii<myF0Nmu[0]+1; ++ii) {
        myMuQoi.push_back(pow(ii*myF0Dsmu[0], 2));
    }
    return;
}

void LagrangeOptimizer::setVth2()
{
    double value = 0;
    if (myF0TEv.size() == 2*myNodeCount) {
        for (int ii=0; ii<myNodeCount; ++ii) {
            // Access f0_T_ev with an offset of myNodeCount to get to the electrons
            value = myF0TEv[myNodeCount+ii]*mySmallElectronCharge/myParticleMass;
            myVth2.push_back(value);
            myVth.push_back(sqrt(value));
        }
    }
    else{
        for (int ii=0; ii<myNodeCount; ++ii) {
            // Access f0_T_ev with an offset of myNodeCount to get to the electrons
            value = myF0TEv[ii]*mySmallElectronCharge/myParticleMass;
            myVth2.push_back(value);
            myVth.push_back(sqrt(value));
        }
    }
    return;
}

void LagrangeOptimizer::compute_C_qois(int iphi,
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
            for (k=0; k<myVyCount; ++k)
                upar.push_back(f0_f[myVxCount*myVyCount*i + myVyCount*j + k] *
                    myVolume[myVxCount*myVyCount*i + myVyCount*j + k] *
                    myVth[i]*myVp[k]);
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

std::vector <double> LagrangeOptimizer::qoi_V2()
{
    int i, j, k;
    std::vector <double> V2;

    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            for (k=0; k<myVyCount; ++k)
                V2.push_back(myVolume[myVxCount*myVxCount*i + myVxCount*j
                    + k] * myVth[i]*myVp[k]);
    }

    return V2;
}

std::vector <double> LagrangeOptimizer::qoi_V3()
{
    int i, j, k;
    std::vector <double> V3;

    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            for (k=0; k<myVyCount; ++k)
                V3.push_back(myVolume[myVxCount*myVxCount*i + myVxCount*j
                    + k] * 0.5 * myMuQoi[j] * myVth2[i] * myParticleMass);
    }

    return V3;
}

std::vector <double> LagrangeOptimizer::qoi_V4()
{
    int i, j, k;
    std::vector <double> V4;

    for (i=0; i<myNodeCount; ++i) {
        for (j=0; j<myVxCount; ++j)
            for (k=0; k<myVyCount; ++k)
                V4.push_back(myVolume[myVxCount*myVyCount*i + myVxCount*j + k] *
                    myVp[k]*myVp[k] * myVth2[i] * myParticleMass);
    }

    return V4;
}

bool LagrangeOptimizer::isConverged(std::vector <double> difflist, double eB)
{
    bool status = false;
    unsigned int vsize = difflist.size();
    if (vsize < 2) {
         return status;
    }
    double last2Val = difflist[vsize-2];
    double last1Val = difflist[vsize-1];
    if (abs(last2Val - last1Val) < eB) {
        status = true;
    }
    return status;
}

void LagrangeOptimizer::compareQoIs(const double* reconData,
        const double* bregData)
{
    int iphi;
    std::vector <double> rdensity;
    std::vector <double> rupara;
    std::vector <double> rtperp;
    std::vector <double> rtpara;
    std::vector <double> rn0;
    std::vector <double> rt0;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        compute_C_qois(iphi, rdensity, rupara, rtperp, rtpara, rn0, rt0, reconData);
    }
    std::vector <double> bdensity;
    std::vector <double> bupara;
    std::vector <double> btperp;
    std::vector <double> btpara;
    std::vector <double> bn0;
    std::vector <double> bt0;
    for (iphi=0; iphi<myPlaneCount; ++iphi) {
        compute_C_qois(iphi, bdensity, bupara, btperp, btpara, bn0, bt0, bregData);
    }
    double pd_error_b = rmseErrorPD(reconData);
    double pd_error_a = rmseErrorPD(bregData);
    printf ("PD errors %g, %g\n", pd_error_b, pd_error_a);
    double density_error_b = rmseError(myDensity, rdensity);
    double density_error_a = rmseError(myDensity, bdensity);
    printf ("Density errors %g, %g\n", density_error_b, density_error_a);
    double upara_error_b = rmseError(myUpara, rupara);
    double upara_error_a = rmseError(myUpara, bupara);
    printf ("Upara errors %g, %g\n", upara_error_b, upara_error_a);
    double tperp_error_b = rmseError(myTperp, rtperp);
    double tperp_error_a = rmseError(myTperp, btperp);
    printf ("Tperp errors %g, %g\n", tperp_error_b, tperp_error_a);
    double tpara_error_b = rmseError(myTpara, rtpara);
    double tpara_error_a = rmseError(myTpara, btpara);
    printf ("Tpara errors %g, %g\n", tpara_error_b, tpara_error_a);
    double n0_error_b = rmseError(myN0, rn0);
    double n0_error_a = rmseError(myN0, bn0);
    printf ("n0 errors %g, %g\n", n0_error_b, n0_error_a);
    double T0_error_b = rmseError(myT0, rt0);
    double T0_error_a = rmseError(myT0, bt0);
    printf ("T0 errors %g, %g\n", T0_error_b, T0_error_a);
    return;
}

double LagrangeOptimizer::rmseErrorPD(const double* y)
{
    double e = 0;
    double maxv = -99999;
    double minv = 99999;
    const double* x = myDataIn;
    int nsize = myLocalElements;
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

double LagrangeOptimizer::rmseError(std::vector <double> &x, std::vector <double> &y)
{
    unsigned int xsize = x.size();
    unsigned int ysize = y.size();
    assert(xsize == ysize);
    double e = 0;
    double maxv = -99999;
    double minv = 99999;
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

double LagrangeOptimizer::rmseError2(std::vector <double> &x, std::vector <double> &y, int start)
{
    unsigned int xsize = x.size();
    unsigned int ysize = y.size();
    double e = 0;
    double maxv = -99999;
    double minv = 99999;
    for (int i=0; i<ysize; ++i) {
        e += pow((x[i+start] - y[i]), 2);
        if (x[i+start] < minv) {
            minv = x[i+start];
        }
        if (x[i+start] > maxv) {
            maxv = x[i+start];
        }
    }
    return sqrt(e/ysize)/(maxv-minv);
}

double LagrangeOptimizer::determinant(double a[4][4], double k)
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

double** LagrangeOptimizer::cofactor(double num[4][4], double f)
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
double** LagrangeOptimizer::transpose(double num[4][4], double fac[4][4], double r)
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
