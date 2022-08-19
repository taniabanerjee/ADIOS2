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
                const int applyPQ);
        size_t getTableSize();
        size_t putResultNoPQ(char* &bufferOut, size_t &bufferOutOffset);
        size_t putResult(char* &bufferOut, size_t &bufferOutOffset);
        size_t putResultV1(char* &bufferOut, size_t &bufferOutOffset);
        size_t putResultV2(char* &bufferOut, size_t &bufferOutOffset);
        char* setDataFromCharBufferV1(double* &dataOut, const char* bufferIn, size_t bufferTotalSize);
        void setDataFromCharBuffer(double* &dataOut, const char* bufferIn, size_t bufferTotalSize);
        void setDataFromCharBuffer2(double* &dataOut, const char* bufferIn, size_t bufferOffset, size_t bufferSize);
        void readCharBuffer(const char* bufferIn, size_t bufferOffset,
                size_t bufferSize);

    private:
        // APIs
        void readF0Params(const std::string meshFile);
        void setVolume();
        void setVolume(std::vector <double> &vol);
        void setVp();
        void setVp(std::vector <double> &vp);
        void setMuQoi();
        void setMuQoi(std::vector <double> &muqoi);
        void setVth2();
        void setVth2(std::vector <double> &vth,
            std::vector <double> &vth2);
        void compute_C_qois(int iphi, std::vector <double> &density,
            std::vector <double> &upara, std::vector <double> &tperp,
            std::vector <double> &tpara, std::vector <double> &n0,
            std::vector <double> &t0, const double* dataIn);
        bool isConverged(std::vector <double> difflist, double eB, int count);
        void compareQoIs(const double* reconData,
            const double* bregData);
        void compareErrorsPD(const double* reconData, const double* bregData, int rank);
        void compareErrorsQoI(std::vector <double> &x,
            std::vector <double> &y, std::vector <double> &z,
            const char* qoi, int rank);
        double rmseErrorPD(const double* y, double &e, double &maxv,
            double &minv, double &ysize);
        double rmseError(std::vector <double> &rqoi,
            std::vector <double> &bqoi, double &e, double &maxv,
            double &minv, int &ysize);
        double determinant(double a[4][4], double k);
        double** cofactor(double num[4][4], double f);
        double** transpose(double num[4][4], double fac[4][4], double r);
        void initializeClusterCenters(double* &clusters,
            double* lagarray, int numObjs);
        void quantizeLagranges(int offset, int* &membership,
            double* &cluster);
        void initializeClusterCentersMPI(double* &clusters, int numP,
            int myRank, double* lagarray, int numObjs);
        void quantizeLagrangesMPI(int offset, int* &membership,
            double* &cluster);
        size_t putPQIndexes(char* &bufferOut, size_t &bufferOutOffset);
        size_t putLagrangeParameters(char* &bufferOut,
                    size_t &bufferOutOffset);
        size_t getPQIndexes(const char* bufferIn);

    private:
        // Members
        at::Tensor myGridVolumeTorch;
        at::Tensor myF0TEvTorch;
        at::Tensor myVolumeTorch;
        at::Tensor myVpTorch;
        at::Tensor myMuQoiTorch;
        at::Tensor myVthTorch;
        at::Tensor myVth2Torch;
        static at::TensorOptions ourGPUOptions;
        static at::TensorOptions ourCPUOptions;
};

#endif
