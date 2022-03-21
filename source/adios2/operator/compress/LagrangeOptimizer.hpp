#ifndef LAGRANGE_OPTIMIZER_HPP
#define LAGRANGE_OPTIMIZER_HPP

#include "adios2/core/Operator.h"

class LagrangeOptimizer
{
    public:
        // Constructor
        LagrangeOptimizer();
        // Destructor
        ~LagrangeOptimizer();
        // Compute mesh parameters and QoIs
        void computeParamsAndQoIs(const std::string meshFile,
                adios2::Dims blockStart, adios2::Dims blockCount,
                const double* dataIn);
        // Compute Lagrange Parameters
        std::vector <double> computeLagrangeParameters(
                const double* reconstructedData);
        // Get the number of bytes needed to store the Lagrange parameters
        long unsigned int getParameterSize();
        // Get the number of bytes needed to store the PQ table
        long unsigned int getTableSize();
        size_t putResult(char* &bufferOut, size_t &bufferOutOffset);
        void setDataFromCharBuffer(double* &dataOut, const char* bufferIn, size_t bufferOffset, size_t bufferSize);
        void readCharBuffer(const char* bufferIn, size_t bufferOffset,
                size_t bufferSize);

    private:
        // APIs
        void readF0Params(const std::string meshFile);
        void setVolume();
        void setVp();
        void setMuQoi();
        void setVth2();
        void compute_C_qois(int iphi, std::vector <double> &density,
            std::vector <double> &upara, std::vector <double> &tperp,
            std::vector <double> &tpara, std::vector <double> &n0,
            std::vector <double> &t0, const double* dataIn);
        std::vector <double> qoi_V2();
        std::vector <double> qoi_V3();
        std::vector <double> qoi_V4();
        bool isConverged(std::vector <double> difflist, double eB);
        void compareQoIs(const double* reconData,
            const double* bregData);
        double rmseErrorPD(const double* y);
        double rmseError(std::vector <double> &x, std::vector <double> &y);
        double rmseError2(std::vector <double> &x, std::vector <double> &y, int start);
        double determinant(double a[4][4], double k);
        double** cofactor(double num[4][4], double f);
        double** transpose(double num[4][4], double fac[4][4], double r);

        // Members
        // Actual data being compressed and related parameters
        const double* myDataIn;
        long unsigned int myNodeOffset;
        long unsigned int myNodeCount;
        long unsigned int myPlaneCount;
        long unsigned int myVxCount;
        long unsigned int myVyCount;
        long unsigned int myLocalElements;
        double myMaxValue;
        // Constant physics parameters
        double mySmallElectronCharge;
        double myParticleMass;
        // Mesh Parameters
        std::vector <double> myGridVolume;
        std::vector <int> myF0Nvp;
        std::vector <int> myF0Nmu;
        std::vector <double> myF0Dvp;
        std::vector <double> myF0Dsmu;
        std::vector <double> myF0TEv;
        std::vector <double> myVolume;
        std::vector <double> myVp;
        std::vector <double> myMuQoi;
        std::vector <double> myVth2;
        std::vector <double> myVth;
        // Original QoIs
        std::vector <double> myDensity;
        std::vector <double> myUpara;
        std::vector <double> myTperp;
        std::vector <double> myTpara;
        std::vector <double> myN0;
        std::vector <double> myT0;
        // Lagrange Parameters
        std::vector <double> myLagranges;
        std::vector <double> myTable;
};

#endif
