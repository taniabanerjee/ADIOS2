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
        void computeParamsAndQoIs(const std::string meshFile,
                adios2::Dims blockStart, adios2::Dims blockCount,
                const double* dataIn);
        std::vector <double> computeLagrangeParameters(const double* reconstructedData);

    private:
        // APIs
        void readF0Params(const std::string meshFile);
        void setVolume();
        void setVp();
        void setMuQoi();
        void setVth2();
        void compute_C_qois(int iphi);
        std::vector <double> qoi_V2();
        std::vector <double> qoi_V3();
        std::vector <double> qoi_V4();
        bool isConverged(std::vector <double> difflist, double eB);
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
};

#endif
