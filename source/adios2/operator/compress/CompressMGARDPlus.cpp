/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * CompressMGARDPlus.cpp :
 *
 *  Created on: Feb 23, 2022
 *      Author: Jason Wang jason.ruonan.wang@gmail.com
 */

#include <vector>
#include <mpi.h>
#include <stdlib.h>
#include "LagrangeOptimizer.hpp"
#include "CompressMGARDPlus.h"
#include "CompressMGARD.h"
#include "CompressSZ.h"
#include "CompressZFP.h"
#include "adios2/core/Engine.h"
#include "adios2/helper/adiosFunctions.h"

namespace adios2
{
namespace core
{
namespace compress
{

CompressMGARDPlus::CompressMGARDPlus(const Params &parameters)
: Operator("mgardplus", COMPRESS_MGARDPLUS, "compress", parameters)
{
}

size_t CompressMGARDPlus::Operate(const char *dataIn, const Dims &blockStart,
                                  const Dims &blockCount, const DataType type,
                                  char *bufferOut)
{
    // Instantiate LagrangeOptimizer
    LagrangeOptimizer optim;
    // Read ADIOS2 files end, use data for your algorithm
    optim.computeParamsAndQoIs(m_Parameters["meshfile"], blockStart, blockCount,  reinterpret_cast<const double*>(dataIn));
    uint8_t compression_method = atoi(m_Parameters["compression_method"].c_str());
    size_t bufferOutOffset = 0;
    const uint8_t bufferVersion = 1;

    MakeCommonHeader(bufferOut, bufferOutOffset, bufferVersion);
    PutParameter(bufferOut, bufferOutOffset, compression_method);
    PutParameter(bufferOut, bufferOutOffset, optim.getPlaneOffset());
    PutParameter(bufferOut, bufferOutOffset, optim.getNodeOffset());
    size_t offsetForDecompresedData = bufferOutOffset;
    bufferOutOffset += sizeof(size_t);

    if (compression_method == 0) {
        CompressMGARD mgard(m_Parameters);
        size_t mgardBufferSize = mgard.Operate(dataIn, blockStart, blockCount, type,
                                bufferOut + bufferOutOffset);

        PutParameter(bufferOut, offsetForDecompresedData, mgardBufferSize);
        std::vector<char> tmpDecompressBuffer(
                helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));

        mgard.InverseOperate(bufferOut + bufferOutOffset, mgardBufferSize,
                             tmpDecompressBuffer.data());
        optim.computeLagrangeParameters(
                reinterpret_cast<const double*>(
                tmpDecompressBuffer.data()));
        bufferOutOffset += mgardBufferSize;
    }
    else if (compression_method == 1) {
        CompressSZ sz(m_Parameters);
        size_t szBufferSize = sz.Operate(dataIn, blockStart, blockCount, type,
                                bufferOut + bufferOutOffset);

        PutParameter(bufferOut, offsetForDecompresedData, szBufferSize);
        std::vector<char> tmpDecompressBuffer(
                helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));

        sz.InverseOperate(bufferOut + bufferOutOffset, szBufferSize,
                             tmpDecompressBuffer.data());
        optim.computeLagrangeParameters(
                reinterpret_cast<const double*>(
                tmpDecompressBuffer.data()));
        bufferOutOffset += szBufferSize;
    }
    else if (compression_method == 2) {
        CompressZFP zfp(m_Parameters);
        size_t zfpBufferSize = zfp.Operate(dataIn, blockStart, blockCount, type,
                                bufferOut + bufferOutOffset);

        PutParameter(bufferOut, offsetForDecompresedData, zfpBufferSize);
        std::vector<char> tmpDecompressBuffer(
                helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));

        zfp.InverseOperate(bufferOut + bufferOutOffset, zfpBufferSize,
                             tmpDecompressBuffer.data());
        optim.computeLagrangeParameters(
                reinterpret_cast<const double*>(
                tmpDecompressBuffer.data()));
        bufferOutOffset += zfpBufferSize;
    }
    // int my_rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // printf ("My compress rank %d, MGARD size %zu\n", my_rank, mgardBufferSize);
    // TODO: now the original data is in dataIn, the compressed and then
    // decompressed data is in tmpDecompressBuffer.data(). However, these
    // are char pointers, you will need to convert them into right types as
    // follows:
#if 0
    if (type == DataType::Double || type == DataType::DoubleComplex)
    {
        // TODO: use reinterpret_cast<double*>(dataIn) and
        // reinterpret_cast<double*>(tmpDecompressBuffer.data())
        // to read original data and decompressed data
        optim.computeLagrangeParameters(
                reinterpret_cast<const double*>(
                tmpDecompressBuffer.data()));
    }
    else if (type == DataType::Float || type == DataType::FloatComplex)
    {
        // TODO: use reinterpret_cast<float*>(dataIn) and
        // reinterpret_cast<double*>(tmpDecompressBuffer.data())
        // to read original data and decompressed data
    }
    // TODO: after your algorithm is done, put the result into
    // *reinterpret_cast<double*>(bufferOut+bufferOutOffset) for your first
    // double number *reinterpret_cast<double*>(bufferOut+bufferOutOffset+8)
    // for your second double number and so on
#endif
    size_t ppsize = optim.putResultV1(bufferOut, bufferOutOffset);
    bufferOutOffset += ppsize;

#ifdef UF_DEBUG
    int arraySize = optim.getPlaneCount()*optim.getNodeCount()*
      optim.getVxCount()*optim.getVyCount()*sizeof(double);
    char* dataOut = new char[arraySize];
    size_t result = InverseOperate(bufferOut, bufferOutOffset, dataOut);
#endif
    return bufferOutOffset;
}

Dims CompressMGARDPlus::GetBlockDims(const char *bufferIn,
        size_t bufferInOffset)
{
    const size_t ndims = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    Dims blockCount(ndims);
    for (size_t i = 0; i < ndims; ++i)
    {
        blockCount[i] = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    }
    return blockCount;
}

size_t CompressMGARDPlus::DecompressV1(const char *bufferIn,
        size_t bufferInOffset, const size_t sizeIn, char *dataOut)
{
    // Do NOT remove even if the buffer version is updated. Data might be still
    // in lagacy formats. This function must be kept for backward compatibility.
    // If a newer buffer format is implemented, create another function, e.g.
    // DecompressV2 and keep this function for decompressing lagacy data.

    const uint8_t compression_method =
        GetParameter<uint8_t>(bufferIn, bufferInOffset);
    const size_t planeOffset =
        GetParameter<size_t>(bufferIn, bufferInOffset);
    const size_t nodeOffset =
        GetParameter<size_t>(bufferIn, bufferInOffset);

    const size_t mgardBufferSize =
        GetParameter<size_t>(bufferIn, bufferInOffset);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    size_t planeCount, vxCount, nodeCount, vyCount;
    Dims blockDims = GetBlockDims(bufferIn, bufferInOffset+4);
    if (blockDims.size() == 3) {
         planeCount = 1;
         vxCount = blockDims[0];
         nodeCount = blockDims[1];
         vyCount = blockDims[2];
    }
    else if (blockDims.size() == 4) {
         planeCount = blockDims[0];
         vxCount = blockDims[1];
         nodeCount = blockDims[2];
         vyCount = blockDims[3];
    }
    // TODO: read your results here from
    // *reinterpret_cast<double*>(bufferIn) for your first double number
    // *reinterpret_cast<double*>(bufferIn+8) for your second double number and
    // so on

    CompressMGARD mgard(m_Parameters);
    size_t sizeOut = mgard.InverseOperate(bufferIn + bufferInOffset,
                                          mgardBufferSize, dataOut);
    // TODO: the regular decompressed buffer is in dataOut, with the size of
    // sizeOut. Here you may want to do your magic to change the decompressed
    // data somehow to improve its accuracy :)
    LagrangeOptimizer optim(planeOffset, nodeOffset, planeCount,
        nodeCount, vxCount, vyCount);
    double* doubleData = reinterpret_cast<double*>(dataOut);
    dataOut = optim.setDataFromCharBufferV1(doubleData,
        bufferIn+bufferInOffset+mgardBufferSize, sizeOut);

    return sizeOut;
}

size_t CompressMGARDPlus::InverseOperate(const char *bufferIn,
                                         const size_t sizeIn, char *dataOut)
{
    size_t bufferInOffset = 1; // skip operator type
    const uint8_t bufferVersion =
        GetParameter<uint8_t>(bufferIn, bufferInOffset);
    bufferInOffset += 2;

    if (bufferVersion == 1)
    {
        return DecompressV1(bufferIn, bufferInOffset, sizeIn, dataOut);
    }
    else if (bufferVersion == 2)
    {
        // TODO: if a Version 2 mgard buffer is being implemented, put it here
        // and keep the DecompressV1 routine for backward compatibility
    }
    else
    {
        helper::Throw<std::runtime_error>("Operator", "CompressMGARDPlus",
                                          "InverseOperate",
                                          "invalid mgard buffer version");
    }

    return 0;
}

bool CompressMGARDPlus::IsDataTypeValid(const DataType type) const
{
    if (type == DataType::Double || type == DataType::Float ||
        type == DataType::DoubleComplex || type == DataType::FloatComplex)
    {
        return true;
    }
    return false;
}

} // end namespace compress
} // end namespace core
} // end namespace adios2
