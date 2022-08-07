/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * CompressMGARDPlus.cpp :
 *
 *  Created on: Feb 23, 2022
 *      Author: Jason Wang jason.ruonan.wang@gmail.com
 */

#include "CompressMGARDPlus.h"
#include "CompressMGARD.h"
#include "CompressSZ.h"
#include "CompressZFP.h"
#include "LagrangeOptimizer.hpp"
#include "adios2/core/Engine.h"
#include "adios2/helper/adiosFunctions.h"
#include <cassert>
#include <mpi.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroupMPI.hpp>
#include <c10d/ProcessGroupNCCL.hpp>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

namespace adios2
{
namespace core
{
namespace compress
{

// Custom Dataset class
class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>
{
  private:
    torch::Tensor fdata;
    size_t ds_size;
    size_t nx, ny;

  public:
    CustomDataset(double *data, std::vector<long int> dims)
    {
        assert(dims.size() == 4);
        std::cout << "CustomDataset: dims = " << dims << std::endl;

        nx = (size_t)dims[1];
        ny = (size_t)dims[3];

        // Convert XGC-f data into tensor
        // Shape of the tensor: (nnodes, nx, ny)
        // nnodes: number of xgc mesh nodes
        // nx: f0_nmu + 1
        // ny: 2 * f0_nvp + 1
        fdata = torch::from_blob((void *)data, {dims[0], dims[1], dims[2], dims[3]}, torch::kFloat64)
                    .permute({0, 2, 1, 3})
                    .reshape({-1, dims[1], dims[3]});
        assert(fdata.dim() == 3);

        // Z-score normalization
        // all_planes = (all_planes - mu[:,:,np.newaxis, np.newaxis])/sig[:,:,np.newaxis, np.newaxis
        // (2022/08) jyc: mean and std should be used later for decoding, etc.
        auto mu = fdata.mean({1, 2});
        auto sig = fdata.std({1, 2});
        // std::cout << "CustomDataset: mu.sizes = " << mu.sizes() << std::endl;
        // std::cout << "CustomDataset: sig.sizes = " << sig.sizes() << std::endl;

        fdata = fdata - mu.reshape({-1, 1, 1});
        fdata = fdata / sig.reshape({-1, 1, 1});
        // double-check if mu ~ 0.0 and sig ~ 1.0
        // mu = fdata.mean({1,2});
        // sig = fdata.std({1,2});
        // std::cout << "CustomDataset: mu = " << mu << std::endl;
        // std::cout << "CustomDataset: sig = " << sig << std::endl;

        ds_size = fdata.size(0);
        std::cout << "CustomDataset: fdata.sizes = " << fdata.sizes() << std::endl;
        std::cout << "CustomDataset: ds_size = " << ds_size << std::endl;
        std::cout << "CustomDataset: nx, ny = " << nx << ", " << ny << std::endl;
    };

    torch::data::Example<> get(size_t index) override
    {
        // Return flattened data
        auto dat = fdata.slice(0, index, index + 1).to(torch::kFloat).squeeze();
        dat = dat.reshape(-1);
        // std::cout << "CustomDataset: dat = " << dat.sizes() << std::endl;
        auto label = torch::ones(1);
        return {dat.clone(), label};
    }

    torch::optional<size_t> size() const
    {
        return ds_size;
    };
};

// AE Training
// This is to convert Jaemon's AE pytorch version to CPP
// https://pytorch.org/cppdocs/frontend.html#end-to-end-example
struct Autoencoder : torch::nn::Module
{
    Autoencoder(int input_dim, int latent) : enc1(input_dim, latent), input_dim(input_dim)
    {
        register_module("enc1", enc1);
    }

    torch::Tensor encode(torch::Tensor x)
    {
        x = enc1->forward(x);
        return x;
    }

    torch::Tensor decode(torch::Tensor x)
    {
        x = torch::nn::functional::linear(x, enc1->weight.transpose(0, 1));
        return x;
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = encode(x);
        x = decode(x);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear enc1;
    int input_dim;
};

template <typename DataLoader>
void train(std::shared_ptr<c10d::ProcessGroupNCCL> pg, Autoencoder &model, torch::Device &device, DataLoader &loader,
           torch::optim::Optimizer &optimizer, size_t epoch, size_t dataset_size)
{
    model.train();

    size_t batch_idx = 0;
    for (auto &batch : loader)
    {
        batch_idx++;
        auto data = batch.data.to(device);
        // std::cout << "Batch: data = " << data.sizes() << std::endl;
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::nn::MSELoss()(output, data);
        loss.backward();

        // Averaging the gradients of the parameters in all the processors
        for (auto &param : model.named_parameters())
        {
            std::vector<torch::Tensor> tmp = {param.value().grad()};
            auto work = pg->allreduce(tmp);
            work->wait(kNoTimeout);
        }

        for (auto &param : model.named_parameters())
        {
            param.value().grad().data() = param.value().grad().data() / pg->getSize();
        }
        // update parameter
        optimizer.step();

        std::printf("Train Epoch: %ld [%5ld/%5ld] Loss: %.4g\n", epoch, batch_idx, dataset_size,
                    loss.template item<double>());
    }
}

// Credit: https://github.com/pytorch/pytorch/blob/master/test/cpp/c10d/TestUtils.hpp
std::string tmppath()
{
    // TMPFILE is for manual test execution during which the user will specify
    // the full temp file path using the environmental variable TMPFILE
    const char *tmpfile = getenv("TMPFILE");
    if (tmpfile)
    {
        return std::string(tmpfile);
    }

    const char *tmpdir = getenv("TMPDIR");
    if (tmpdir == nullptr)
    {
        tmpdir = "/tmp";
    }

    // Create template
    std::vector<char> tmp(256);
    auto len = snprintf(tmp.data(), tmp.size(), "%s/testXXXXXX", tmpdir);
    tmp.resize(len);

    // Create temporary file
    auto fd = mkstemp(&tmp[0]);
    if (fd == -1)
    {
        throw std::system_error(errno, std::system_category());
    }
    close(fd);
    return std::string(tmp.data(), tmp.size());
}

struct TemporaryFile
{
    std::string path;

    TemporaryFile()
    {
        path = tmppath();
    }

    ~TemporaryFile()
    {
        unlink(path.c_str());
    }
};

CompressMGARDPlus::CompressMGARDPlus(const Params &parameters)
    : Operator("mgardplus", COMPRESS_MGARDPLUS, "compress", parameters)
{
}

size_t CompressMGARDPlus::Operate(const char *dataIn, const Dims &blockStart, const Dims &blockCount,
                                  const DataType type, char *bufferOut)
{
    int my_rank;
    int comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    std::cout << "rank,size:" << my_rank << " " << comm_size << std::endl;

    // Instantiate LagrangeOptimizer
    LagrangeOptimizer optim;
    // Read ADIOS2 files end, use data for your algorithm
    optim.computeParamsAndQoIs(m_Parameters["meshfile"], blockStart, blockCount,
                               reinterpret_cast<const double *>(dataIn));
    uint8_t compression_method = atoi(m_Parameters["compression_method"].c_str());
    uint8_t pq_yes = atoi(m_Parameters["pq"].c_str());
    size_t bufferOutOffset = 0;
    const uint8_t bufferVersion = pq_yes ? 1 : 2;

    MakeCommonHeader(bufferOut, bufferOutOffset, bufferVersion);
    PutParameter(bufferOut, bufferOutOffset, compression_method);
    PutParameter(bufferOut, bufferOutOffset, optim.getPlaneOffset());
    PutParameter(bufferOut, bufferOutOffset, optim.getNodeOffset());
    size_t offsetForDecompresedData = bufferOutOffset;
    bufferOutOffset += sizeof(size_t);

    // torch::jit::script::Module module;
    // try {
    //     // Deserialize the ScriptModule from a file using torch::jit::load().
    //     module = torch::jit::load("my_ae.pt");
    // }
    // catch (const c10::Error& e) {
    //     std::cerr << "error loading the model\n";
    // }

    std::cout << "dim:" << blockStart.size();
    std::cout << "blockStart:" << blockStart << std::endl;
    std::cout << "blockCount:" << blockCount << std::endl;

    assert(blockStart.size() == 4);
    assert(blockCount.size() == 4);

    int input_dim = blockCount[1] * blockCount[3];
    int latent_dim = 4;
    std::cout << "input_dim, latent_dim: " << input_dim << " " << latent_dim << std::endl;

    // Pytorch DDP
    torch::Device device(torch::kCUDA);

    std::string path = ".filestore";
    remove(path.c_str());
    auto store = c10::make_intrusive<::c10d::FileStore>(path, comm_size);
    c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts = c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();
    auto pg = std::make_shared<::c10d::ProcessGroupNCCL>(store, my_rank, comm_size, std::move(opts));

    // check if pg is working
    auto mytensor = torch::ones(1) * my_rank;
    mytensor = mytensor.to(device);
    std::vector<torch::Tensor> tmp = {mytensor};
    auto work = pg->allreduce(tmp);
    work->wait(kNoTimeout);
    auto expected = (comm_size * (comm_size - 1)) / 2;
    assert (mytensor.item<int>() == expected);

    Autoencoder model(input_dim, latent_dim);
    model.to(device);

    auto dataset = CustomDataset((double *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]})
                       .map(torch::data::transforms::Stack<>());
    const size_t dataset_size = dataset.size().value();
    auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), 128);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3 /*learning rate*/));

    double start = MPI_Wtime();
    for (size_t epoch = 1; epoch <= 100; ++epoch)
    {
        train(pg, model, device, *loader, optimizer, epoch, dataset_size);
    }

    if (my_rank == 0)
    {
        double end = MPI_Wtime();
        printf("Time taken for Training: %f\n", (end - start));
    }

    if (compression_method == 0)
    {
        CompressMGARD mgard(m_Parameters);
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        size_t mgardBufferSize = mgard.Operate(dataIn, blockStart, blockCount, type, bufferOut + bufferOutOffset);

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();
        if (my_rank == 0)
        {
            printf("Time taken for MGARD compression: %f\n", (end - start));
        }
        PutParameter(bufferOut, offsetForDecompresedData, mgardBufferSize);
        std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));

        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        mgard.InverseOperate(bufferOut + bufferOutOffset, mgardBufferSize, tmpDecompressBuffer.data());
        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();
        if (my_rank == 0)
        {
            printf("Time taken for MGARD decompression: %f\n", (end - start));
        }
        optim.computeLagrangeParameters(reinterpret_cast<const double *>(tmpDecompressBuffer.data()), pq_yes);
        bufferOutOffset += mgardBufferSize;
    }
    else if (compression_method == 1)
    {
        CompressSZ sz(m_Parameters);
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        size_t szBufferSize = sz.Operate(dataIn, blockStart, blockCount, type, bufferOut + bufferOutOffset);

        double end = MPI_Wtime();
        if (my_rank == 0)
        {
            printf("Time taken for SZ compression: %f\n", (end - start));
            printf("Size after SZ compression: %zu\n", szBufferSize);
        }
        PutParameter(bufferOut, offsetForDecompresedData, szBufferSize);
        std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));

        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        sz.InverseOperate(bufferOut + bufferOutOffset, szBufferSize, tmpDecompressBuffer.data());
        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();
        if (my_rank == 0)
        {
            printf("Time taken for SZ decompression: %f\n", (end - start));
        }
        optim.computeLagrangeParameters(reinterpret_cast<const double *>(tmpDecompressBuffer.data()), pq_yes);
        bufferOutOffset += szBufferSize;
    }
    else if (compression_method == 2)
    {
        CompressZFP zfp(m_Parameters);
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        size_t zfpBufferSize = zfp.Operate(dataIn, blockStart, blockCount, type, bufferOut + bufferOutOffset);

        double end = MPI_Wtime();
        if (my_rank == 0)
        {
            printf("Time taken for ZFP compression: %f\n", (end - start));
            printf("Size after ZFP compression: %zu\n", zfpBufferSize);
        }
        PutParameter(bufferOut, offsetForDecompresedData, zfpBufferSize);
        std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));

        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        zfp.InverseOperate(bufferOut + bufferOutOffset, zfpBufferSize, tmpDecompressBuffer.data());
        MPI_Barrier(MPI_COMM_WORLD);
        end = MPI_Wtime();
        if (my_rank == 0)
        {
            printf("Time taken for ZFP decompression: %f\n", (end - start));
        }
        optim.computeLagrangeParameters(reinterpret_cast<const double *>(tmpDecompressBuffer.data()), pq_yes);
        bufferOutOffset += zfpBufferSize;
    }
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
    if (bufferVersion == 1)
    {
        size_t ppsize = optim.putResultV1(bufferOut, bufferOutOffset);
        bufferOutOffset += ppsize;
        // printf ("Rank %d PQ indexes %zu\n", my_rank, ppsize);
    }
    else
    {
        size_t ppsize = optim.putResultV2(bufferOut, bufferOutOffset);
        bufferOutOffset += ppsize;
    }

#ifdef UF_DEBUG
    int arraySize =
        optim.getPlaneCount() * optim.getNodeCount() * optim.getVxCount() * optim.getVyCount() * sizeof(double);
    char *dataOut = new char[arraySize];
    size_t result = InverseOperate(bufferOut, bufferOutOffset, dataOut);
#endif
    return bufferOutOffset;
}

Dims CompressMGARDPlus::GetBlockDims(const char *bufferIn, size_t bufferInOffset)
{
    const size_t ndims = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    Dims blockCount(ndims);
    for (size_t i = 0; i < ndims; ++i)
    {
        blockCount[i] = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    }
    return blockCount;
}

size_t CompressMGARDPlus::DecompressV1(const char *bufferIn, size_t bufferInOffset, const size_t sizeIn, char *dataOut)
{
    // Do NOT remove even if the buffer version is updated. Data might be still
    // in lagacy formats. This function must be kept for backward compatibility.
    // If a newer buffer format is implemented, create another function, e.g.
    // DecompressV2 and keep this function for decompressing lagacy data.

    const uint8_t compression_method = GetParameter<uint8_t>(bufferIn, bufferInOffset);
    const size_t planeOffset = GetParameter<size_t>(bufferIn, bufferInOffset);
    const size_t nodeOffset = GetParameter<size_t>(bufferIn, bufferInOffset);

    const size_t mgardBufferSize = GetParameter<size_t>(bufferIn, bufferInOffset);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    size_t planeCount, vxCount, nodeCount, vyCount;
    Dims blockDims = GetBlockDims(bufferIn, bufferInOffset + 4);
    if (blockDims.size() == 3)
    {
        planeCount = 1;
        vxCount = blockDims[0];
        nodeCount = blockDims[1];
        vyCount = blockDims[2];
    }
    else if (blockDims.size() == 4)
    {
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
    size_t sizeOut = mgard.InverseOperate(bufferIn + bufferInOffset, mgardBufferSize, dataOut);
    // TODO: the regular decompressed buffer is in dataOut, with the size of
    // sizeOut. Here you may want to do your magic to change the decompressed
    // data somehow to improve its accuracy :)
    LagrangeOptimizer optim(planeOffset, nodeOffset, planeCount, nodeCount, vxCount, vyCount);
    double *doubleData = reinterpret_cast<double *>(dataOut);
    dataOut = optim.setDataFromCharBufferV1(doubleData, bufferIn + bufferInOffset + mgardBufferSize, sizeOut);

    return sizeOut;
}

size_t CompressMGARDPlus::InverseOperate(const char *bufferIn, const size_t sizeIn, char *dataOut)
{
    size_t bufferInOffset = 1; // skip operator type
    const uint8_t bufferVersion = GetParameter<uint8_t>(bufferIn, bufferInOffset);
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
        helper::Throw<std::runtime_error>("Operator", "CompressMGARDPlus", "InverseOperate",
                                          "invalid mgard buffer version");
    }

    return 0;
}

bool CompressMGARDPlus::IsDataTypeValid(const DataType type) const
{
    if (type == DataType::Double || type == DataType::Float || type == DataType::DoubleComplex ||
        type == DataType::FloatComplex)
    {
        return true;
    }
    return false;
}

} // end namespace compress
} // end namespace core
} // end namespace adios2
