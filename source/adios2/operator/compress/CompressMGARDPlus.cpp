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

#include "KmeansMPI.h"

namespace adios2
{
namespace core
{
namespace compress
{

// Options for pytorch training.
struct Options
{
    size_t batch_size = 1024;
    size_t iterations = 100;
    size_t batch_log_interval = 1000;
    size_t epoch_log_interval = 20;
    torch::DeviceType device = torch::kCPU;
};

// Custom Dataset class
class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>
{
  public:
    torch::Tensor forg;  // Original, un-normalized f-data
    torch::Tensor fdata; // Normalized f-data
    size_t ds_size;
    size_t nx, ny;
    torch::Tensor mu;
    torch::Tensor sig;

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
        assert(dims[0] == 1);
        forg = fdata.detach().clone();

        // Z-score normalization
        // all_planes = (all_planes - mu[:,:,np.newaxis, np.newaxis])/sig[:,:,np.newaxis, np.newaxis
        // (2022/08) jyc: mean and std should be used later for decoding, etc.
        mu = fdata.mean({1, 2});
        sig = fdata.std({1, 2});
        // std::cout << "CustomDataset: mu.sizes = " << mu.sizes() << std::endl;
        // std::cout << "CustomDataset: sig.sizes = " << sig.sizes() << std::endl;

        // auto fdata = forg.clone().detach();
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
struct AutoencoderImpl : torch::nn::Module
{
    AutoencoderImpl(int input_dim, int latent) : enc1(input_dim, latent)
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
};

// see the following why we need: https://discuss.pytorch.org/t/libtorch-how-to-save-model-in-mnist-cpp-example/34234/4
TORCH_MODULE(Autoencoder);

template <typename DataLoader>
void train(std::shared_ptr<c10d::ProcessGroupNCCL> pg, Autoencoder &model, DataLoader &loader,
           torch::optim::Optimizer &optimizer, size_t epoch, size_t dataset_size, Options &options)
{
    model->train();

    size_t batch_idx = 0;
    double batch_loss = 0;
    for (auto &batch : loader)
    {
        batch_idx++;
        auto data = batch.data.to(options.device);
        // std::cout << "Batch: data = " << data.sizes() << std::endl;
        optimizer.zero_grad();
        auto output = model->forward(data);
        auto loss = torch::nn::MSELoss()(output, data);
        loss.backward();

        // Averaging the gradients of the parameters in all the processors
        for (auto &param : model->named_parameters())
        {
            std::vector<torch::Tensor> tmp = {param.value().grad()};
            auto work = pg->allreduce(tmp);
            work->wait(kNoTimeout);
        }

        for (auto &param : model->named_parameters())
        {
            param.value().grad().data() = param.value().grad().data() / pg->getSize();
        }
        // update parameter
        optimizer.step();
        batch_loss += loss.template item<double>();

        if (batch_idx % options.batch_log_interval == 0)
        {
            std::printf("Train Batch: %ld [%5ld/%5ld] Loss: %.4g\n", epoch, batch_idx * batch.data.size(0),
                        dataset_size, loss.template item<double>());
        }
    }

    if (epoch % options.epoch_log_interval == 0)
    {
        std::printf("Train Epoch: %ld Loss: %.4g\n", epoch, batch_loss / batch_idx);
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

// helper lambda function to get parameters
std::string get_param(Params &params, std::string key, std::string def)
{
    if (params.find(key) != params.end())
    {
        return params[key];
    }
    else
    {
        return def;
    }
};

CompressMGARDPlus::CompressMGARDPlus(const Params &parameters)
    : Operator("mgardplus", COMPRESS_MGARDPLUS, "compress", parameters)
{
}

void initClusterCenters(float *&clusters, float *lagarray, int numObjs, int numClusters)
{
    clusters = new float[numClusters];
    assert(clusters != NULL);

    srand(time(NULL));
    float *myNumbers = new float[numClusters];
    std::map<int, int> mymap;
    for (int i = 0; i < numClusters; ++i)
    {
        int index = rand() % numObjs;
        while (mymap.find(index) != mymap.end())
        {
            index = rand() % numObjs;
        }
        clusters[i] = lagarray[index];
        mymap[index] = i;
    }
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
    else if (compression_method == 3)
    {
        std::cout << "dim:" << blockStart.size();
        std::cout << "blockStart:" << blockStart << std::endl;
        std::cout << "blockCount:" << blockCount << std::endl;

        assert(blockStart.size() == 4);
        assert(blockCount.size() == 4);

        int input_dim = blockCount[1] * blockCount[3];
        int latent_dim = 4;
        std::cout << "input_dim, latent_dim: " << input_dim << " " << latent_dim << std::endl;

        // Pytorch
        Options options;
        options.device = torch::kCUDA;
        uint8_t train_yes = atoi(get_param(m_Parameters, "train", "1").c_str());

        Autoencoder model(input_dim, latent_dim);
        model->to(options.device);

        auto ds = CustomDataset((double *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]});
        auto dataset = ds.map(torch::data::transforms::Stack<>());
        const size_t dataset_size = dataset.size().value();
        auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset),
                                                                                              options.batch_size);
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3 /*learning rate*/));

        double start = MPI_Wtime();
        if (train_yes == 1)
        {
            // Pytorch DDP
            std::string path = ".filestore";
            auto store = c10::make_intrusive<::c10d::FileStore>(path, comm_size);
            c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts =
                c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();
            auto pg = std::make_shared<::c10d::ProcessGroupNCCL>(store, my_rank, comm_size, std::move(opts));

            // check if pg is working
            auto mytensor = torch::ones(1) * my_rank;
            mytensor = mytensor.to(options.device);
            std::vector<torch::Tensor> tmp = {mytensor};
            auto work = pg->allreduce(tmp);
            work->wait(kNoTimeout);
            auto expected = (comm_size * (comm_size - 1)) / 2;
            assert(mytensor.item<int>() == expected);

            for (size_t epoch = 1; epoch <= options.iterations; ++epoch)
            {
                train(pg, model, *loader, optimizer, epoch, dataset_size, options);
            }
            torch::save(model, "xgcf_ae_model.pt");
        }
        else
        {
            // (2022/08) jyc: We can restart from the model saved in python.
            const char* mname = get_param(m_Parameters, "ae", "").c_str();
            torch::load(model, mname);
        }

        // Encode
        model->eval();
        std::vector<torch::Tensor> encode_vector;
        for (auto &batch : *loader)
        {
            auto data = batch.data.to(options.device);
            auto _encode = model->encode(data);
            // std::cout << "_encode.sizes = " << _encode.sizes() << std::endl;
            _encode = _encode.to(torch::kCPU);
            encode_vector.push_back(_encode);
        }
        // encodes shape is (nmesh, latent_dim), where nmesh = number of total mesh nodes this process has
        auto encode = torch::cat(encode_vector, 0);
        std::cout << "encode.sizes = " << encode.sizes() << std::endl;

        // Kmean
        *reinterpret_cast<int *>(bufferOut) = latent_dim;
        int offset = sizeof(int);
        int numObjs = encode.size(0);
        for (int i = 0; i < latent_dim; ++i)
        {
            // subselect each column from encode
            float *pq_input = encode.slice(1, i, i + 1).contiguous().data_ptr<float>();
            float pq_threshold = 0.0001;
            int numClusters = 256;
            float *clusters = new float[numClusters];
            initClusterCenters(clusters, pq_input, numObjs, numClusters);
            int *membership = new int[numObjs];
            memset(membership, 0, numObjs * sizeof(int));
            kmeans_float(pq_input, numObjs, numClusters, pq_threshold, membership, clusters);
            // write PQ table
            for (int j = 0; j < numClusters; ++j)
                *reinterpret_cast<float *>(bufferOut + offset + j * sizeof(float)) = clusters[j];
            offset += numClusters * sizeof(float);
            // write indexes to bufferOutput
            for (int j = 0; j < numObjs; ++j)
                *reinterpret_cast<float *>(bufferOut + offset + j * sizeof(int)) = membership[j];
            offset += numObjs * sizeof(int);
            // write to decode input

            // We update encode tensor directly
            for (int j = 0; j < numObjs; ++j)
                encode[j, i] = clusters[membership[j]];
        }
        std::cout << "kmean is done " << std::endl;

        // Decode
        std::vector<torch::Tensor> decode_vector;
        for (int i = 0; i < numObjs; i += options.batch_size)
        {
            auto batch = encode.slice(0, i, i + options.batch_size < numObjs ? i + options.batch_size : numObjs);
            batch = batch.to(options.device);
            auto _decode = model->decode(batch);
            _decode = _decode.to(torch::kCPU);
            decode_vector.push_back(_decode);
        }
        // All of decoded data: shape (nmesh, nx, ny)
        auto decode = torch::cat(decode_vector, 0);
        decode = decode.to(torch::kFloat64).reshape({-1, ds.nx, ds.ny});
        std::cout << "decode.sizes = " << decode.sizes() << std::endl;

        // Un-normalize
        auto mu = ds.mu;
        auto sig = ds.sig;
        std::cout << "mu.sizes = " << mu.sizes() << std::endl;
        std::cout << "sig.sizes = " << sig.sizes() << std::endl;
        decode = decode * sig.reshape({-1, 1, 1});
        decode = decode + mu.reshape({-1, 1, 1});
        // std::cout << "decode.sizes = " << decode.sizes() << std::endl;

        // forg and docode shape: (nmesh, nx, ny)
        auto diff = ds.forg - decode;
        std::cout << "forg min,max = " << ds.forg.min().item<double>() << " " << ds.forg.max().item<double>()
                  << std::endl;
        std::cout << "decode min,max = " << decode.min().item<double>() << " " << decode.max().item<double>()
                  << std::endl;
        std::cout << "diff min,max = " << diff.min().item<double>() << " " << diff.max().item<double>() << std::endl;

        // use MGARD to compress the residuals
        auto perm_diff = diff.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
        std::vector<double> diff_data(perm_diff.data_ptr<double>(), perm_diff.data_ptr<double>() + perm_diff.numel());

        // reserve space in output buffer to store MGARD buffer size
        size_t offsetForDecompresedData = offset;
        offset += sizeof(size_t);
        std::cout << "residual data is ready" << std::endl;

        // apply MGARD operate.
        // Make sure that the shape of the input and the output of MGARD is (1, nmesh, nx, ny)
        CompressMGARD mgard(m_Parameters);
        size_t mgardBufferSize =
            mgard.Operate(reinterpret_cast<char *>(diff_data.data()), blockStart, blockCount, type, bufferOut + offset);
        // size_t mgardBufferSize = mgard.Operate(dataIn, blockStart, blockCount, type, bufferOut + offset);
        std::cout << "mgard is ready" << std::endl;

        PutParameter(bufferOut, offsetForDecompresedData, mgardBufferSize);

        // use MGARD decompress
        std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));
        mgard.InverseOperate(bufferOut + offset, mgardBufferSize, tmpDecompressBuffer.data());
        std::cout << "mgard inverse is ready" << std::endl;
        offset += mgardBufferSize;

        // reconstruct data from the residuals
        auto decompressed_residual_data =
            torch::from_blob((void *)tmpDecompressBuffer.data(),
                             {blockCount[0], blockCount[1], blockCount[2], blockCount[3]}, torch::kFloat64)
                .permute({0, 2, 1, 3});
        auto recon_data = decode.reshape({1, -1, ds.nx, ds.ny}) + decompressed_residual_data;
        // recon_data shape (1, nmesh, nx, ny) and make it contiguous in memory
        recon_data = recon_data.contiguous().cpu();
        std::vector<double> recon_vec(recon_data.data_ptr<double>(),
                                      recon_data.data_ptr<double>() + recon_data.numel());

        // apply post processing
        // recon_vec shape: (1, nmesh, nx, ny)
        optim.computeLagrangeParameters(recon_vec.data(), pq_yes);
        std::cout << "Lagrange is ready" << std::endl;

        // for (auto &batch : *loader)
        // {
        // auto decode = model->decode(encode);
        // std::cout << "decode.sizes = " << decode.sizes() << std::endl;
        // }
        // compute residuals
        // dataIn - decode_vector, after dataIn dimensions have been swapped
        // convert tensor to cpp data is the residual data
        // apply MGARD on residuals
        // apply MGARD inverse operate to get back decompressed residuals
        // add decompressed residuals to decode vector
        // With this total decompressed output, compute Lagrange parameters
        // Figure out PD and QoI errors

        if (my_rank == 0)
        {
            double end = MPI_Wtime();
            printf("Time taken for Training: %f\n", (end - start));
        }
        bufferOutOffset += offset;
    }
    else if (compression_method == 4)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        const char* mname = m_Parameters["ae"].c_str();
        std::cout << "Module name:" << mname;
        torch::jit::script::Module module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load(mname);
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            return -1;
        }
        torch::Device device(torch::kCUDA);
        Options options;
        options.batch_size = 128;
        int latent_dim = 4;
        auto ds = CustomDataset((double *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]});
        auto dataset = ds.map(torch::data::transforms::Stack<>());
        const size_t dataset_size = dataset.size().value();
        auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset),
                                                                                              options.batch_size);
        std::vector<torch::Tensor> encode_vector;
        for (auto &batch : *loader)
        {
            auto data = batch.data.to(torch::kCUDA);
            auto _encode = module.run_method("encode", data).toTensor();
            // std::cout << "_encode.sizes = " << _encode.sizes() << std::endl;
            _encode = _encode.to(torch::kCPU);
            encode_vector.push_back(_encode);
        }
        // encodes shape is (nmesh, latent_dim), where nmesh = number of total mesh nodes this process has
        auto encode = torch::cat(encode_vector, 0);
        std::cout << "encode.sizes = " << encode.sizes() << std::endl;

        // Kmean
        *reinterpret_cast<int *>(bufferOut) = latent_dim;
        int offset = sizeof(int);
        int numObjs = encode.size(0);
        for (int i = 0; i < latent_dim; ++i)
        {
            // subselect each column from encode
            float *pq_input = encode.slice(1, i, i + 1).contiguous().data_ptr<float>();
            float pq_threshold = 0.0001;
            int numClusters = 256;
            float *clusters = new float[numClusters];
            initClusterCenters(clusters, pq_input, numObjs, numClusters);
            int *membership = new int[numObjs];
            memset(membership, 0, numObjs * sizeof(int));
            kmeans_float(pq_input, numObjs, numClusters, pq_threshold, membership, clusters);
            // write PQ table
            for (int j = 0; j < numClusters; ++j)
                *reinterpret_cast<float *>(bufferOut + offset + j * sizeof(float)) = clusters[j];
            offset += numClusters * sizeof(float);
            // write indexes to bufferOutput
            for (int j = 0; j < numObjs; ++j)
                *reinterpret_cast<float *>(bufferOut + offset + j * sizeof(int)) = membership[j];
            offset += numObjs * sizeof(int);
            // write to decode input

            // We update encode tensor directly
            for (int j = 0; j < numObjs; ++j)
                encode[j, i] = clusters[membership[j]];
        }
        std::cout << "kmean is done " << std::endl;

        // Decode
        std::vector<torch::Tensor> decode_vector;
        for (int i = 0; i < numObjs; i += options.batch_size)
        {
            auto batch = encode.slice(0, i, i + options.batch_size < numObjs ? i + options.batch_size : numObjs);
            batch = batch.to(torch::kCUDA);
            auto _decode = module.run_method("decode", batch).toTensor();
            _decode = _decode.to(torch::kCPU);
            decode_vector.push_back(_decode);
        }
        // All of decoded data: shape (nmesh, nx, ny)
        auto decode = torch::cat(decode_vector, 0);
        decode = decode.to(torch::kFloat64).reshape({-1, ds.nx, ds.ny});
        std::cout << "decode.sizes = " << decode.sizes() << std::endl;

        // Un-normalize
        auto mu = ds.mu;
        auto sig = ds.sig;
        std::cout << "mu.sizes = " << mu.sizes() << std::endl;
        std::cout << "sig.sizes = " << sig.sizes() << std::endl;
        decode = decode * sig.reshape({-1, 1, 1});
        decode = decode + mu.reshape({-1, 1, 1});
        // std::cout << "decode.sizes = " << decode.sizes() << std::endl;

        // forg and docode shape: (nmesh, nx, ny)
        auto diff = ds.forg - decode;
        std::cout << "forg min,max = " << ds.forg.min().item<double>() << " " << ds.forg.max().item<double>()
                  << std::endl;
        std::cout << "decode min,max = " << decode.min().item<double>() << " " << decode.max().item<double>()
                  << std::endl;
        std::cout << "diff min,max = " << diff.min().item<double>() << " " << diff.max().item<double>() << std::endl;

        // use MGARD to compress the residuals
        auto perm_diff = diff.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
        std::vector<double> diff_data(perm_diff.data_ptr<double>(), perm_diff.data_ptr<double>() + perm_diff.numel());

        // reserve space in output buffer to store MGARD buffer size
        size_t offsetForDecompresedData = offset;
        offset += sizeof(size_t);
        std::cout << "residual data is ready" << std::endl;

        // apply MGARD operate.
        // Make sure that the shape of the input and the output of MGARD is (1, nmesh, nx, ny)
        CompressMGARD mgard(m_Parameters);
        size_t mgardBufferSize =
            mgard.Operate(reinterpret_cast<char *>(diff_data.data()), blockStart, blockCount, type, bufferOut + offset);
        // size_t mgardBufferSize = mgard.Operate(dataIn, blockStart, blockCount, type, bufferOut + offset);
        std::cout << "mgard is ready" << std::endl;

        PutParameter(bufferOut, offsetForDecompresedData, mgardBufferSize);

        // use MGARD decompress
        std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));
        mgard.InverseOperate(bufferOut + offset, mgardBufferSize, tmpDecompressBuffer.data());
        std::cout << "mgard inverse is ready" << std::endl;
        offset += mgardBufferSize;

        // reconstruct data from the residuals
        auto decompressed_residual_data =
            torch::from_blob((void *)tmpDecompressBuffer.data(),
                             {blockCount[0], blockCount[1], blockCount[2], blockCount[3]}, torch::kFloat64)
                .permute({0, 2, 1, 3});
        auto recon_data = decode.reshape({1, -1, ds.nx, ds.ny}) + decompressed_residual_data;
        // recon_data shape (1, nmesh, nx, ny) and make it contiguous in memory
        recon_data = recon_data.contiguous().cpu();
        std::vector<double> recon_vec(recon_data.data_ptr<double>(),
                                      recon_data.data_ptr<double>() + recon_data.numel());

        // apply post processing
        // recon_vec shape: (1, nmesh, nx, ny)
        optim.computeLagrangeParameters(recon_vec.data(), pq_yes);
        std::cout << "Lagrange is ready" << std::endl;

        // for (auto &batch : *loader)
        // {
        // auto decode = model->decode(encode);
        // std::cout << "decode.sizes = " << decode.sizes() << std::endl;
        // }
        // compute residuals
        // dataIn - decode_vector, after dataIn dimensions have been swapped
        // convert tensor to cpp data is the residual data
        // apply MGARD on residuals
        // apply MGARD inverse operate to get back decompressed residuals
        // add decompressed residuals to decode vector
        // With this total decompressed output, compute Lagrange parameters
        // Figure out PD and QoI errors

        if (my_rank == 0)
        {
            double end = MPI_Wtime();
            printf("Time taken for Training: %f\n", (end - start));
        }
        bufferOutOffset += offset;
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
