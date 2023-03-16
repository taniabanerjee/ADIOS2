/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * CompressMGARDPlus.cpp :
 *
 *  Created on: Feb 23, 2022
 *      Author: Jason Wang jason.ruonan.wang@gmail.com
 */

#include "CompressMGARD.h"
#include "CompressMGARDPlus.h"
#include "CompressSZ.h"
#include "CompressZFP.h"
#include "LagrangeTorchL2.hpp"
#include "adios2/core/Engine.h"
#include "adios2/helper/adiosFunctions.h"
#include <cassert>
#include <mpi.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>

#include <cstdio>
#include <iostream>
#include <string>

#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroupMPI.hpp>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/TCPStore.hpp>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <gptl.h>
#include <gptlmpi.h>

#include "KmeansMPI.h"

#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>

template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

namespace adios2
{
namespace core
{
namespace compress
{

// Options for pytorch training.
struct Options
{
    size_t batch_size = 128;
    size_t iterations = 100;
    size_t batch_log_interval = 1000000000;
    size_t epoch_log_interval = 20;
    torch::DeviceType device = torch::kCUDA;
    size_t batch_max = 0;
    bool use_ddp = 0;
    int training_paradigm = 0;
    double learning_rate = 1e-3;
};

constexpr int defaultTimeout = 60;

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

    CustomDataset(double *data, std::vector<long int> dims, int paradigm = 0)
    {
        assert(dims.size() == 4);
        // std::cout << "CustomDataset: dims = " << dims << std::endl;

        nx = (size_t)dims[1];
        ny = (size_t)dims[3];

        // Convert XGC-f data into tensor
        // Shape of the tensor: (nnodes, nx, ny)
        // nnodes: number of xgc mesh nodes
        // nx: f0_nmu + 1
        // ny: 2 * f0_nvp + 1
        if (paradigm == 0 || paradigm == 5) {
            fdata = torch::from_blob((void *)data, {dims[0], dims[1], dims[2], dims[3]}, torch::kFloat64)
                    .permute({0, 2, 1, 3})
                    .reshape({-1, dims[1], dims[3]});
            // std::cout << "fdata " << fdata.sizes() << " " << paradigm << " " << dims[0] << std::endl;
        }
        else if (paradigm == 1) {
            fdata = torch::from_blob((void *)data, {1, dims[1], dims[2], dims[3]}, torch::kFloat64)
                    .permute({0, 2, 1, 3})
                    .reshape({-1, dims[1], dims[3]});
            // std::cout << "fdata " << fdata.sizes() << " " << paradigm << " " << dims[0] << std::endl;
        }
        else if (paradigm == 2 || paradigm == 3 || paradigm == 4) {
            auto t = torch::from_blob((void *)data, {1, dims[1], dims[2], dims[3]}, torch::kFloat64)
                    .permute({0, 2, 1, 3})
                    .reshape({-1, dims[1], dims[3]});
            auto idx = at::randperm(at::size(t, 0));
            int end = (int) at::size(t,0) * 0.25 * (paradigm-1);
            auto idx_25 = idx.slice(0, 0, end);
            // std::cout << "idx " << idx.sizes() << " " << idx_25.sizes() << std::endl;
            using namespace torch::indexing;
            fdata = t.index({idx_25, Slice(None), Slice(None)});
        }
        else if (paradigm == 6) {
            auto t = torch::from_blob((void *)data, {dims[0], dims[1], dims[2], dims[3]}, torch::kFloat64)
                    .permute({0, 2, 1, 3})
                    .reshape({-1, dims[1], dims[3]});
            auto idx = at::randperm(at::size(t, 0));
            int end = dims[2];
            auto idx_25 = idx.slice(0, 0, end);
            // std::cout << "idx " << idx.sizes() << " " << idx_25.sizes() << std::endl;
            using namespace torch::indexing;
            fdata = t.index({idx_25, Slice(None), Slice(None)});
            // std::cout << "fdata " << fdata.sizes() << " " << paradigm << " " << dims[0] << std::endl;
        }
        else if (paradigm == 7 || paradigm == 8 || paradigm == 9 || paradigm == 10) {
            auto t = torch::from_blob((void *)data, {dims[0], dims[1], dims[2], dims[3]}, torch::kFloat64)
                    .permute({0, 2, 1, 3})
                    .reshape({-1, dims[1], dims[3]});
            at::TensorOptions opt = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
            auto idx = at::randint(8, {dims[2]}, opt);
            const at::Scalar start = 0;
            const at::Scalar last = dims[2]-1;

            auto dim2 = at::range(start, last, opt);
            // std::vector <at::Tensor> tensors;
            // tensors.push_back(idx);
            // tensors.push_back(dim2);
            // auto combine_idx = at::stack(tensors, 1);
            auto combine_idx = idx * dims[2] + dim2;
            using namespace torch::indexing;
            if (paradigm == 7) {
                fdata = t.index({combine_idx, Slice(None), Slice(None)});
                // std::cout << "fdata " << fdata.sizes() << " " << paradigm << " " << dims[0] << std::endl;
            }
            else {
                auto randidx = at::randperm(at::size(combine_idx, 0));
                int end = (int) at::size(combine_idx,0) * 0.25 * (paradigm-7);
                auto idx_25 = randidx.slice(0, 0, end);
                auto cidx_25 = combine_idx.index({idx_25});
                // std::cout << "cidx_25 " << cidx_25 << std::endl;
                fdata = t.index({cidx_25, Slice(None), Slice(None)});
                // std::cout << "fdata " << fdata.sizes() << " " << paradigm << " " << dims[0] << std::endl;
            }
        }
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
        // std::cout << "CustomDataset: fdata.sizes = " << fdata.sizes() << std::endl;
        // std::cout << "CustomDataset: ds_size = " << ds_size << std::endl;
        // std::cout << "CustomDataset: nx, ny = " << nx << ", " << ny << std::endl;
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

void waitWork(std::shared_ptr<c10d::ProcessGroupNCCL> pg,
              std::vector<c10::intrusive_ptr<c10d::ProcessGroup::Work>> works)
{
    for (auto &work : works)
    {
        try
        {
            work->wait();
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Exception received: " << ex.what() << std::endl;
            // pg->abort();
        }
    }
}

template <typename DataLoader>
void train(std::shared_ptr<c10d::ProcessGroupNCCL> pg, Autoencoder &model, DataLoader &loader,
           torch::optim::Optimizer &optimizer, size_t epoch, size_t dataset_size, Options &options)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    model->train();

    size_t batch_idx = 0;
    double batch_loss = 0;
    int append = 0;

    for (auto &batch : loader)
    {
        batch_idx++;
        auto data = batch.data.to(options.device);
        optimizer.zero_grad();
        auto output = model->forward(data);
        auto loss = torch::nn::MSELoss()(output, data);
        loss.backward();

        if (options.use_ddp)
        {
            // Averaging the gradients of the parameters in all the processors
            std::vector<c10::intrusive_ptr<::c10d::ProcessGroup::Work>> works;
            for (auto &param : model->named_parameters())
            {
                std::vector<torch::Tensor> tmp = {param.value().grad()};
                auto work = pg->allreduce(tmp);
                works.push_back(std::move(work));
            }

            waitWork(pg, works);

            for (auto &param : model->named_parameters())
            {
                param.value().grad().data() = param.value().grad().data() / pg->getSize();
            }
        }

        // update parameter
        optimizer.step();
        batch_loss += loss.template item<double>();

        if (batch_idx % options.batch_log_interval == 0)
        {
            std::printf("Train Batch: %ld [%5ld/%5ld] Loss: %.4g\n", epoch, batch_idx * batch.data.size(0),
                        dataset_size, loss.template item<double>());
        }

        if (batch_idx >= options.batch_max)
        {
            break;
        }
    }
    if (epoch % options.epoch_log_interval == 0)
    {
        int comm_size;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        double myLoss = batch_loss / batch_idx;
        double minLoss, maxLoss, sumLoss, sumCount;
        MPI_Allreduce(&myLoss, &minLoss, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&myLoss, &maxLoss, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&myLoss, &sumLoss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        std::printf("Train Epoch: %ld Loss: %.4g, Min: %.8g, Max: %.8g, Sum: %.8g, Avg: %.8g\n", epoch, batch_loss / batch_idx, minLoss, maxLoss, sumLoss, float(sumLoss/comm_size));
        // std::printf("Train Epoch: %ld Loss: %.4g\n", epoch, batch_loss / batch_idx);
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
    int i;
    clusters = new float[numClusters];
    if (numClusters < numObjs) {
        std::vector <int> numbers;
        for (i=0; i<numObjs; ++i) {
            numbers.push_back(i);
        }
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(numbers.begin(), numbers.end(), std::default_random_engine(seed));
        for (i = 0; i < numClusters; ++i)
        {
            clusters[i] = lagarray[numbers[i]];
        }
    }
    else if (numObjs == numClusters) {
        for (i = 0; i < numClusters; ++i)
        {
            clusters[i] = lagarray[i];
        }
    }
    else {
        for (i = 0; i < numObjs; ++i)
        {
            clusters[i] = lagarray[i];
        }
        for (i = numObjs; i < numClusters; ++i)
        {
            clusters[i] = -99999999;
        }
    }
}

void dump(void *data, std::vector<long unsigned int> dims, char *vname, int rank)
{
    char fname[255];
    sprintf(fname, "tensor-%s-%d.pt", vname, rank);
    torch::Tensor ten;
    if (dims.size() == 3)
    {
        ten = torch::from_blob((void *)data, {dims[0], dims[1], dims[2]}, torch::kFloat64);
    }
    else if (dims.size() == 4)
    {
        ten = torch::from_blob((void *)data, {dims[0], dims[1], dims[2], dims[3]}, torch::kFloat64);
    }
    else
    {
        throw std::invalid_argument("The size of dimension is not supported yet");
    }
    torch::save(ten, fname);
}

void dump(torch::Tensor ten, char *vname, int rank)
{
    char fname[255];
    sprintf(fname, "tensor-%s-%d.pt", vname, rank);
    torch::save(ten, fname);
}

void save_model(Autoencoder &model, Options &options, int my_rank)
{
    int eligible = my_rank;

    if (options.use_ddp)
        eligible = 0;

    if (my_rank == eligible)
    {
        // std::string fname = "xgcf_ae_model_0.pt";
        std::string fname = "xgcf_ae_model_" + std::to_string(my_rank) + ".pt";
        torch::save(model, fname.c_str());
        if (my_rank == 0) std::cout << "Save model: " << fname << std::endl;
    }
}

void load_model(Autoencoder &model, Options &options, int my_rank)
{
    int eligible = my_rank;

    if (options.use_ddp)
        eligible = 0;

    std::string fname = "xgcf_ae_model_" + std::to_string(eligible) + ".pt";
    torch::load(model, fname.c_str());
    if (my_rank == 0) std::cout << "Load model: " << fname << std::endl;
}


double CompressMGARDPlus::binarySearchEB(double lowereb, double uppereb, at::Tensor &orig, at::Tensor &decode,
at::Tensor &perm_diff, Dims blockStart, Dims blockCount, const DataType type, char *bufferOut, double vx, double vy, double pd_omax_b, double pd_omin_b, double targetE)
{
    // double targetE = 0.09;
    double accuracy = 0.001;
    double e = getPDError(uppereb, orig, decode, perm_diff, blockStart, blockCount, type, bufferOut, vx, vy, pd_omax_b, pd_omin_b);
    if (e < targetE) {
        std::cout << "Upper error bound " << uppereb << " error " << e << std::endl;
        return uppereb;
    }
    e = getPDError(lowereb, orig, decode, perm_diff, blockStart, blockCount, type, bufferOut, vx, vy, pd_omax_b, pd_omin_b);
    if (e > targetE) {
        std::cout << "Lower error bound " << lowereb << " error " << e << std::endl;
        return lowereb;
    }
    int partitions = 10000;
    double error[partitions];
    int i;
    error[0] = lowereb;
    error[partitions-1] = uppereb;
    int low = 0, high = partitions-1, mid;
    for (i=1; i<partitions-1; ++i) error[i] = lowereb + i*uppereb/partitions;
    while (low <= high) {
        mid = low + (high-low)/2;
        e = getPDError(error[mid], orig, decode, perm_diff, blockStart, blockCount, type, bufferOut, vx, vy, pd_omax_b, pd_omin_b);
        // std::cout << "Mid " << mid << " error bound " << error[mid] << " error " << e << std::endl;
        if (e < targetE) {
            low = mid + 1;
        }
        else if (e > targetE) {
            high = mid - 1;
        }
        else if (abs(e-targetE) < accuracy) {
            std::cout << "Return middle error bound " << mid << " " << error[mid] << " error " << e << std::endl;
            return error[mid];
        }
    }
    std::cout << "Return lowest error bound " << mid << " " << error[mid] << std::endl;
    return error[mid];
}

double CompressMGARDPlus::getPDError(double eb, at::Tensor &orig, at::Tensor &decode, at::Tensor &perm_diff, Dims blockStart, Dims blockCount, const DataType type, char *bufferOut, double vx, double vy, double pd_omax_b, double pd_omin_b)
{
    double pderror = 0;
    m_Parameters["tolerance"] = std::to_string(eb).c_str();
    CompressMGARD mgard(m_Parameters);
    std::vector<double> diff_data(perm_diff.data_ptr<double>(), perm_diff.data_ptr<double>() + perm_diff.numel());
    // std::cout << "Block Start " << blockStart << " " << blockCount << std::endl;
    size_t mgardBufferSize = mgard.Operate(reinterpret_cast<char *>(diff_data.data()), blockStart, blockCount, type, bufferOut);
    std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));
    mgard.InverseOperate(bufferOut, mgardBufferSize, tmpDecompressBuffer.data());
    // apply MGARD operate.
    auto decompressed_residual_data =
        torch::from_blob((void *)tmpDecompressBuffer.data(), {blockCount[0], blockCount[1], blockCount[2], blockCount[3]},
                         torch::kFloat64);
    // pderror = at::sqrt(at::divide(at::pow((orig-decode-decompressed_residual_data),2).sum(), at::Scalar(perm_diff.numel()))).item().to<double>()/perm_diff.max().item().to<double>();
    auto diff = orig-decode-decompressed_residual_data;
    pderror = at::max(at::divide(at::sqrt(at::divide(at::pow(diff, 2).sum({1, 2}),at::Scalar(vx*vy))), at::Scalar(pd_omax_b - pd_omin_b))).item().to<double>();
#if 0
    // std::cout << "decompressed sizes " << decompressed_residual_data.sizes() << std::endl;
    if (resNodes > data_ceil)
    {
        recon_data = decode.reshape({1, -1, ds.nx, ds.ny}) + decompressed_residual_data;
    }
    else
    {
        using namespace torch::indexing;
        auto res = at::zeros({blockCount[0], blockCount[2], blockCount[1], blockCount[3]}, torch::kFloat64);
        // std::cout << "res sizes 1 " << res.sizes() << std::endl;
        res.index_put_({Slice(None), nrmse_index[0], Slice(None), Slice(None)}, decompressed_residual_data);
        // std::cout << "res sizes 2 " << res.sizes() << std::endl;
        recon_data = decode.reshape({1, -1, ds.nx, ds.ny}) + res;
    }
    recon_data = recon_data.contiguous().cpu();
    std::vector<double> recon_vec(recon_data.data_ptr<double>(),
                                  recon_data.data_ptr<double>() + recon_data.numel());

    // apply post processing
    // recon_vec shape: (1, nmesh, nx, ny)
    // // Return PD
    double pderror = optim.computeLagrangeParametersE(recon_vec.data(), blockCount);
#endif
    return pderror;
}

static int _isfirst = 1;

size_t CompressMGARDPlus::Operate(const char *dataIn, const Dims &blockStart, const Dims &blockCount,
                                  const DataType type, char *bufferOut)
{
    int my_rank;
    int comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (_isfirst == 1)
    {
        GPTLinitialize();
        _isfirst = 0;
    }
#if 0
    // Disabling this print temporarily to check swamping of output
    std::cout << "rank,size:" << my_rank << " " << comm_size << std::endl;

    std::cout << "Parameters:" << std::endl;
    for (auto const &x : m_Parameters)
    {
        std::cout << "  " << x.first << ": " << x.second << std::endl;
    }
#endif

    // Instantiate LagrangeTorchL2
    LagrangeTorchL2 optim(m_Parameters["species"].c_str(), m_Parameters["prec"].c_str());
    // Read ADIOS2 files end, use data for your algorithm
    optim.computeParamsAndQoIs(m_Parameters["meshfile"], blockStart, blockCount,
                               reinterpret_cast<const double *>(dataIn));
    uint8_t compression_method = atoi(m_Parameters["compression_method"].c_str());
    uint8_t pq_yes = atoi(m_Parameters["pq"].c_str());
    size_t bufferOutOffset = 0;
    // const uint8_t bufferVersion = pq_yes ? 1 : 2;
    const uint8_t bufferVersion = 1;

    // Pytorch options
    Options options;
    options.device = torch::kCUDA;
    options.use_ddp = atoi(get_param(m_Parameters, "use_ddp", "0").c_str());
    options.batch_size = atoi(get_param(m_Parameters, "batch_size", "128").c_str());
    options.iterations = atoi(get_param(m_Parameters, "nepoch", "100").c_str());
    options.training_paradigm = atoi(get_param(m_Parameters, "decomp", "0").c_str());
    options.learning_rate = atof(get_param(m_Parameters, "lr", "1e-3").c_str());
    int train_yes = atoi(get_param(m_Parameters, "train", "1").c_str());
    int use_pretrain = atoi(get_param(m_Parameters, "use_pretrain", "0").c_str());
    float ae_thresh = atof(get_param(m_Parameters, "ae_thresh", "0.001").c_str());
    float target_e = atof(get_param(m_Parameters, "res_thresh", "0.09").c_str());
    int pqbits = atoi(get_param(m_Parameters, "pqbits", "8").c_str());
    double leb = std::stod(get_param(m_Parameters, "leb", "-1"));
    double ueb = std::stod(get_param(m_Parameters, "ueb", "-1"));
    // std::cout << "Train " << train_yes << " Pre-train " << use_pretrain << " AE threshold " << ae_thresh << " leb " << leb << " ueb " << ueb << " decomp " << options.training_paradigm << std::endl;

    MakeCommonHeader(bufferOut, bufferOutOffset, bufferVersion);
    PutParameter(bufferOut, bufferOutOffset, optim.getSpecies());
    PutParameter(bufferOut, bufferOutOffset, optim.getPrecision());
    PutParameter(bufferOut, bufferOutOffset, compression_method);
    PutParameter(bufferOut, bufferOutOffset, optim.getPlaneOffset());
    PutParameter(bufferOut, bufferOutOffset, optim.getNodeOffset());
    size_t offsetForDecompresedData = bufferOutOffset;
    bufferOutOffset += sizeof(size_t);

    GPTLstart("total");
    if (compression_method == 0)
    {
#if 0
        if (leb > 0 && ueb > 0) {
            size_t offset = 0;
            auto perm_diff = torch::from_blob((void *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]}, torch::kFloat64).to(torch::kCUDA);
            m_Parameters["tolerance"] = std::to_string(binarySearchEB(leb, ueb, perm_diff, {0, 0, 0, 0}, blockCount, type, bufferOut+offset, vx, vy, pd_omax_b, pd_omin_b)).c_str();
            // std::cout << "came here 1" << std::endl;
        }
#endif
        // m_Parameters["tolerance"] = std::to_string(1e17+my_rank * 1e15);
        // std::cout << "tolerance " << m_Parameters["tolerance"] << std::endl;
        CompressMGARD mgard(m_Parameters);
        // MPI_Barrier(MPI_COMM_WORLD);
        // double start = MPI_Wtime();
        GPTLstart("mgard");
        size_t mgardBufferSize = mgard.Operate(dataIn, blockStart, blockCount, type, bufferOut + bufferOutOffset);
        GPTLstop("mgard");
        std::ofstream myfile;
        std::string fname = "mgard-size-iter_" + std::to_string(my_rank) + ".txt";
        myfile.open(fname.c_str());
        myfile << my_rank << " - mgard size:" << mgardBufferSize << " :num images " << blockCount[2] << std::endl;
        myfile.close();

        // MPI_Barrier(MPI_COMM_WORLD);
        // double end = MPI_Wtime();
        // if (my_rank == 0)
        // {
        // printf("%d Time taken for MGARD compression: %f\n", optim.getSpecies(), (end - start));
        // }
        PutParameter(bufferOut, offsetForDecompresedData, mgardBufferSize);
        std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));

        // MPI_Barrier(MPI_COMM_WORLD);
        // start = MPI_Wtime();
        GPTLstart("mgard decompress");
        mgard.InverseOperate(bufferOut + bufferOutOffset, mgardBufferSize, tmpDecompressBuffer.data());
        GPTLstop("mgard decompress");
        // MPI_Barrier(MPI_COMM_WORLD);
        // end = MPI_Wtime();
        // if (my_rank == 0)
        // {
        // printf("%d Time taken for MGARD decompression: %f\n", optim.getSpecies(), (end - start));
        // }
        GPTLstart("Lagrange");
        optim.computeLagrangeParameters(reinterpret_cast<const double *>(tmpDecompressBuffer.data()), blockCount);
        GPTLstop("Lagrange");
        bufferOutOffset += mgardBufferSize;

        // dump((void *)dataIn, blockCount, "forg", my_rank);
        // dump((void *)tmpDecompressBuffer.data(), blockCount, "fbar", my_rank);
        // double nbytes_org = (double) (blockCount[0] * blockCount[1] * blockCount[2] * blockCount[3] * 8);
        // double nbytes_compressed = (double) mgardBufferSize;
        // printf("%d MGARD compression org, compressed, ratio: %g %g %g\n", my_rank, nbytes_org, nbytes_compressed,
        // nbytes_org/nbytes_compressed);
    }
    else if (compression_method == 1)
    {
        CompressSZ sz(m_Parameters);
        MPI_Barrier(MPI_COMM_WORLD);
        GPTLstart("sz");
        size_t szBufferSize = sz.Operate(dataIn, blockStart, blockCount, type, bufferOut + bufferOutOffset);
        GPTLstop("sz");

        std::ofstream myfile;
        std::string fname = "sz-size-iter_" + std::to_string(my_rank) + ".txt";
        myfile.open(fname.c_str());
        myfile << my_rank << " - sz size:" << szBufferSize << " :num images " << blockCount[2] << std::endl;
        myfile.close();
        PutParameter(bufferOut, offsetForDecompresedData, szBufferSize);
        std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));

        GPTLstart("sz-decomp");
        sz.InverseOperate(bufferOut + bufferOutOffset, szBufferSize, tmpDecompressBuffer.data());
        GPTLstop("sz-decomp");
        GPTLstart("Lagrange");
        optim.computeLagrangeParameters(reinterpret_cast<const double *>(tmpDecompressBuffer.data()), blockCount);
        GPTLstop("Lagrange");
        bufferOutOffset += szBufferSize;
    }
    else if (compression_method == 2)
    {
        m_Parameters["prec"] = "";
        CompressZFP zfp(m_Parameters);
        GPTLstart("zfp");
        size_t zfpBufferSize = zfp.Operate(dataIn, blockStart, blockCount, type, bufferOut + bufferOutOffset);
        GPTLstop("zfp");

        std::ofstream myfile;
        std::string fname = "zfp-size-iter_" + std::to_string(my_rank) + ".txt";
        myfile.open(fname.c_str());
        myfile << my_rank << " - zfp size:" << zfpBufferSize << " :num images " << blockCount[2] << std::endl;
        myfile.close();

        PutParameter(bufferOut, offsetForDecompresedData, zfpBufferSize);
        std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type)));

        GPTLstart("zfp decomp");
        zfp.InverseOperate(bufferOut + bufferOutOffset, zfpBufferSize, tmpDecompressBuffer.data());
        GPTLstop("zfp decomp");
        GPTLstart("Lagrange");
        optim.computeLagrangeParameters(reinterpret_cast<const double *>(tmpDecompressBuffer.data()), blockCount);
        GPTLstop("Lagrange");
        bufferOutOffset += zfpBufferSize;
    }
    else if (compression_method == 3)
    {
        // std::cout << "dim:" << blockStart.size();
        // std::cout << "blockStart:" << blockStart << std::endl;
        // std::cout << "blockCount:" << blockCount << std::endl;

        assert(blockStart.size() == 4);
        assert(blockCount.size() == 4);

        int input_dim = blockCount[1] * blockCount[3];
        int latent_dim = atoi(get_param(m_Parameters, "latent_dim", "5").c_str());
        ;
        // std::cout << "input_dim, latent_dim: " << input_dim << " " << latent_dim << std::endl;

        Autoencoder model(input_dim, latent_dim);
        model->to(options.device);

        GPTLstart("prep");
        // double start = MPI_Wtime();
        // std::cout << "Is Column decomposition " << options.training_paradigm  << std::endl;
        // std::printf("%d: CustomDataset: %d %d %d %d\n", my_rank, blockCount[0], blockCount[1], blockCount[2], blockCount[3]);
        // std::printf("%d: training_paradigm: %d\n", my_rank, options.training_paradigm);
        auto train_ds = CustomDataset((double *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]}, options.training_paradigm );
        std::vector <torch::data::datasets::MapDataset<adios2::core::compress::CustomDataset, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >> train_datasets;
        auto train_dataset = train_ds.map(torch::data::transforms::Stack<>());
        train_datasets.push_back(train_dataset);
        // std::cout << "decltype(train_dataset) is " << type_name<decltype(train_dataset)>() << std::endl;
        const size_t train_dataset_size = train_dataset.size().value();
        // std::cout << "Train dataset size " << train_dataset_size << std::endl;
        std::vector <std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<adios2::core::compress::CustomDataset, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<adios2::core::compress::CustomDataset, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler> > >*> train_loaders;
        auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), options.batch_size);
        train_loaders.push_back(&train_loader);
        // std::cout << "decltype(train_loader) is " << type_name<decltype(train_loader)>() << std::endl;


        auto ds = CustomDataset((double *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]});
        auto dataset = ds.map(torch::data::transforms::Stack<>());
        const size_t dataset_size = dataset.size().value();
        auto encode_loader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), options.batch_size);
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(options.learning_rate /*learning rate*/));
        // if (my_rank == 0)
        // {
        //     double end = MPI_Wtime();
        //     printf("Time taken for Prep: %f\n", (end - start));
        // }
        GPTLstop("prep");

        GPTLstart("training");
        // start = MPI_Wtime();
        if (train_yes == 1)
        {
            std::shared_ptr<c10d::ProcessGroupNCCL> pg;
            if (options.use_ddp)
            {
                // Pytorch DDP
                // std::string path = std::tmpnam(nullptr);
                // auto store = c10::make_intrusive<c10d::FileStore>(path, 0);
                char *MASTER_ADDR = std::getenv("MASTER_ADDR");
                if (MASTER_ADDR == NULL)
                {
                    MASTER_ADDR = "127.0.0.1";
                }
                char *MASTER_PORT = std::getenv("MASTER_PORT");
                if (MASTER_PORT == NULL)
                {
                    MASTER_PORT = "29500";
                }
                auto store = c10::make_intrusive<c10d::TCPStore>(MASTER_ADDR, atoi(MASTER_PORT),
                                                                 /* numWorkers */ 0,
                                                                 /* isServer */ my_rank == 0 ? true : false,
                                                                 std::chrono::seconds(defaultTimeout),
                                                                 /* wait */ false);
                auto opts = c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();
                std::cout << "TCPStore: " << MASTER_ADDR << " " << MASTER_PORT << std::endl;
                pg = std::make_shared<c10d::ProcessGroupNCCL>(store, my_rank, comm_size, std::move(opts));

                // check if pg is working
                auto mytensor = torch::ones(1) * my_rank;
                mytensor = mytensor.to(options.device);
                std::vector<torch::Tensor> tmp = {mytensor};
                auto work = pg->allreduce(tmp);
                work->wait();
                auto expected = (comm_size * (comm_size - 1)) / 2;
                assert(mytensor.item<int>() == expected);
            }

            // The number of iteration should be same for all processes due to sync
            int nbatch = dataset_size / options.batch_size + 1;
            MPI_Allreduce(MPI_IN_PLACE, &nbatch, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            // std::cout << "Loader size: " << dataset_size / options.batch_size + 1 << " " << nbatch << std::endl;
            options.batch_max = nbatch;

            if (use_pretrain)
            {
                load_model(model, options, my_rank);
            }

            for (size_t epoch = 1; epoch <= options.iterations; ++epoch)
            {
                train(pg, model, *train_loader, optimizer, epoch, dataset_size, options);
            }

            int eligible = my_rank;
            if (options.use_ddp)
                eligible = 0;
            if (my_rank == eligible)
            {
                // std::string fname = "xgcf_ae_model_0.pt";
                std::string fname = "xgcf_ae_model_" + std::to_string(my_rank) + ".pt";
                torch::save(model, fname.c_str());
            }

            // This is just for testing purpose.
            // We load a pre-trained model anyway to ensure the rest of the computation is stable.
            // const char *mname = get_param(m_Parameters, "ae", "").c_str();
            // model = torch::jit::load(mname);
        }
        else
        {
            load_model(model, options, my_rank);
        }
        // if (my_rank == 0)
        // {
        //     double end = MPI_Wtime();
        //     printf("Time taken for Training: %f\n", (end - start));
        // }
        GPTLstop("training");

        // Encode
        GPTLstart("encode");
        // start = MPI_Wtime();
        model->eval();
        std::vector<torch::Tensor> encode_vector;
        for (auto &batch : *encode_loader)
        {
            auto data = batch.data.to(options.device);
            auto _encode = model->encode(data);
            // std::cout << "_encode.sizes = " << _encode.sizes() << std::endl;
            _encode = _encode.to(torch::kCPU);
            encode_vector.push_back(_encode);
        }
        // encodes shape is (nmesh, latent_dim), where nmesh = number of total mesh nodes this process has
        auto encode = torch::cat(encode_vector, 0);
        // std::cout << "encode.sizes = " << encode.sizes() << std::endl;
        // if (my_rank == 0)
        // {
        //     double end = MPI_Wtime();
        //     printf("Time taken for encode: %f\n", (end - start));
        // }
        GPTLstop("encode");
        int offset = sizeof(int);
        int numObjs = encode.size(0);

        // Kmean
        GPTLstart("kmean");
        // start = MPI_Wtime();
        *reinterpret_cast<int *>(bufferOut) = latent_dim;
        for (int i = 0; i < latent_dim; ++i)
        {
            // subselect each column from encode
            auto pq_input_t = encode.slice(1, i, i + 1).contiguous().to(torch::kCPU);
            auto unique_elem = at::_unique(pq_input_t);
            auto elem = std::get<0>(unique_elem);
            float* unique_e = elem.data_ptr<float>();
            float *pq_input = pq_input_t.data_ptr<float>();
            float pq_threshold = 0.0001;
            int numClusters = pow(2, pqbits);
            float *clusters = new float[numClusters];
            initClusterCenters(clusters, unique_e, elem.sizes()[0], numClusters);
            int *membership = new int[numObjs];
            memset(membership, 0, numObjs * sizeof(int));
            kmeans_float(pq_input, numObjs, numClusters, pq_threshold, membership, clusters);
#if UF_DEBUG
            std::cout << "Print clusters:" << std::endl;
            for (int j = 0; j < numClusters; ++j)
                std::cout << clusters[j] << std::endl;
            std::cout << "Print membership:" << std::endl;
            for (int j = 0; j < numObjs; ++j)
                std::cout << membership[j] << " " << pq_input[j] << " " << clusters[membership[j]] << std::endl;
#endif
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
                encode[j][i] = clusters[membership[j]];
        }
        // if (my_rank == 0)
        // {
        //     double end = MPI_Wtime();
        //     printf("Time taken for kmean: %f\n", (end - start));
        // }
        GPTLstop("kmean");

        // Decode
        GPTLstart("decode");
        // start = MPI_Wtime();
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
        // std::cout << "decode.sizes = " << decode.sizes() << std::endl;
        // if (my_rank == 0)
        // {
        //     double end = MPI_Wtime();
        //     printf("Time taken for decode: %f\n", (end - start));
        // }
        GPTLstop("decode");

        // Un-normalize
        GPTLstart("residual");
        auto mu = ds.mu;
        auto sig = ds.sig;
        // std::cout << "mu.sizes = " << mu.sizes() << std::endl;
        // std::cout << "sig.sizes = " << sig.sizes() << std::endl;
        decode = decode * sig.reshape({-1, 1, 1});
        decode = decode + mu.reshape({-1, 1, 1});
        // std::cout << "decode.sizes = " << decode.sizes() << std::endl;

        // forg and docode shape: (nmesh, nx, ny)
        auto diff = ds.forg - decode;
        // std::cout << "forg min,max = " << ds.forg.min().item<double>() << " " << ds.forg.max().item<double>()
        // << std::endl;
        // std::cout << "decode min,max = " << decode.min().item<double>() << " " << decode.max().item<double>()
        // << std::endl;
        // std::cout << "diff min,max = " << diff.min().item<double>() << " " << diff.max().item<double>() << std::endl;
        // std::cout << "orig sizes = " << ds.forg.sizes() << " " << "decode sizes = " << decode.sizes() << std::endl;
        double pd_max_b, pd_max_a;
        double pd_min_b, pd_min_a;
        double pd_omin_b;
        double pd_omax_b;
        pd_max_a = pd_max_b = ds.forg.max().item().to<double>();
        pd_min_a = pd_min_b = ds.forg.min().item().to<double>();
        MPI_Allreduce(&pd_min_b, &pd_omin_b, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&pd_max_b, &pd_omax_b, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        int vx = optim.getVxCount();
        int vy = optim.getVyCount();
        auto nrmse = at::divide(at::sqrt(at::divide(at::pow(diff, 2).sum({1, 2}),at::Scalar(vx*vy))), at::Scalar(pd_omax_b - pd_omin_b));
        // std::ofstream myfile;
        // std::string fname = "nrmse-d3d" + std::to_string(my_rank) + ".txt";
        // myfile.open(fname.c_str());
        // myfile << nrmse << std::endl;
        // myfile.close();
        auto nrmse_index = at::where((nrmse > ae_thresh));
        // std::cout << "Rank " << my_rank << " NRMSE index " << nrmse_index << std::endl;
        int resNodes = nrmse_index[0].sizes()[0];
        at::Tensor perm_diff;
        at::Tensor recon_data;
        at::Tensor rogue_images;
        at::Tensor decoded_images;
        int nodeCount = optim.getNodeCount() * optim.getPlaneCount();
        int data_ceil = int(nodeCount*0.94);
        Dims bC = blockCount;
        if (resNodes > data_ceil)
        {
            perm_diff = diff.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
            bC[2] = nodeCount;
            rogue_images = ds.forg.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
            decoded_images = decode.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
        }
        else if (resNodes == 0) {
            recon_data = decode.reshape({1, -1, ds.nx, ds.ny});
            perm_diff = diff.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
            // perm_diff is never used later
        }
        else
        {
            if (resNodes < 4) {
                std::vector<long> nrmse_vec(nrmse_index[0].data_ptr<long>(),
                             nrmse_index[0].data_ptr<long>() + nrmse_index[0].numel());
                // std::cout << "nrmse values torch " << nrmse_index[0] << std::endl;
                // std::cout << "nrmse values vec " << nrmse_vec << std::endl;
                auto nrmse_sorted_index = at::argsort(nrmse);
                // std::cout << "nrmse sorted index " << nrmse_sorted_index << std::endl;
                while (nrmse_vec.size() < 4) {
                    // for (int ii=0; ii<optim.getNodeCount(); ++ii) {
                    for (int ii=0; ii<nodeCount; ++ii) {
                        int value = nrmse_sorted_index[ii].item().to<long>();
                        if (std::find(nrmse_vec.begin(), nrmse_vec.end(), value) == nrmse_vec.end()){
                            nrmse_vec.push_back(value);
                            // std::cout << "found value " << value << std::endl;
                            break;
                        }
                    }
                }
                nrmse_index[0] = torch::from_blob((void *)nrmse_vec.data(), {nrmse_vec.size()}, torch::kInt64).to(torch::kCUDA);
                resNodes =  nrmse_index[0].sizes()[0];
            }
            using namespace torch::indexing;
            auto diff_reduced = diff.index({nrmse_index[0], Slice(None), Slice(None)});
            std::cout << "nrmse index size " << nrmse_index[0].sizes() << " total nodes " << blockCount[2] << std::endl;
            // std::endl; if (my_rank == 0) { std::cout << "nrmse indexes " << nrmse_index << std::endl; std::cout <<
            // "nrmse values " << nrmse.index({nrmse_index[0]}) << std::endl; std::cout << "diff_reduced sizes " <<
            // diff_reduced.sizes() << std::endl;
            // }
            perm_diff = diff_reduced.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
            bC[2] = nrmse_index[0].sizes()[0];
            rogue_images = (ds.forg).index({nrmse_index[0], Slice(None), Slice(None)});
            rogue_images = rogue_images.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
            decoded_images = decode.index({nrmse_index[0], Slice(None), Slice(None)});
            decoded_images = decoded_images.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
        }
        // std::cout << "Rank " << my_rank << " resNodes " << resNodes << std::endl;
        // use MGARD to compress the residuals
        std::vector<double> diff_data(perm_diff.data_ptr<double>(), perm_diff.data_ptr<double>() + perm_diff.numel());

        // reserve space in output buffer to store MGARD buffer size
        size_t offsetForDecompresedData = offset;
        offset += sizeof(size_t);
        // std::cout << "residual data is ready" << std::endl;
        GPTLstop("residual");

        if (resNodes > 0) {
        GPTLstart("find eb");
        if (leb > 0 && ueb > 0) {
            // std::cout << "Rogue images " << rogue_images.sizes() << " Decoded images " << decoded_images.sizes() << " Diff dims " << perm_diff.sizes() << std::endl;
            m_Parameters["tolerance"] = std::to_string(binarySearchEB(leb, ueb, rogue_images, decoded_images, perm_diff, {0, 0, 0, 0}, {1, bC[1], bC[2], bC[3]}, type, bufferOut+offset, vx, vy, pd_omax_b, pd_omin_b, target_e)).c_str();
            // std::cout << "came here 1" << std::endl;
        }
        GPTLstop("find eb");
        GPTLstart("mgard");
        // Make sure that the shape of the input and the output of MGARD is (1, nmesh, nx, ny)
        CompressMGARD mgard(m_Parameters);
        size_t mgardBufferSize =
            // mgard.Operate(reinterpret_cast<char *>(diff_data.data()), blockStart, bC, type, bufferOut + offset);
            mgard.Operate(reinterpret_cast<char *>(diff_data.data()), {0, 0, 0, 0}, {1, bC[1], bC[2], bC[3]}, type, bufferOut + offset);
        std::cout << my_rank << " - mgard size:" << mgardBufferSize << std::endl;
        // std::cout << "came here 2" << std::endl;
        std::ofstream myfile;
        std::string fname = "mgard-size-iter_" + std::to_string(my_rank) + ".txt";
        myfile.open(fname.c_str());
        myfile << my_rank << " - mgard size:" << mgardBufferSize << " :num images " << bC[2] << std::endl;
        myfile.close();
        GPTLstop("mgard");

        GPTLstart("mgard-decomp");
        PutParameter(bufferOut, offsetForDecompresedData, mgardBufferSize);

        // use MGARD decompress
        std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(bC, helper::GetDataTypeSize(type)));
        mgard.InverseOperate(bufferOut + offset, mgardBufferSize, tmpDecompressBuffer.data());
        // std::cout << "came here 3" << std::endl;
        // std::cout << "mgard inverse is ready" << std::endl;
        offset += mgardBufferSize;
        GPTLstop("mgard-decomp");

        // reconstruct data from the residuals
        auto decompressed_residual_data =
            // torch::from_blob((void *)tmpDecompressBuffer.data(), {blockCount[0], blockCount[1], bC[2], blockCount[3]},
            torch::from_blob((void *)tmpDecompressBuffer.data(), {1, blockCount[1], bC[2], blockCount[3]},
                             torch::kFloat64)
                .permute({0, 2, 1, 3});
        // std::cout << "decompressed sizes " << decompressed_residual_data.sizes() << std::endl;
        if (resNodes > data_ceil)
        {
            recon_data = decode.reshape({1, -1, ds.nx, ds.ny}) + decompressed_residual_data;
        }
        else
        {
            using namespace torch::indexing;
            // auto res = at::zeros({blockCount[0], blockCount[2], blockCount[1], blockCount[3]}, torch::kFloat64);
            auto res = at::zeros({1, blockCount[0]*blockCount[2], blockCount[1], blockCount[3]}, torch::kFloat64);
            // std::cout << "res sizes 1 " << res.sizes() << std::endl;
            res.index_put_({Slice(None), nrmse_index[0], Slice(None), Slice(None)}, decompressed_residual_data);
            // std::cout << "res sizes 2 " << res.sizes() << std::endl;
            recon_data = decode.reshape({1, -1, ds.nx, ds.ny}) + res;
        }
        }
        // std::cout << "came here 4" << std::endl;
        // std::cout << "decode sizes " << decode.reshape({1, -1, ds.nx, ds.ny}).sizes() << std::endl;
        recon_data = recon_data.permute({0, 2, 1, 3});
        // auto recon_data = decode + diff;
        // recon_data = recon_data.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3});
        // std::cout << "recon data sizes " << recon_data.sizes() << std::endl;
#if UF_DEBUG
        auto recon_data_interm = decode.reshape({1, -1, ds.nx, ds.ny}) + diff;
        recon_data_interm = recon_data_interm.contiguous().cpu();
        auto datain_t = torch::from_blob((void *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]},
                                         torch::kFloat64)
                            .to(torch::kCPU)
                            .permute({0, 2, 1, 3});
        double pd_b = at::pow((datain_t - recon_data_interm), 2).sum().item().to<double>();
        double pd_size_b, pd_size_a;
        pd_max_a = pd_max_b = datain_t.max().item().to<double>();
        pd_min_a = pd_min_b = datain_t.min().item().to<double>();
        pd_size_a = pd_size_b = recon_data_interm.numel();
        // get total error for recon
        double pd_e_b;
        double pd_s_b;
        MPI_Allreduce(&pd_b, &pd_e_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&pd_size_b, &pd_s_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (my_rank == 0)
        {
            std::cout << " PD error: " << sqrt(pd_e_b / pd_s_b) / (pd_omax_b - pd_omin_b) << std::endl;
        }
#endif
        GPTLstart("Lagrange");
        // recon_data shape (1, nmesh, nx, ny) and make it contiguous in memory
        recon_data = recon_data.contiguous().cpu();
        std::vector<double> recon_vec(recon_data.data_ptr<double>(),
                                      recon_data.data_ptr<double>() + recon_data.numel());

        // apply post processing
        // recon_vec shape: (1, nmesh, nx, ny)
        // int unconvsize = optim.computeLagrangeParameters(recon_vec.data(), blockCount);
        int unconvsize = optim.computeLagrangeParameters(recon_vec.data(), {1, blockCount[1], blockCount[0]*blockCount[2], blockCount[3]});
        // std::cout << "came here 5" << std::endl;
        GPTLstop("Lagrange");
        offset += unconvsize;
        std::ofstream myfile;
        std::string fname = "unconv-size-iter_" + std::to_string(my_rank) + ".txt";
        myfile.open(fname.c_str());
        myfile << my_rank << " - unconv size:" << unconvsize << std::endl;
        myfile.close();
        bufferOutOffset += offset;

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

        bufferOutOffset += offset;
    }
#if 0
    else if (compression_method == 4)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        const char *mname = m_Parameters["ae"].c_str();
        // std::cout << "Module name:" << mname;
        torch::jit::script::Module module;
        try
        {
            GPTLstart("load model");
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load(mname);
            GPTLstop("load model");
        }
        catch (const c10::Error &e)
        {
            std::cerr << "error loading the model\n";
            return -1;
        }

        int latent_dim = atoi(get_param(m_Parameters, "latent_dim", "5").c_str());
        ;
        GPTLstart("prep");
        auto ds = CustomDataset((double *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]});
        auto dataset = ds.map(torch::data::transforms::Stack<>());
        const size_t dataset_size = dataset.size().value();
        auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset),
                                                                                              options.batch_size);
        GPTLstop("prep");

        GPTLstart("encode");
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
        GPTLstop("encode");
        // std::cout << "encode.sizes = " << encode.sizes() << std::endl;

        // Kmean
        int offset = sizeof(int);
        int numObjs = encode.size(0);
        *reinterpret_cast<int *>(bufferOut) = latent_dim;
        GPTLstart("kmean");
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
        // std::cout << "kmean is done " << std::endl;
        GPTLstop("kmean");

        // Decode
        GPTLstart("decode");
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
        // std::cout << "decode.sizes = " << decode.sizes() << std::endl;
        GPTLstop("decode");

        // Un-normalize
        GPTLstart("residual");
        auto mu = ds.mu;
        auto sig = ds.sig;
        // std::cout << "mu.sizes = " << mu.sizes() << std::endl;
        // std::cout << "sig.sizes = " << sig.sizes() << std::endl;
        decode = decode * sig.reshape({-1, 1, 1});
        decode = decode + mu.reshape({-1, 1, 1});
        // std::cout << "decode.sizes = " << decode.sizes() << std::endl;

        // forg and docode shape: (nmesh, nx, ny)
        auto diff = ds.forg - decode;
        // std::cout << "forg min,max = " << ds.forg.min().item<double>() << " " << ds.forg.max().item<double>()
        // << std::endl;
        // std::cout << "decode min,max = " << decode.min().item<double>() << " " << decode.max().item<double>()
        // << std::endl;
        // std::cout << "diff min,max = " << diff.min().item<double>() << " " << diff.max().item<double>() << std::endl;
        // std::cout << "orig sizes = " << ds.forg.sizes() << " " << "decode sizes = " << decode.sizes() << std::endl;
        double pd_max_b, pd_max_a;
        double pd_min_b, pd_min_a;
        double pd_omin_b;
        double pd_omax_b;
        pd_max_a = pd_max_b = ds.forg.max().item().to<double>();
        pd_min_a = pd_min_b = ds.forg.min().item().to<double>();
        MPI_Allreduce(&pd_min_b, &pd_omin_b, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&pd_max_b, &pd_omax_b, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        auto nrmse = at::divide(at::sqrt(at::divide(at::pow(diff, 2).sum({1, 2}),at::Scalar(optim.getVxCount() * optim.getVyCount()))), at::Scalar(pd_omax_b - pd_omin_b));
        auto nrmse_index = at::where((nrmse > ae_thresh));
        int resNodes = nrmse_index[0].sizes()[0];
        at::Tensor perm_diff;
        at::Tensor recon_data;
        int data_ceil = int(optim.getNodeCount() * 0.94);
        Dims bC = blockCount;
        if (resNodes > data_ceil)
        {
            perm_diff = diff.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
            bC[2] = optim.getNodeCount();
        }
        else if (resNodes > data_ceil) {
            recon_data = decode.reshape({1, -1, ds.nx, ds.ny});
            perm_diff = diff.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
            // perm_diff is never used later
        }
        else
        {
            if (resNodes < 4) {
                std::vector<long> nrmse_vec(nrmse_index[0].data_ptr<long>(),
                             nrmse_index[0].data_ptr<long>() + nrmse_index[0].numel());
                // std::cout << "nrmse values torch " << nrmse_index[0] << std::endl;
                // std::cout << "nrmse values vec " << nrmse_vec << std::endl;
                auto nrmse_sorted_index = at::argsort(nrmse);
                // std::cout << "nrmse sorted index " << nrmse_sorted_index << std::endl;
                while (nrmse_vec.size() < 4) {
                    for (int ii=0; ii<optim.getNodeCount(); ++ii) {
                        int value = nrmse_sorted_index[ii].item().to<long>();
                        if (std::find(nrmse_vec.begin(), nrmse_vec.end(), value) == nrmse_vec.end()){
                            nrmse_vec.push_back(value);
                            // std::cout << "found value " << value << std::endl;
                            break;
                        }
                    }
                }
                nrmse_index[0] = torch::from_blob((void *)nrmse_vec.data(), {nrmse_vec.size()}, torch::kInt64).to(torch::kCUDA);
                resNodes = nrmse_index[0].sizes()[0];
            }
            using namespace torch::indexing;
            auto diff_reduced = diff.index({nrmse_index[0], Slice(None), Slice(None)});
            // std::cout << "nrmse index size " << nrmse_index[0].sizes() << " total nodes " << blockCount[2] << std::endl;
            // std::endl; if (my_rank == 0) { std::cout << "nrmse indexes " << nrmse_index << std::endl; std::cout <<
            // "nrmse values " << nrmse.index({nrmse_index[0]}) << std::endl; std::cout << "diff_reduced sizes " <<
            // diff_reduced.sizes() << std::endl;
            // }
            perm_diff = diff_reduced.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3}).contiguous().cpu();
            bC[2] = nrmse_index[0].sizes()[0];
        }
        // use MGARD to compress the residuals
        std::vector<double> diff_data(perm_diff.data_ptr<double>(), perm_diff.data_ptr<double>() + perm_diff.numel());

        // reserve space in output buffer to store MGARD buffer size
        size_t offsetForDecompresedData = offset;
        offset += sizeof(size_t);
        // std::cout << "residual data is ready" << std::endl;
        GPTLstop("residual");

        if (resNodes > 0) {
        // std::cout << "block Count orig " << blockCount << std::endl;
        // std::cout << "block Count reduced " << bC << std::endl;
        // apply MGARD operate.
        GPTLstart("mgard");
        // Make sure that the shape of the input and the output of MGARD is (1, nmesh, nx, ny)
        CompressMGARD mgard(m_Parameters);
        size_t mgardBufferSize =
            mgard.Operate(reinterpret_cast<char *>(diff_data.data()), blockStart, bC, type, bufferOut + offset);
        std::cout << my_rank << " - mgard size:" << mgardBufferSize << " :num images " << bC[2] << std::endl;
        GPTLstop("mgard");

        GPTLstart("mgard-decomp");
        PutParameter(bufferOut, offsetForDecompresedData, mgardBufferSize);

        // use MGARD decompress
        std::vector<char> tmpDecompressBuffer(helper::GetTotalSize(bC, helper::GetDataTypeSize(type)));
        mgard.InverseOperate(bufferOut + offset, mgardBufferSize, tmpDecompressBuffer.data());
        // std::cout << "mgard inverse is ready" << std::endl;
        offset += mgardBufferSize;
        GPTLstop("mgard-decomp");

        // reconstruct data from the residuals
        auto decompressed_residual_data =
            torch::from_blob((void *)tmpDecompressBuffer.data(), {blockCount[0], blockCount[1], bC[2], blockCount[3]},
                             torch::kFloat64).permute({0, 2, 1, 3});
        // std::cout << "decompressed sizes " << decompressed_residual_data.sizes() << std::endl;
        if (resNodes > data_ceil)
        {
            recon_data = decode.reshape({1, -1, ds.nx, ds.ny}) + decompressed_residual_data;
        }
        else
        {
            using namespace torch::indexing;
            auto res = at::zeros({blockCount[0], blockCount[2], blockCount[1], blockCount[3]}, torch::kFloat64);
            // std::cout << "res sizes 1 " << res.sizes() << std::endl;
            res.index_put_({Slice(None), nrmse_index[0], Slice(None), Slice(None)}, decompressed_residual_data);
            // std::cout << "res sizes 2 " << res.sizes() << std::endl;
            recon_data = decode.reshape({1, -1, ds.nx, ds.ny}) + res;
        }
        // std::cout << "decode sizes " << decode.reshape({1, -1, ds.nx, ds.ny}).sizes() << std::endl;
        recon_data = recon_data.permute({0, 2, 1, 3});
        // auto recon_data = decode + diff;
        // recon_data = recon_data.reshape({1, -1, ds.nx, ds.ny}).permute({0, 2, 1, 3});
        // std::cout << "recon data sizes " << recon_data.sizes() << std::endl;
#if UF_DEBUG
        auto recon_data_interm = decode.reshape({1, -1, ds.nx, ds.ny}) + diff;
        recon_data_interm = recon_data_interm.contiguous().cpu();
        auto datain_t = torch::from_blob((void *)dataIn, {blockCount[0], blockCount[1], blockCount[2], blockCount[3]},
                                         torch::kFloat64)
                            .to(torch::kCPU)
                            .permute({0, 2, 1, 3});
        double pd_b = at::pow((datain_t - recon_data_interm), 2).sum().item().to<double>();
        double pd_size_b, pd_size_a;
        pd_max_a = pd_max_b = datain_t.max().item().to<double>();
        pd_min_a = pd_min_b = datain_t.min().item().to<double>();
        pd_size_a = pd_size_b = recon_data_interm.numel();
        // get total error for recon
        double pd_e_b;
        double pd_s_b;
        MPI_Allreduce(&pd_b, &pd_e_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&pd_size_b, &pd_s_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (my_rank == 0)
        {
            std::cout << " PD error: " << sqrt(pd_e_b / pd_s_b) / (pd_omax_b - pd_omin_b) << std::endl;
        }
#endif
        }
        GPTLstart("Lagrange");
        // recon_data shape (1, nmesh, nx, ny) and make it contiguous in memory
        recon_data = recon_data.contiguous().cpu();
        std::vector<double> recon_vec(recon_data.data_ptr<double>(),
                                      recon_data.data_ptr<double>() + recon_data.numel());

        // apply post processing
        // recon_vec shape: (1, nmesh, nx, ny)
        optim.computeLagrangeParameters(recon_vec.data(), blockCount);
        GPTLstop("Lagrange");
        bufferOutOffset += offset;
    }
#endif
    GPTLstop("total");

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
        size_t ppsize = optim.putResult(bufferOut, bufferOutOffset, m_Parameters["prec"].c_str());
        bufferOutOffset += ppsize;
    }

#ifdef UF_DEBUG
    int arraySize =
        optim.getPlaneCount() * optim.getNodeCount() * optim.getVxCount() * optim.getVyCount() * sizeof(double);
    char *dataOut = new char[arraySize];
    size_t result = InverseOperate(bufferOut, bufferOutOffset, dataOut);
#endif

    char fname[80];
    sprintf(fname, "mgardplus-timing.%d.txt", my_rank);
    // GPTLpr_file(fname);
    GPTLpr_summary_file(MPI_COMM_WORLD, "mgardplus-timing.summary.txt");
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

    const uint8_t species = GetParameter<uint8_t>(bufferIn, bufferInOffset);
    const uint8_t precision = GetParameter<uint8_t>(bufferIn, bufferInOffset);
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
    LagrangeTorchL2 optim(planeOffset, nodeOffset, planeCount, nodeCount, vxCount, vyCount, species, precision);
    double *doubleData = reinterpret_cast<double *>(dataOut);
    optim.setDataFromCharBuffer(doubleData, bufferIn + bufferInOffset + mgardBufferSize, sizeOut);

    return sizeOut;
}

size_t CompressMGARDPlus::InverseOperate(const char *bufferIn, const size_t sizeIn, char *dataOut)
{
    size_t bufferInOffset = 1; // skip operator type
    const uint8_t bufferVersion = GetParameter<uint8_t>(bufferIn, bufferInOffset);
    int bV = (int) bufferVersion;
    bufferInOffset += 2;

    if (bV == 1)
    {
        return DecompressV1(bufferIn, bufferInOffset, sizeIn, dataOut);
    }
    else if (bV == 2)
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
