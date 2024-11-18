//
// Created by xianyang on 24-11-14.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;


class Logger : public ILogger {
public:
    Severity reportableSeverity;

    Logger(Severity severity = Severity::kINFO) :
            reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) noexcept override {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
};


void get_engine(const std::string onnx_path, const std::string engine_path, const bool fp16) {
    const int kInputH = 640;
    const int kInputW = 640;
    IRuntime *runtime;
    ICudaEngine *engine;
    Logger gLogger;
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30);
    IInt8Calibrator *pCalibrator = nullptr;
    if (fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(onnx_path.c_str(), int(gLogger.reportableSeverity))) {
        std::cout << std::string("Failed parsing .onnx file!") << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            auto *error = parser->getError(i);
            std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc())
                      << std::endl;
        }
        return;
    }
    std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

    ITensor *inputTensor = network->getInput(0);
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32{4, {1, 3, kInputH, kInputW}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32{4, {1, 3, kInputH, kInputW}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32{4, {1, 3, kInputH, kInputW}});
    config->addOptimizationProfile(profile);

    IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Succeeded building serialized engine!" << std::endl;

    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    if (engine == nullptr) {
        std::cout << "Failed building engine!" << std::endl;
        return;
    }
    std::cout << "Succeeded building engine!" << std::endl;

//    if (bINT8Mode && pCalibrator != nullptr) {
//        delete pCalibrator;
//    }

    std::ofstream engineFile(engine_path, std::ios::binary);
    engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
    std::cout << "Succeeded saving .plan file!" << std::endl;

    delete engineString;
    delete parser;
    delete config;
    delete network;
    delete builder;
}

int main(int argc, char **argv) {
    if (argc<=2){
        std::cout<<"you must provide two or three input, onnx file, engine output file name,you also can provide a bool true or false, it control fp16"<<std::endl;
        std::cout<<"such as: ./get_engine test.onnx filename.engine true"<<std::endl;
        return 0;
    }
    std::string onnx = argv[1];
    std::string engine = argv[2];
    bool fp16 = argv[3];
    get_engine(onnx, engine, fp16);

}
/*
 g++ -o get_engine get_trt.cpp -I/home/xianyang/Documents/TensorRT-8.6.1.6/include -L/home/xianyang/Documents/TensorRT-8.6.1.6/lib -lnvinfer -I/usr/local/cuda/include -L/usr/local/cuda/lib64  -lcudart -lnvonnxparser
 * */