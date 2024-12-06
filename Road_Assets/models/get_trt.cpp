//
// Created by xianyang on 24-11-14.
//

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
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


void get_engine(const std::string onnx_path, const std::string engine_path, const int size, const bool fp16) {
    const int kInputH = size;
    const int kInputW = size;
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
bool is_integer(const std::string &str) {
    return str.find_first_not_of("0123456789") == std::string::npos;
}
bool is_boolean(const std::string &str) {
    return str == "true" || str == "false";
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        std::cout << "You must provide two or three inputs: onnx file, engine output file name.\n";
        std::cout << "You can also provide: 1. model input size (default: 640).\n";
        std::cout << " 			    2. a bool (true/false) to control fp16.\n";
        std::cout << "Example: ./get_engine test.onnx filename.engine 640 true\n";
        return 0;
    }
    std::string onnx = argv[1];
    std::string engine = argv[2];
    // 检查文件是否存在
    std::ifstream onnx_file(onnx);
    if (!onnx_file) {
        std::cerr << "Error: ONNX file " << onnx << " does not exist.\n";
        return 1;
    }
    // 可选参数
    int size = 640;       // 默认输入大小
    bool fp16 = false;    // 默认不启用 FP16
    bool size_provided = false; // 标记是否提供了输入大小
    bool fp16_provided = false; // 标记是否提供了 FP16

    // 参数解析
    if (argc > 3) {
        std::string arg3 = argv[3];
        if (is_integer(arg3)) {
            size = std::stoi(arg3);
            size_provided = true;
        } else if (is_boolean(arg3)) {
            fp16 = (arg3 == "true");
            fp16_provided = true;
        } else {
            std::cerr << "Error: Invalid third argument. It must be an integer (size) or a boolean (true/false). Provided: " << arg3 << "\n";
            return 1;
        }
    }
    if (argc > 4) {
        std::string arg4 = argv[4];
        if (is_boolean(arg4)) {
            fp16 = (arg4 == "true");
            fp16_provided = true;
        } else {
            std::cerr << "Error: Invalid fourth argument. It must be a boolean (true/false). Provided: " << arg4 << "\n";
            return 1;
        }
    }
    
    std::cout << "---------------------------------------------------------" << "\n";
    std::cout << "ONNX file: " << onnx << "\n";
    std::cout << "Engine file: " << engine << "\n";
    if (size_provided) {
        std::cout << "Input size: " << size << " (provided by user)\n";
    } else {
        std::cout << "Input size: " << size << " (default value)\n";
    }

    if (fp16_provided) {
        std::cout << "FP16 enabled: " << (fp16 ? "true" : "false") << " (provided by user)\n";
    } else {
        std::cout << "FP16 enabled: " << (fp16 ? "true" : "false") << " (default value)\n";
    }
    std::cout << "---------------------------------------------------------" << "\n";
    
    
    get_engine(onnx, engine, size, fp16);

    return 0;
    
}
/*
 g++ -o get_engine get_trt.cpp -I/home/xianyang/Documents/TensorRT-8.6.1.6/include -L/home/xianyang/Documents/TensorRT-8.6.1.6/lib -lnvinfer -I/usr/local/cuda/include -L/usr/local/cuda/lib64  -lcudart -lnvonnxparser
 * */
