/*
 * Copyright (c) 2012-2022
 * All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 * You may not use, copy, distribute, modify, transmit in any form this file
 * except in compliance with THUNDERCOMM in writing by applicable law.
 * 
 * @Description: Inference SDK based on SNPE.
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-05-18 09:48:36
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-06-10 07:51:39
 */


#include "SNPETask.h"

namespace snpetask{

static size_t calcSizeFromDims(const zdl::DlSystem::Dimension* dims, size_t rank, size_t elementSize)
{
    if (rank == 0) return 0;
    size_t size = elementSize;
    while (rank--) {
        size *= *dims;
        dims++;
    }
    return size;
}


static void createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                      std::unordered_map<std::string, float*>& applicationBuffers,
                      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                      const zdl::DlSystem::TensorShape& bufferShape,
                      const char* name,
                      float* buffer)
{
    // Calculate the stride based on buffer strides, assuming tightly packed.
    // Note: Strides = Number of bytes to advance to the next element in each dimension.
    // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
    // Note: Buffer stride is usually known and does not need to be calculated.
    std::vector<size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = sizeof(float);
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--)
    {
        stride *= bufferShape[i];
        strides[i-1] = stride;
    }
    // const size_t bufferElementSize = sizeof(float);
    size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), 1);
    buffer = new float[bufSize];

    // set the buffer encoding type
    zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, buffer);
    // create SNPE user buffer from the user-backed buffer
    zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
    snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(applicationBuffers.at(name),
                                                                bufSize,
                                                                strides,
                                                                &userBufferEncodingFloat));
    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
    userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

SNPETask::SNPETask()
{
    static zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    TS_INFO_LOG("Using SNPE(%s)", version.asString().c_str());
}

SNPETask::~SNPETask()
{

}

bool SNPETask::init(const std::string& model_path, const runtime_t runtime)
{

    m_container = zdl::DlContainer::IDlContainer::open(model_path);

    switch (runtime) {
        case CPU:
            m_runtime = zdl::DlSystem::Runtime_t::CPU;
            break;
        case GPU:
            m_runtime = zdl::DlSystem::Runtime_t::GPU;
            break;
        case GPU_16:
            m_runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
            break;
        case DSP:
            m_runtime = zdl::DlSystem::Runtime_t::DSP;
            break;
        case AIP:
            m_runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
            break;
        default:
            m_runtime = zdl::DlSystem::Runtime_t::CPU;
            break;
    }

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(m_runtime)) {
        TS_ERROR_LOG("Selected runtime not present. Falling back to CPU.");
        m_runtime = zdl::DlSystem::Runtime_t::CPU;
    }

    zdl::DlSystem::PerformanceProfile_t profile = zdl::DlSystem::PerformanceProfile_t::BURST;

    zdl::SNPE::SNPEBuilder snpeBuilder(m_container.get());
    m_snpe = snpeBuilder.setOutputLayers(m_outputLayers)
       .setRuntimeProcessorOrder(m_runtime)
       .setPerformanceProfile(profile)
       .setUseUserSuppliedBuffers(true)
       .build();

    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = m_snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    m_inputTensorSize = inputNames.size();
    m_inputBuffers = (float **)malloc(m_inputTensorSize * sizeof(float*));
    int i = 0;
    // create SNPE user buffers for each application storage buffer
    for (const char* name : inputNames) {
        // get attributes of buffer by name
        auto bufferAttributesOpt = m_snpe->getInputOutputBufferAttributes(name);
        if (!bufferAttributesOpt) {
            TS_ERROR_LOG("Error obtaining attributes for input tensor: %s", name);
            return false;
        }

        const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
        std::vector<size_t> tensorShape;
        for (size_t j = 0; j < bufferShape.rank(); j++) {
            tensorShape.push_back(bufferShape[j]);
        }
        m_inputShapes.emplace(name, tensorShape);

        createUserBuffer(m_inputUserBufferMap, m_inputTensors, m_inputUserBuffers, bufferShape, name, m_inputBuffers[i++]);
    }

    // get output tensor names of the network that need to be populated
    const auto& outputNamesOpt = m_snpe->getOutputTensorNames();
    if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names");
    const zdl::DlSystem::StringList& outputNames = *outputNamesOpt;
    m_outputTensorSize = outputNames.size();
    m_outputBuffers = (float **)malloc(m_outputTensorSize * sizeof(float*));
    i = 0;
    // create SNPE user buffers for each application storage buffer
    for (const char* name : outputNames) {
        // get attributes of buffer by name
        auto bufferAttributesOpt = m_snpe->getInputOutputBufferAttributes(name);
        if (!bufferAttributesOpt) {
            TS_ERROR_LOG("Error obtaining attributes for input tensor: %s", name);
            return false;
        }

        const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
        std::vector<size_t> tensorShape;
        for (size_t j = 0; j < bufferShape.rank(); j++) {
            tensorShape.push_back(bufferShape[j]);
        }
        m_outputShapes.emplace(name, tensorShape);

        createUserBuffer(m_outputUserBufferMap, m_outputTensors, m_outputUserBuffers, bufferShape, name, m_outputBuffers[i++]);
    }

    m_isInit = true;

    return true;
}

bool SNPETask::deInit()
{
    if (nullptr != m_snpe) {
        m_snpe.reset(nullptr);
    }

    for (int i = 1; i < m_inputTensorSize; i++) {
        free(m_inputBuffers[i]);
    }
	free(m_inputBuffers);

    for (int i = 1; i < m_outputTensorSize; i++) {
        free(m_outputBuffers[i]);
    }
	free(m_outputBuffers);

}

bool SNPETask::setOutputLayers(std::vector<std::string>& outputLayers)
{
    for (size_t i = 0; i < outputLayers.size(); i ++) {
        m_outputLayers.append(outputLayers[i].c_str());
    }

    return true;
}

std::vector<size_t> SNPETask::getInputShape(const std::string& name)
{
    if (isInit()) {
        if (m_inputShapes.find(name) != m_inputShapes.end()) {
            return m_inputShapes.at(name);
        }
        TS_ERROR_LOG("Can't find any input layer named %s", name.c_str());
        return {};
    } else {
        TS_ERROR_LOG("The getInputShape() needs to be called after AICContext is initialized!");
        return {};
    }
}

std::vector<size_t> SNPETask::getOutputShape(const std::string& name)
{
    if (isInit()) {
        if (m_outputShapes.find(name) != m_outputShapes.end()) {
            return m_outputShapes.at(name);
        }
        TS_ERROR_LOG("Can't find any ouput layer named %s", name.c_str());
        return {};
    } else {
        TS_ERROR_LOG("The getOutputShape() needs to be called after AICContext is initialized!");
        return {};
    }
}

float* SNPETask::getInputTensor(const std::string& name)
{
    if (isInit()) {
        if (m_inputTensors.find(name) != m_inputTensors.end()) {
            return m_inputTensors.at(name);
        }
        TS_ERROR_LOG("Can't find any input tensor named %s", name.c_str());
        return nullptr;
    } else {
        TS_ERROR_LOG("The getInputTensor() needs to be called after AICContext is initialized!");
        return nullptr;
    }
}

float* SNPETask::getOutputTensor(const std::string& name)
{
    if (isInit()) {
        if (m_outputTensors.find(name) != m_outputTensors.end()) {
            return m_outputTensors.at(name);
        }
        TS_ERROR_LOG("Can't find any output tensor named %s", name.c_str());
        return nullptr;
    } else {
        TS_ERROR_LOG("The getOutputTensor() needs to be called after AICContext is initialized!");
        return nullptr;
    }
}

bool SNPETask::execute()
{
    if (!m_snpe->execute(m_inputUserBufferMap, m_outputUserBufferMap)) {
        TS_ERROR_LOG("SNPETask execute failed: %s", zdl::DlSystem::getLastErrorString());
        return false;
    }

    return true;
}

}   // namespace snpetask