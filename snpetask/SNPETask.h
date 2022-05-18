/*
 * Copyright (c) 2012-2022
 * All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 * You may not use, copy, distribute, modify, transmit in any form this file
 * except in compliance with THUNDERCOMM in writing by applicable law.
 * 
 * @Description: Inference SDK based on SNPE. 
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-05-17 20:28:01
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-05-18 13:00:55
 */

#include <memory>
#include <vector>
#include <map>
#include <string>

#include <SNPE/SNPE.h>

#include "TSStruct.h"

namespace snpetask {

class SNPETask {
public:
    SNPETask();
    ~SNPETask();

    bool init(const std::string& model_path, const runtime_t runtime);
    bool deInit();
    bool setOutputLayers(const std::vector<std::string>& outputLayers);

    zdl::DlSystem::TensorShape getInputShape(const std::string name);
    zdl::DlSystem::TensorShape getOutputShape(const std::string name);

    float* getTensor(const std::string& name);

    bool isInit() {
        return m_isInit;
    }

    bool execute();

private:
    bool m_isInit = false;

    std::unique_ptr<zdl::DlContainer::IDlContainer> m_container;
    std::unique_ptr<zdl::SNPE::SNPE> m_snpe;
    zdl::DlSystem::Runtime_t m_runtime;
    zdl::DlSystem::StringList m_outputLayers;

    const zdl::DlSystem::TensorShape m_inputShape;
    std::map<std::string, const zdl::DlSystem::TensorShape> m_outputShapes;

    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer> > m_inputUserBuffers;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer> > m_outputUserBuffers;
    zdl::DlSystem::UserBufferMap m_inputUserBufferMap;
    zdl::DlSystem::UserBufferMap m_outputUserBufferMap;
    std::unordered_map<std::string, std::vector<uint8_t> > m_inputTensors;
    std::unordered_map<std::string, std::vector<uint8_t> > m_outputTensors;
};


}   // namespace snpetask