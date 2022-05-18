/*
 * Copyright (c) 2012-2022
 * All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 * You may not use, copy, distribute, modify, transmit in any form this file
 * except in compliance with THUNDERCOMM in writing by applicable law.
 * 
 * @Description: Implementation of Face Detection algorithm APIs.
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-05-17 20:26:56
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-05-17 21:20:47
 */

#include <unistd.h>

#include "TSYolov5sImpl.h"

namespace ts {

TSObjectDetection::TSObjectDetection()
{
    impl = new TSObjectDetectionImpl();
}

TSObjectDetection::~TSObjectDetection()
{
    if (nullptr != impl) {
        delete static_cast<TSObjectDetectionImpl*>(impl);
        impl = nullptr;
    }
}

bool TSObjectDetection::Init(const std::string& model_path, const runtime_t runtime)
{
    if (IsInitialized()) {
        return static_cast<TSObjectDetectionImpl*>(impl)->DeInitialize() &&
               static_cast<TSObjectDetectionImpl*>(impl)->Initialize(model_path, runtime);
    } else {
        return static_cast<TSObjectDetectionImpl*>(impl)->Initialize(model_path, runtime);
    }
}

bool TSObjectDetection::Deinit()
{
    if (nullptr != impl && IsInitialized()) {
        return static_cast<TSObjectDetectionImpl*>(impl)->DeInitialize();
    } else {
        TS_ERROR_LOG("TSObjectDetection: deinit failed!");
        return false;
    }
}

bool TSObjectDetection::IsInitialized()
{
    return static_cast<TSObjectDetectionImpl*>(impl)->IsInitialized();
}

bool TSObjectDetection::Detect(const ts::TSImgData& image, std::vector<ts::ObjectData>& vec_res)
{
    if (nullptr != impl && IsInitialized()) {
        auto ret = static_cast<TSObjectDetectionImpl*>(impl)->Detect(image, vec_res);
        return ret;
    } else {
        TS_ERROR_LOG("TSObjectDetection::Detect failed caused by incompleted initialization!");
        return false;
    }
}

bool TSObjectDetection::SetScoreThreshold(const float& conf_thresh, const float& nms_thresh)
{
    if (nullptr != impl) {
        return static_cast<TSObjectDetectionImpl*>(impl)->SetScoreThresh(conf_thresh, nms_thresh);
    } else {
        TS_ERROR_LOG("TSObjectDetection::setScoreThresh failed because incompleted initialization!");
        return false;
    }
}

}   // namespace ts