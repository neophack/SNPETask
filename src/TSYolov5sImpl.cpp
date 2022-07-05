/*
 * Copyright (c) 2012-2022
 * All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 * You may not use, copy, distribute, modify, transmit in any form this file
 * except in compliance with THUNDERCOMM in writing by applicable law.
 * 
 * @Description: Implementation of object detection algorithm handler.
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-05-17 20:28:01
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-07-05 10:34:46
 */

#include <math.h>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "TSYolov5sImpl.h"

TSObjectDetectionImpl::TSObjectDetectionImpl() : m_task(nullptr) {

}

TSObjectDetectionImpl::~TSObjectDetectionImpl() {
    DeInitialize();
}

bool TSObjectDetectionImpl::Initialize(const std::string& model_path, const runtime_t runtime)
{
    m_task = std::move(std::unique_ptr<snpetask::SNPETask>(new snpetask::SNPETask()));

    m_outputLayers.push_back(OUTPUT_NODE0);        // stride: 8
    m_outputLayers.push_back(OUTPUT_NODE1);        // stride: 16
    m_outputLayers.push_back(OUTPUT_NODE2);        // stride: 32
    m_outputTensors.push_back(OUTPUT_TENSOR0);     // 1*80*80*3*85
    m_outputTensors.push_back(OUTPUT_TENSOR1);     // 1*40*40*3*85
    m_outputTensors.push_back(OUTPUT_TENSOR2);     // 1*20*20*3*85

    m_task->setOutputLayers(m_outputLayers);

    m_task->init(model_path, runtime);

    m_output = new float[MODEL_OUTPUT_GRIDS * MODEL_OUTPUT_CHANNEL];

    m_isInit = true;
    return true;
}

bool TSObjectDetectionImpl::DeInitialize()
{
    if (m_task) {
        m_task->deInit();
        m_task.reset(nullptr);
    }

    if (m_output) {
        delete[] m_output;
        m_output = nullptr;
    }

    m_isInit = false;
    return true;
}

bool TSObjectDetectionImpl::PreProcess(const ts::TSImgData& image)
{
    auto inputShape = m_task->getInputShape(INPUT_TENSOR);

    size_t batch = inputShape[0];
    size_t inputHeight = inputShape[1];
    size_t inputWidth = inputShape[2];
    size_t channel = inputShape[3];

    if (m_task->getInputTensor(INPUT_TENSOR) == nullptr) {
        TS_ERROR_LOG("Empty input tensor");
        return false;
    }

    cv::Mat input(inputHeight, inputWidth, CV_32FC3, m_task->getInputTensor(INPUT_TENSOR), inputWidth * channel);

    if (image.empty()) {
        TS_ERROR_LOG("Invalid image!");
        return false;
    }

    int imgFormat = image.format();
    if (imgFormat != TYPE_BGR_U8 && imgFormat != TYPE_RGB_U8) {
        TS_ERROR_LOG("Invaild image format %d, expected to be rgb or bgr!", imgFormat);
        return false;
    }

    int imgWidth = image.width();
    int imgHeight = image.height();
    
    m_scale = std::min(inputHeight /(float)imgHeight, inputWidth / (float)imgWidth);
    int scaledWidth = imgWidth * m_scale;
    int scaledHeight = imgHeight * m_scale;
    m_xOffset = (inputWidth - scaledWidth) / 2;
    m_yOffset = (inputHeight - scaledHeight) / 2;

    cv::Mat image_tmp(imgHeight, imgWidth, CV_8UC3, image.data(), image.stride());
    if (imgFormat == TYPE_BGR_U8) {
        cv::cvtColor(image_tmp, image_tmp, cv::COLOR_BGR2RGB);
    }

    cv::Mat inputMat(inputHeight, inputWidth, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat roiMat(inputMat, cv::Rect(m_xOffset, m_yOffset, scaledWidth, scaledHeight));
    cv::resize(image_tmp, roiMat, cv::Size(scaledWidth, scaledHeight), cv::INTER_LINEAR);

    inputMat.convertTo(input, CV_32FC3);
    input /= 255.0f;

}

bool TSObjectDetectionImpl::Detect(const ts::TSImgData& image,
    std::vector<ts::ObjectData>& results)
{

    if (m_roi.empty()) {
        PreProcess(image);
    } else {
        auto roi_image = image.roi(m_roi);
        PreProcess(roi_image);
    }

    if (!m_task->execute()) {
        TS_ERROR_LOG("SNPETask execute failed.");
        return false;
    }

    PostProcess(results);

    return true;
}

bool TSObjectDetectionImpl::PostProcess(std::vector<ts::ObjectData> &results)
{
    float strides[3] = {8, 16, 32};
    float anchorGrid[][6] = {
        {10, 13, 16, 30, 33, 23},       // 8*8
        {30, 61, 62, 45, 59, 119},      // 16*16
        {116, 90, 156, 198, 373, 326},  // 32*32
    };

    // copy all outputs to one array.
    // [80 * 80 * 3 * 85]----\
    // [40 * 40 * 3 * 85]--------> [25200 * 85]
    // [20 * 20 * 3 * 85]----/
    float* tmpOutput = m_output;
    for (size_t i = 0; i < 3; i++) {
        auto outputShape = m_task->getOutputShape(m_outputTensors[i]);
        const float *predOutput = m_task->getOutputTensor(m_outputTensors[i]);

        int batch = outputShape[0];
        int height = outputShape[1];
        int width = outputShape[2];
        int channel = outputShape[3];

        for (int j = 0; j < height; j++) {      // 80/40/20
            for (int k = 0; k < width; k++) {   // 80/40/20
                int anchorIdx = 0;
                for (int l = 0; l < 3; l++) {   // 3
                    for (int m = 0; m < channel / 3; m++) {     // 85
                        if (m < 2) {
                            float value = *predOutput;
                            float gridValue = m == 0 ? k : j;
                            *tmpOutput = (value * 2 - 0.5 + gridValue) * strides[i];
                        } else if (m < 4) {
                            float value = *predOutput;
                            *tmpOutput = value * value * 4 * anchorGrid[i][anchorIdx++];
                        } else {
                            *tmpOutput = *predOutput;
                        }
                        tmpOutput++;
                        predOutput++;
                    }
                }
            }
        }
    }
    
    std::vector<int> boxIndexs;
    std::vector<float> boxConfidences;
    std::vector<ts::ObjectData> winList;

    for (int i = 0; i< MODEL_OUTPUT_GRIDS; i++) {
        float boxConfidence = m_output[i * MODEL_OUTPUT_CHANNEL + 4];
        if (boxConfidence > 0.001) {
            boxIndexs.push_back(i);
            boxConfidences.push_back(boxConfidence);
        }
    }

    float curMaxScore = 0.0f;
    float curMinScore = 0.0f;

    for (size_t i = 0; i < boxIndexs.size(); i++) {
        int curIdx = boxIndexs[i];
        float curBoxConfidence = boxConfidences[i];

        for (int j = 5; j < MODEL_OUTPUT_CHANNEL; j++) {
            float score = curBoxConfidence * m_output[curIdx * MODEL_OUTPUT_CHANNEL + j];
            if (score > m_confThresh) {
                ts::ObjectData rect;
                rect.width = m_output[curIdx * MODEL_OUTPUT_CHANNEL + 2];
                rect.height = m_output[curIdx * MODEL_OUTPUT_CHANNEL + 3];
                rect.x = std::max(0, static_cast<int>(m_output[curIdx * MODEL_OUTPUT_CHANNEL] - rect.width / 2)) - m_xOffset;
                rect.y = std::max(0, static_cast<int>(m_output[curIdx * MODEL_OUTPUT_CHANNEL + 1] - rect.height / 2)) - m_yOffset;

                rect.width /= m_scale;
                rect.height /= m_scale;
                rect.x /= m_scale;
                rect.y /= m_scale;
                rect.confidence = score;
                rect.label = j - 5;

                winList.push_back(rect);
            }
        }
    }

    winList = nms(winList, m_nmsThresh);

    for (size_t i = 0; i < winList.size(); i++) {
        if (winList[i].width >= m_minBoxBorder || winList[i].height >= m_minBoxBorder) {
            if (!m_roi.empty()) {
                winList[i].x += m_roi.x;
                winList[i].y += m_roi.y;
            }
            results.push_back(winList[i]);
        }
    }

    return true;
}
