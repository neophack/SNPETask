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
 * @LastEditTime: 2022-05-17 21:25:58
 */

#include  <math.h>

#include <opencv2/opencv.hpp>

#include "TSYolov5sImpl.h"

TSObjectDetectionImpl::TSObjectDetectionImpl() : m_task(nullptr) {

}

TSObjectDetectionImpl::~TSObjectDetectionImpl() {
    DeInitialize();
}

bool TSObjectDetectionImpl::Initialize(const std::string& model_path, const runtime_t runtime)
{


    m_isInit = true;
    return true;
}

bool TSObjectDetectionImpl::DeInitialize()
{
    if (m_task) {
        m_task->deinitialize();
        m_task = nullptr;
    }

    m_isInit = false;
    return true;
}

bool TSObjectDetectionImpl::PreProcess(const ts::TSImgData& image)
{
    auto input_shape = m_task->getInputShape();

    size_t batch = input_shape[0];
    size_t input_h = input_shape[1];
    size_t input_w = input_shape[2];
    size_t channel = input_shape[3];

    int data_type;
    if (m_bufferType == ts::BufferType::HEAP) {
        data_type = CV_32FC3;
    } else if (m_bufferType == ts::BufferType::DMA_BUF) {
        data_type = CV_8UC3;
    }

    if (m_task->getInputTensor() == nullptr) {
        TS_ERROR_LOG("Empty input tensor");
        return false;
    }

    cv::Mat input(input_h, input_w, data_type, m_task->getInputTensor(), input_w * channel);

    if (image.empty()) {
        TS_ERROR_LOG("Invalid image!");
        return false;
    }

    int img_format = image.format();
    if (img_format != TYPE_BGR_U8 && img_format != TYPE_RGB_U8) {
        TS_ERROR_LOG("Invaild image format %d, expected to be rgb or bgr!", img_format);
        return false;
    }

    int img_w = image.width();
    int img_h = image.height();
    float scale_w = (1.0f * input_w) / img_w;
    float scale_h = (1.0f * input_h) / img_h;
    m_scaleWidth =  scale_w;
    m_scaleHeight = scale_h;

    cv::Mat image_tmp(img_h, img_w, CV_8UC3, image.data(), image.stride());
    if (img_format == TYPE_BGR_U8) {
        cv::cvtColor(image_tmp, image_tmp, cv::COLOR_BGR2RGB);
    }

    cv::Mat crop(input, cv::Rect(0, 0, input_w, input_h));
    cv::Mat resize_img;
    cv::resize(image_tmp, resize_img, cv::Size(input_w, input_h), cv::INTER_LINEAR);
    if (m_bufferType == ts::BufferType::HEAP) {
        resize_img.convertTo(crop, CV_32FC3);
    } else if (m_bufferType == ts::BufferType::DMA_BUF) {
        resize_img.convertTo(crop, CV_8UC3);
    }
    crop = (crop - 127.5) / 127.5;
}

bool TSObjectDetectionImpl::Detect(const ts::TSImgData& image,
    std::vector<ts::ObjectData>& vec_res)
{

    PreProcess(image);

    if (m_task->asyncExec()) {
        TS_ERROR_LOG("AICTask asyncExec failed.");
        return false;
    }

    PostProcess(vec_res);

    return true;
}

bool TSObjectDetectionImpl::PostProcess(std::vector<ts::ObjectData> &vec_res)
{
    auto input_shape = m_task->getInputShape();
    size_t mHeight = input_shape[1];
    size_t mWidth = input_shape[2];

    float mStrides[3] = {8, 16, 32};
    std::vector<ts::ObjectData> winList;

    for (size_t i = 0; i < 3; i++) {
        auto output_shape = m_task->getOutputShape(i * 2);
        const float *out_score = m_task->getOutputTensor(i * 2 + 0);
        const float *out_loc = m_task->getOutputTensor(i * 2 + 1);

        int batch = output_shape[0];
        int height = output_shape[1];
        int width = output_shape[2];
        int channel = output_shape[3];

        float conf = 0;
        float max_idx = 0;
        float preXCenter = 0;
        float preYCenter = 0;

        for (int j = 0; j < height; j++) {
            preYCenter = j * mStrides[i] + mStrides[i] / 2.0;
            for (int k = 0; k < width; k++) {
                preXCenter = k * mStrides[i] + mStrides[i] / 2.0;
                conf = out_score[j * width * channel + k * channel + 0];
                max_idx = 0;
                for (int m = 1; m < channel; m++) {
                    float tmp_score = out_score[j * width * channel + k * channel + m];
                    if (tmp_score > conf) {
                        conf = tmp_score;
                        max_idx = m;
                    }
                }

                if (conf >= m_confThresh) {
                    ts::ObjectData rect;
                    rect.confidence = conf;
                    rect.label = max_idx;

                    float xMin = preXCenter - out_loc[(j * width + k) * 4 + 0] * mStrides[i];
                    float yMin = preYCenter - out_loc[(j * width + k) * 4 + 1] * mStrides[i];
                    float xMax = preXCenter + out_loc[(j * width + k) * 4 + 2] * mStrides[i];
                    float yMax = preYCenter + out_loc[(j * width + k) * 4 + 3] * mStrides[i];

                    if (xMin < mWidth - 1 && xMax > 0 && yMin < mHeight - 1 && yMax > 0) {
                        xMin /= m_scaleWidth;
                        yMin /= m_scaleHeight;
                        xMax /= m_scaleWidth;
                        yMax /= m_scaleHeight;
                        rect.x = std::max(0, static_cast<int>(xMin));
                        rect.y = std::max(0, static_cast<int>(yMin));
                        rect.width  = xMax - rect.x;
                        rect.height = yMax - rect.y;
                        winList.push_back(rect);
                    }
                }
            }
        }
    }

    winList = nms(winList, m_nmsThresh);

    for (size_t i = 0; i < winList.size(); i++) {
        if (winList[i].width >= m_minBoxBorder || winList[i].height >= m_minBoxBorder) {
            vec_res.push_back(winList[i]);
        }
    }
    return true;
}
