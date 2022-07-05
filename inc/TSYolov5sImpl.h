/*
 * Copyright (c) 2012-2022
 * All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 * You may not use, copy, distribute, modify, transmit in any form this file
 * except in compliance with THUNDERCOMM in writing by applicable law.
 * 
 * @Description: Object detection algorithm handler.
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-05-17 20:27:51
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-07-05 10:33:49
 */

#ifndef __TS_FACE_DETECTION_IMPL_H__
#define __TS_FACE_DETECTION_IMPL_H__

#include <vector>
#include <string>
#include <unistd.h>
#include <memory>

#include "SNPETask.h"
#include "TSYolov5s.h"

#define MODEL_OUTPUT_CHANNEL    85
#define MODEL_OUTPUT_GRIDS      25200    // (80 * 80 + 40 * 40 + 20 * 20) * 3

#define INPUT_TENSOR            "images"
#define OUTPUT_NODE0            "Sigmoid_199"
#define OUTPUT_NODE1            "Sigmoid_201"
#define OUTPUT_NODE2            "Sigmoid_203"
#define OUTPUT_TENSOR0          "output"
#define OUTPUT_TENSOR1          "329"
#define OUTPUT_TENSOR2          "331"

class TSObjectDetectionImpl {
public:
    TSObjectDetectionImpl();
    ~TSObjectDetectionImpl();
    bool Detect(const ts::TSImgData& image, std::vector<ts::ObjectData>& results);
    bool Initialize(const std::string& model_path, const runtime_t runtime);
    bool DeInitialize();

    bool SetScoreThresh(const float& conf_thresh, const float& nms_thresh = 0.5) noexcept {
        this->m_nmsThresh  = nms_thresh;
        this->m_confThresh = conf_thresh;
        return true;
    }

    bool IsInitialized() const {
        return m_isInit;
    }

    static std::vector<ts::ObjectData> nms(std::vector<ts::ObjectData> winList, const float& nms_thresh) {
        if (winList.empty()) {
            return winList;
        }

        std::sort(winList.begin(), winList.end(), [] (const ts::ObjectData& left, const ts::ObjectData& right) {
            if (left.confidence > right.confidence) {
                return true;
            } else {
                return false;
            }
        });

        std::vector<bool> flag(winList.size(), false);
        for (int i = 0; i < winList.size(); i++) {
            if (flag[i]) {
                continue;
            }

            for (int j = i + 1; j < winList.size(); j++) {
                if (ts::calcIoU(
                        reinterpret_cast<const ts::TSRect_T<int>*>(&winList[i]),
                        reinterpret_cast<const ts::TSRect_T<int>*>(&winList[j])) > nms_thresh) {
                    flag[j] = true;
                }
            }
        }

        std::vector<ts::ObjectData> ret;
        for (int i = 0; i < winList.size(); i++) {
            if (!flag[i])
                ret.push_back(winList[i]);
        }

        return ret;
    }

private:
    bool m_isInit = false;

    bool PreProcess(const ts::TSImgData& frame);
    bool PostProcess(std::vector<ts::ObjectData>& results);

    // to-do: aic inference task resources
    std::unique_ptr<snpetask::SNPETask> m_task;
    std::vector<std::string> m_outputLayers;
    std::vector<std::string> m_outputTensors;

    ts::TSRect_T<int> m_roi = {0, 0, 0, 0};
    float* m_output;
    uint32_t m_minBoxBorder = 16;
    float m_nmsThresh = 0.5f;
    float m_confThresh = 0.5f;
    float m_scaleWidth;
    float m_scaleHeight;
    float m_scale;
    int m_xOffset, m_yOffset;
};

#endif // __TS_FACE_DETECTION_IMPL_H__
