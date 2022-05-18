/*
 * Copyright (c) 2012-2022
 * All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 * You may not use, copy, distribute, modify, transmit in any form this file
 * except in compliance with THUNDERCOMM in writing by applicable law.
 * 
 * @Description: Abstraction of yolov5s object detection algorithm inference APIs.
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-05-17 20:26:39
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-05-18 09:31:30
 */

#ifndef __TS_YOLOV5S_H__
#define __TS_YOLOV5S_H__

#include <vector>
#include <string>

#include "TSStruct.h"

namespace ts
{

/**
 * @brief: Object detection result structure.
 * 
 */
class ObjectData : public TSRect_T<int> {
public:
    // Bounding box information: top-left coordinate and width, height
    using TSRect_T<int>::TSRect_T;
    // Confidence of this bounding box
    float confidence = -1.0f;
    // The label of this Bounding box
    int label = -1;
    // Time cost of detecting this frame
    size_t time_cost = 0;
};

/**
 * @brief: Object detection instance object.
 */
class TSObjectDetection {
public:
    /**
     * @brief: Constructor.
     * @Author: Ricardo Lu
     * @param {*}
     * @return {*}
     */
    TSObjectDetection();

    /**
     * @brief: Deconstructor.
     * @Author: Ricardo Lu
     * @param {*}
     * @return {*}
     */
    ~TSObjectDetection();

    /**
     * @brief: Init a object detection instance, must be called before inference.
     * @Author: Ricardo Lu
     * @param {std::string&} model_path: Absolute path of model file.
     * @param {runtime_t} runtime: Inference hardware runtime.
     * @return {bool} true if init successfully, false if failed.
     */    
    bool Init(const std::string& model_path, const runtime_t runtime);

    /**
     * @brief: Release relevant resources.
     * @Author: Ricardo Lu
     * @param {*}
     * @return {*}
     */
    bool Deinit();

    /**
     * @brief: Balance the accuracy and recall.
     * Setting these to a higher value can be used to improve verification accuracy.
     * But the recall might be reduced. It means some fuzzy objects might not be detected.
     * You can change any threshold any time, no matter whether it is initialized or running.
     * @Author: Ricardo Lu
     * @param {float&} conf_thresh: Confidence threshold of inference output probability.
     * @param {float&} nms_thresh: Threshold of NMS task, [0.0f, 1.0f]
     * @return {bool} true if setter successfully, false if failed.
     */
    bool SetScoreThreshold(const float& conf_thresh, const float& nms_thresh);

    /**
     * @brief: Core method of object detection.
     * @Author: Ricardo Lu
     * @param {ts::TSImgData&} image: A RGB format image needs to be detected.
     * @param {std::vector<std::vector<ts::ObjectData> >&} results: Detection results vector for each image.
     * @return {bool} true if detect successfullly, false if failed.
     */
    bool Detect(const ts::TSImgData& image, std::vector<ts::ObjectData>& results);

    /**
     * @brief: Check object detection instance initialization state.
     * @Author: Ricardo Lu
     * @return {bool} true if initialized, false if not.
     */
    bool IsInitialized();

private:
    // object detection handler: all methods of TSObjectDetection will be forward to it.
    void* impl = nullptr;
};

} // namespace ts


#endif // __TS_YOLOV5S_H__
