/*
 * Copyright (c) 2012-2022
 * All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 * You may not use, copy, distribute, modify, transmit in any form this file
 * except in compliance with THUNDERCOMM in writing by applicable law.
 * 
 * @Description: Common data structure header.
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-05-17 20:32:59
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-05-18 12:08:46
 */

#pragma once

#include <algorithm>
#include <functional>
#include <math.h>


#ifndef EXPORT_API
#define EXPORT_API __attribute__ ((visibility("default")))
#endif

// Log print macro define.
#define TS_ERROR_LOG(msg, ...)  \
    printf("** ERROR: <%s:%s:%d>: " msg "\n", \
        __FILE__, __func__, __LINE__, ##__VA_ARGS__)

#define TS_INFO_LOG(msg, ...) \
    printf("** INFO:  <%s:%s:%d>: " msg "\n", \
        __FILE__, __func__, __LINE__, ##__VA_ARGS__)

#define TS_WARN_LOG(msg, ...) \
    printf("** WARN:  <%s:%s:%d>: " msg "\n", \
        __FILE__, __func__, __LINE__, ##__VA_ARGS__)


// Inference hardware runtime.
typedef enum runtime {
    CPU = 0,
    GPU,
    GPU_16,
    DSP,
    AIP
}runtime_t;

template <typename Dtype>
struct DPair {
    Dtype x; Dtype y;

    DPair() {}
    ~DPair() {}

    DPair(Dtype v1, Dtype v2) {
        x = v1;
        y = v2;
    }
};

typedef DPair<int> pairInt;

/**
 * @brief Basic data structure group.
 */
namespace ts {

/// Base definition for point
template <typename T>
struct TSPoint_T {
    TSPoint_T() {}
    TSPoint_T(T x, T y) {
        this->x = x;
        this->y = y;
    }
    T x = T(0.0f);
    T y = T(0.0f);
    TSPoint_T<T> operator+(const TSPoint_T<T> &t) const {
        return {x + t.x, y + t.y};
    }
    TSPoint_T<T> operator/(const T &t) const {
        return {x / t, y / t};
    }
};
using TSPoint = TSPoint_T<float>;

/// Base definition for (rectangle) region
template <typename T>
struct TSRect_T {
    TSRect_T() {}
    TSRect_T(T x, T y, T width, T height) {
        this->x = x;
        this->y = y;
        this->width = width;
        this->height = height;
    }
    T x = 0.0f;
    T y = 0.0f;
    T width = 0.0f;
    T height = 0.0f;
    TSRect_T(TSPoint_T<T> lt, TSPoint_T<T> rb) {
        this->set(lt, rb);
    }
    void set(TSPoint_T<T> lt, TSPoint_T<T> rb) {
        this->x = lt.x < rb.x ? lt.x : rb.x;
        this->y = lt.y < rb.y ? lt.y : rb.y;
        T x_max = lt.x > rb.x ? lt.x : rb.x;
        T y_max = lt.y > rb.y ? lt.y : rb.y;
        this->width = x_max - this->x;
        this->height = y_max - this->y;
    }
    TSPoint_T<T> centre() const {
        return {x + width / 2, y + height / 2};
    }
    TSPoint_T<T> lt() const {
        return {x, y};
    }
    TSPoint_T<T> rb() const {
        return {x + width, y + height};
    }
    bool empty() const {
        if (0 == this->width || 0 == this->height) {
            return true;
        } else {
            return false;
        }
    }
    T area() const {
        return this->width * this->height;
    }
    TSRect_T<T> operator&(const TSRect_T<T> &t) const {
        TSPoint_T<T> lt;
        TSPoint_T<T> rb;
        lt.x = std::max(this->lt().x, t.lt().x);
        lt.y = std::max(this->lt().y, t.lt().y);
        rb.x = std::min(this->rb().x, t.rb().x);
        rb.y = std::min(this->rb().y, t.rb().y);
        if (lt.x < rb.x && lt.y < rb.y) {
            return TSRect_T<T>(lt, rb);
        } else {
            return TSRect_T<T>();
        }
    }
    bool operator!=(const TSRect_T<T> &t) const {
        if (std::fabs(this->x - t.x) + std::fabs(this->y - t.y) < 0.00001) {
            return true;
        } else {
            return false;
        }
    }
};
using TSRect = TSRect_T<float>;

template <typename T>
static float calcIoU(const TSRect_T<T>* a, const TSRect_T<T>* b) {
    float xOverlap = std::max(
        0.,
        std::min(a->x + a->width, b->x + b->width) - std::max(a->x, b->x) + 1.);
    float yOverlap = std::max(
        0.,
        std::min(a->y + a->height, b->y + b->height) - std::max(a->y, b->y) + 1.);
    float intersection = xOverlap * yOverlap;
    float unio =
        (a->width + 1.) * (a->height + 1.) +
        (b->width + 1.) * (b->height + 1.) - intersection;
    return intersection / unio;
}

/**
 *  @brief Package of image buffer.\n
 */
class EXPORT_API TSImgData {
public:
    /**
     *  @brief Default constructor.
     *  There are various constructors to allocate ts::TSImgData.
     *  Using default constructor to apply an empty image object.
     */
    EXPORT_API TSImgData();

    /**
     *  @brief Package an image using an allocated memory or not.
     *  @param[in] width  Image width.
     *  @param[in] height Image height.
     *  @param[in] type   Only TYPE_XXX_U8 can be accepted.
     *  @param[in] data   Buffer head address.
     *  @todo For wide-bit RGBD camera, TYPE_XXX_F32 will be supported in the future.
     *  @param[in] stride Number of Bytes each image row occupies.
     */
    EXPORT_API TSImgData(int32_t width, int32_t height,
            int32_t type = TYPE_BGR_U8, uint8_t* data = nullptr, int32_t stride = -1);

    /**
     *  @brief Copy constructor.
     *  Implemented by deep copy command internally.
     */
    EXPORT_API TSImgData(const TSImgData& img);

    /**
     *  @brief TSImgData is just a package of image.
     *  It does not have any responsible for buffer management and cleanup.
     */
    EXPORT_API virtual ~TSImgData();

    /**
     *  @brief Get the head address.
     *  @retval nullptr The TSImgData is empty
     */
    EXPORT_API uint8_t* data() const;

    /**
     *  @brief Get the width of image.
     *  @retval 0 The TSImgData is empty
     */
    EXPORT_API int32_t width() const;

    /**
     *  @brief Get the height of image.
     *  @retval 0 The TSImgData is empty
     */
    EXPORT_API int32_t height() const;

    /**
     *  @brief Get number of Bytes each image row occupies.
     *  @retval 0 The TSImgData is empty
     */
    EXPORT_API int32_t stride() const;

    /**
     *  @brief Get image channels.
     *  @retval 0 The TSImgData is empty
     */
    EXPORT_API int32_t channels() const;

    /**
     *  @brief Check if the image is empty.
     *  @retval true  The TSImgData is empty.
     *  @retval false The TSImgData is not empty.
     */
    EXPORT_API bool empty() const;

    /**
     * @brief Get the format the image.
     * @retval @see IMAGE_TYPE
     */
    EXPORT_API int format() const;

    /**
     *  @brief Crop a sub region in an image.
     *  @param[in] reg The interested region.
     *  @return The ROI image using shallow copy.
     */
    EXPORT_API TSImgData roi(TSRect_T<int> reg) const;

    /**
     *  @brief Shallow copy command.
     */
    EXPORT_API void operator=(const TSImgData &t);

    /**
     *  @brief Deep copy command.
     */
    EXPORT_API void copyTo(TSImgData& dst) const;

private:
    /// Context
    void* impl = nullptr;
};

};  // namespace ts