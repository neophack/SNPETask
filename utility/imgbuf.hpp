/*
 * Copyright © 2020-2021 Thunder Software Technology Co., Ltd.
 * All rights reserved.
 */

#ifndef _IMGBUF_HPP_
#define _IMGBUF_HPP_

#include <opencv2/opencv.hpp>
#include "TSStruct.h"

using RDCImgType = int;
constexpr int RDC_AUTO_STRIDE(0);

#define RDC_8U        1
#define RDC_32F       4
#define RDC_8UC1      0x00
#define RDC_8UC3      0x01
#define RDC_8UC4      0x02
#define RDC_32FC1     0x03
#define RDC_32FC3     0x04
#define RDC_32FC4     0x05
#define RDC_UNKNOWN   0x06

/** @brief Data structure of image, matrix or 2D-tensor.
 * */
class ImgBuf {
public:
    /** @brief Default constructor.
     * There are various constructors to allocate rdc::ImgBuf.
     * Using default constructor to apply an empty image object.
     * */
    ImgBuf ();

    /** @overload Custom constructor.
     * @param width The column number of 2D image.
     * @param height The row number of 2D image.
     * @format Specifying the format of the image.
     * All types has been defined in "format.h".
     * Such as RDC_8UC3 to allocate an image with 256 colors and 3 channels.
     * @param data If it equals to zero, allocate a new memory;
     * otherwise using buffer from 'data'.
     * @param stride Number of bytes each row occupies.
     * */
    ImgBuf (int width, int height, int format, void* data = nullptr, int stride = RDC_AUTO_STRIDE);
    ImgBuf (cv::Size size, int format, void* data = nullptr, int stride = 0);

    /** @brief Copy constructor.
     * Implemented by deep copy command internally.
     * */
    ImgBuf (const ImgBuf& img);

    /** @brief Default destructor.
     * */
    virtual ~ImgBuf ();

    /** @brief Shallow copy command.
     * */
    void operator= (const ImgBuf &t);

    /** @brief Create a row array shallow copy from an image.
     * Any operation will affect source image.
     * */
    ImgBuf row(int j) const;

    /** @brief Reconstruct a 2D image.
     * @param See also the rdc::ImgBuf::ImgBuf.
     * */
    void create (int width, int height, int format, void* data = nullptr, int stride = 0);

    /** @brief Deep copy command.
     * */
    void copyTo (ImgBuf& dst) const;

    /** @brief Insert a new row array at the bootom of the image.
     * In this version, the width, channel, and stride of image 't' must be same as source image.
     * */
    void push_back (const ImgBuf& t);

    /** @brief If the image is empty, then return true.
     * */
    bool empty () const;

    /** @brief Get the head pointer of row number j.
     * */
    template<typename T>
    T* ptr(unsigned int j) const {
        if (nullptr == data_ || 0 == pixel_len_) {
            return nullptr;
        }
        return reinterpret_cast<T*>(data_) + j * stride_ / this->pixel_len_;
    }

    template<typename T>
    T* at(unsigned int x, unsigned int y) const {
        if (static_cast<int>(x) >= width_ || static_cast<int>(y) >= height_ || nullptr == data_ || 0 == pixel_len_) {
            return nullptr;
        }
        return reinterpret_cast<T*>(data_) + y * stride_ / this->pixel_len_ + x;
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int stride() const { return stride_; }
    int channel() const { return channel_; }
    int type() const { return type_; }
    int format() const { return format_; }
    cv::Size size() const { return size_; }

    /** @brief Return the head pointer of image native buffer.
     * */
    void* data() const { return data_; }

    /** @brief Return the byte length of element.
     * */
    int pixel_len() const { return pixel_len_; }

    template<typename T>
    static void copydata(T* src, T* dst, int src_stride, int dst_stride, int rows) {
        int buf_len = 0;
        if (src_stride == dst_stride) {
            buf_len = src_stride / sizeof(T) * rows;
            std::copy(src, src + buf_len, dst);
        } else {
            int min_stride = std::min(src_stride, dst_stride);
            for (int j = 0; j < rows; j++) {
                T* src_rows = src + j * src_stride / sizeof(T);
                T* dst_rows = dst + j * dst_stride / sizeof(T);
                std::copy(src_rows, src_rows + min_stride / sizeof(T), dst_rows);
            }
        }
        return;
    }

private:
    cv::Size size_;
    int format_ = TYPE_UNKNOWN;
    int width_ = 0;
    int height_ = 0;
    int stride_ = 0;
    int channel_ = 0;
    int pixel_len_ = 0;
    RDCImgType type_ = RDC_UNKNOWN;
    void* data_ = nullptr;
    bool allocated = false;
    void* tryAllocate(void* p);
    void tryDeallocate();

    static int channels_of_type(RDCImgType type);
    static int pixellen_of_type(RDCImgType type);
};

#endif  // _IMGBUF_HPP_
