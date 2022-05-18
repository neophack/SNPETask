/*
 * Copyright © 2020-2021 Thunder Software Technology Co., Ltd.
 * All rights reserved.
 */

#include <opencv2/opencv.hpp>
#include "imgbuf.hpp"

int cvtTypeImgData2Buf(int format) {
    if (TYPE_BGR_F32 == format || TYPE_RGB_F32 == format) {
        return RDC_32FC3;
    } else if (TYPE_BGR_U8 == format || TYPE_RGB_U8 == format) {
        return RDC_8UC3;
    } else if (TYPE_C1_F32 == format) {
        return RDC_32FC1;
    } else if (TYPE_C1_U8 == format) {
        return RDC_8UC1;
    }
    return RDC_UNKNOWN;
}

int cvt_rdctype2cv(int type) {
    int cv_type = 0;
    switch (type) {
        case RDC_32FC1:
            cv_type = CV_32FC1;
            break;
        case RDC_32FC3:
            cv_type = CV_32FC3;
            break;
        case RDC_32FC4:
            cv_type = CV_32FC4;
            break;
        case RDC_8UC1:
            cv_type = CV_8UC1;
            break;
        case RDC_8UC3:
            cv_type = CV_8UC3;
            break;
        case RDC_8UC4:
            cv_type = CV_8UC4;
            break;
    }
    return cv_type;
}

int cvt_cvtype2rdc(int type) {
    int cv_type = 0;
    switch (type) {
        case CV_32FC1:
            cv_type = RDC_32FC1;
            break;
        case CV_32FC3:
            cv_type = RDC_32FC3;
            break;
        case CV_32FC4:
            cv_type = RDC_32FC4;
            break;
        case CV_8UC1:
            cv_type = RDC_8UC1;
            break;
        case CV_8UC3:
            cv_type = RDC_8UC3;
            break;
        case CV_8UC4:
            cv_type = RDC_8UC4;
            break;
    }
    return cv_type;
}

void cvt_Mat2ImgBuf(const cv::Mat& src, ImgBuf& dst, bool deep_cp = false) {
    if (NULL == src.data || src.empty()) {
        TS_ERROR_LOG("ERROR: Failed in rdc::cvt_Mat2ImgBuf cased by empty source.");
        return;
    }
    if (true == deep_cp) {
        dst.create(src.cols, src.rows, cvt_cvtype2rdc(src.type()));
        switch (dst.pixel_len()) {
            case RDC_8U:
                ImgBuf::copydata(reinterpret_cast<uchar*>(src.data), reinterpret_cast<uchar*>(dst.data()), \
                    src.step, dst.stride(), dst.height());
                break;
            case RDC_32F:
                ImgBuf::copydata(reinterpret_cast<float*>(src.data), reinterpret_cast<float*>(dst.data()), \
                    src.step, dst.stride(), dst.height());
                break;
        }
    } else {
        dst.create(src.cols, src.rows, cvt_cvtype2rdc(src.type()), src.data, src.step);
    }
}

void cvt_ImgBuf2Mat(const ImgBuf& src, cv::Mat& dst, bool deep_cp = false) {
    if (NULL == src.data() || src.empty()) {
        TS_ERROR_LOG("ERROR: Failed in rdc::cvt_ImgBuf2Mat cased by empty source.");
        return;
    }
    if (true == deep_cp) {
        dst.create(src.height(), src.width(), cvt_rdctype2cv(src.type()));
        switch (src.pixel_len()) {
            case RDC_8U:
                ImgBuf::copydata(reinterpret_cast<uchar*>(src.data()), reinterpret_cast<uchar*>(dst.data), \
                    src.stride(), dst.step, src.height());
                break;
            case RDC_32F:
                ImgBuf::copydata(reinterpret_cast<float*>(src.data()), reinterpret_cast<float*>(dst.data), \
                    src.stride(), dst.step, src.height());
                break;
        }
    } else {
        dst = cv::Mat(src.height(), src.width(), cvt_rdctype2cv(src.type()), src.data(), src.stride());
    }
    return;
}

ImgBuf::ImgBuf() {
    create(0, 0, RDC_UNKNOWN, NULL, 0);
}

ImgBuf::ImgBuf(int width, int height, RDCImgType type, void* data, int stride) {
    create(width, height, type, data, stride);
}

ImgBuf::ImgBuf(cv::Size size, RDCImgType type, void* data, int stride) {
    create(size.width, size.height, type, data, stride);
}

ImgBuf::ImgBuf(const ImgBuf& img) {
    create(img.width_, img.height_, img.format_, NULL, img.stride_);
    int buf_len = stride_ * height_ / pixel_len_;
    switch (pixel_len_) {
        case RDC_8U:
            std::copy(reinterpret_cast<uchar*>(img.data_), reinterpret_cast<uchar*>(img.data_) + buf_len, \
                reinterpret_cast<uchar*>(this->data_));
            break;
        case RDC_32F:
            std::copy(reinterpret_cast<float*>(img.data_), reinterpret_cast<float*>(img.data_) + buf_len, \
                reinterpret_cast<float*>(this->data_));
            break;
    }
}

ImgBuf::~ImgBuf() {
    tryDeallocate();
}

void ImgBuf::operator= (const ImgBuf &t) {
    this->size_ = t.size_;
    this->width_ = t.width_;
    this->height_ = t.height_;
    this->stride_ = t.stride_;
    this->channel_ = t.channel_;
    this->pixel_len_ = t.pixel_len_;
    this->type_ = t.type_;
    this->allocated = false;
    this->data_ = t.data_;
    return;
}

ImgBuf ImgBuf::row(int j) const {
    ImgBuf dst;
    if (j >= this->height_) {
        TS_ERROR_LOG("ERROR: In rdc::ImgBuf::row, the row number must be lower than image height!");
        return dst;
    }
    switch (pixel_len_) {
        case RDC_8U:
            dst.create(this->width_, 1, this->type_, ptr<unsigned char>(j), this->stride_);
            break;
        case RDC_32F:
            dst.create(this->width_, 1, this->type_, ptr<float>(j), this->stride_);
            break;
    }
    return dst;
}

void ImgBuf::create(int width, int height, int format, void *data, int stride) {
    tryDeallocate();
    format_ = format;
    type_ = cvtTypeImgData2Buf(format);
    channel_ = ImgBuf::channels_of_type(type_);
    pixel_len_ = ImgBuf::pixellen_of_type(type_);
    width_ = width; height_ = height;
    stride_ = std::max(stride, width * channel_ * pixel_len_);
    size_ = cv::Size(width, height);
    data_ = tryAllocate(data);
    return;
}

void ImgBuf::copyTo(ImgBuf &dst) const {
    if (this->empty()) {
        TS_ERROR_LOG("ERROR: The source ImgBuf is empty, and it cannot be copied!");
        return;
    }
#if 1
    dst.create(this->width_, this->height_, this->type_, NULL, dst.stride());
    cv::Mat src_tmp, dst_tmp;
    cvt_ImgBuf2Mat(*this, src_tmp, false);
    cvt_ImgBuf2Mat(dst, dst_tmp, false);
    src_tmp.copyTo(dst_tmp);
#else
    dst.create(this->width_, this->height_, this->type_, NULL, stride_);
    int buf_len = stride_ * height_ / this->pixel_len_;
    switch (pixel_len_) {
        case RDC_8U:
            std::copy(reinterpret_cast<uchar*>(this->data_), reinterpret_cast<uchar*>(this->data_) + buf_len, \
                reinterpret_cast<uchar*>(dst.data_));
            break;
        case RDC_32F:
            std::copy(reinterpret_cast<float*>(this->data_), reinterpret_cast<float*>(this->data_) + buf_len, \
                reinterpret_cast<float*>(dst.data_));
            break;
    }
#endif
    return;
}

void ImgBuf::push_back(const ImgBuf& t) {
    if (this->empty()) {
        t.copyTo(*this);
        return;
    }
    if (this->width_ != t.width_         ||
        this->stride_ != t.stride_       ||
        this->channel_ != t.channel_     ||
        this->type_ != t.type_           ||
        this->pixel_len_ != t.pixel_len_ ||
        RDC_UNKNOWN == this->type_) {
        TS_ERROR_LOG("ERROR: Func \"ImgBuf::push_back\" same types required: (width == t.width && type == t.type && etc.)");
        return;
    }
    cv::Size target_size(this->width_, this->height_ + t.height_);

    int target_length = this->stride_ * (this->height_ + t.height_) / pixel_len_;
    int top_length = this->stride_ * this->height_ / pixel_len_;
    int bottom_length = this->stride_ * t.height_ / pixel_len_;
    void* target_data = NULL;
    switch (pixel_len_) {
        case RDC_8U:
            target_data = new char[target_length];
            std::copy(reinterpret_cast<char*>(this->data_), reinterpret_cast<char*>(this->data_) + top_length, \
                reinterpret_cast<char*>(target_data));
            std::copy(reinterpret_cast<char*>(t.data_), reinterpret_cast<char*>(t.data_) + bottom_length, \
                reinterpret_cast<char*>(target_data) + top_length);
            break;
        case RDC_32F:
            target_data = new float[target_length];
            std::copy(reinterpret_cast<float*>(this->data_), reinterpret_cast<float*>(this->data_) + top_length, \
                reinterpret_cast<float*>(target_data));
            std::copy(reinterpret_cast<float*>(t.data_), reinterpret_cast<float*>(t.data_) + bottom_length, \
                reinterpret_cast<float*>(target_data) + top_length);
            break;
    }
    create(target_size.width, target_size.height, this->type_, target_data, this->stride_);
    this->allocated = true;  // Switch the flag to true!
    return;
}

bool ImgBuf::empty() const {
    if (NULL == data_
        || 0 == width_ || 0 == stride_ || 0 == height_ || 0 == channel_
        || RDC_UNKNOWN == type_) {
        return true;
    }
    return false;
}

void* ImgBuf::tryAllocate(void* p) {
    if (NULL != p) {
        data_ = p;
        allocated = false;
    } else {
        if (0 == pixel_len_) {
            allocated = false;
            return data_;
        }
        int buf_len = stride_ * height_ / pixel_len_;
        if (0 == buf_len) {
            allocated = false;
            return data_;
        }
        switch (pixel_len_) {
            case RDC_8U:
                data_ = new char[buf_len];
                break;
            case RDC_32F:
                data_ = new float[buf_len];
                break;
        }
        allocated = true;
    }
    return data_;
}

void ImgBuf::tryDeallocate() {
    if (allocated && NULL != data_) {
        switch (pixel_len_) {
            case RDC_8U:
                delete[] reinterpret_cast<char*>(data_);
                break;
            case RDC_32F:
                delete[] reinterpret_cast<float*>(data_);
                break;
        }
    }
    data_ = NULL;
    type_ = RDC_UNKNOWN;
    allocated = false;
    return;
}

int ImgBuf::channels_of_type(RDCImgType type) {
    if (RDC_32FC1 == type || RDC_8UC1 == type) {
        return 1;
    } else if (RDC_32FC3 == type || RDC_8UC3 == type) {
        return 3;
    } else if (RDC_32FC4 == type || RDC_8UC4 == type) {
        return 4;
    }
    return 0;  // RDC_UNKNOWN.
}

int ImgBuf::pixellen_of_type(RDCImgType type) {
    if (RDC_8UC1 == type || RDC_8UC3 == type || RDC_8UC4 == type) {
        return RDC_8U;
    } else if (RDC_32FC1 == type || RDC_32FC3 == type || RDC_32FC4 == type) {
        return RDC_32F;
    }
    return 0;  // RDC_UNKNOWN.
}
