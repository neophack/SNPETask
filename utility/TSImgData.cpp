/*
 * Copyright © 2020-2021 Thunder Software Technology Co., Ltd.
 * All rights reserved.
 */

#include "TSStruct.h"
#include "imgbuf.hpp"

ts::TSImgData::TSImgData() {
    impl = new ImgBuf();
}

ts::TSImgData::TSImgData(
        int32_t  width,
        int32_t  height,
        int32_t  format,
        uint8_t* data,
        int32_t  stride) {
    impl = new ImgBuf(width, height, format, data, stride);
}

ts::TSImgData::TSImgData(const TSImgData& img) {
    delete reinterpret_cast<ImgBuf*>(impl);
    impl = new ImgBuf(*(reinterpret_cast<ImgBuf*>(img.impl)));
}

ts::TSImgData ts::TSImgData::roi(TSRect_T<int> reg) const {
    assert(reg.x >= 0 && reg.y >= 0 && reg.x + reg.width <= this->width() && reg.y + reg.height <= this->height());
    ImgBuf* src_buf = reinterpret_cast<ImgBuf*>(this->impl);
    uint8_t* data = reinterpret_cast<uint8_t*>(src_buf->data()) + reg.y * (src_buf->stride())
            + reg.x * (src_buf->channel()) * (src_buf->pixel_len());
    ts::TSImgData dst;
    dst.impl = new ImgBuf({reg.width, reg.height}, src_buf->format(), data, src_buf->stride());
    return dst;
}

void ts::TSImgData::operator= (const TSImgData &t) {
    *(reinterpret_cast<ImgBuf*>(impl)) = *(reinterpret_cast<ImgBuf*>(t.impl));
}

void ts::TSImgData::copyTo(TSImgData& dst) const {
    reinterpret_cast<ImgBuf*>(impl)->copyTo(*(reinterpret_cast<ImgBuf*>(dst.impl)));
    return;
}

ts::TSImgData::~TSImgData() {
    delete reinterpret_cast<ImgBuf*>(impl);
}

uint8_t* ts::TSImgData::data() const {
    return reinterpret_cast<uint8_t*>(reinterpret_cast<ImgBuf*>(impl)->data());
}

int32_t ts::TSImgData::width() const {
    return reinterpret_cast<ImgBuf*>(impl)->width();
}

int32_t ts::TSImgData::height() const {
    return reinterpret_cast<ImgBuf*>(impl)->height();
}

int32_t ts::TSImgData::stride() const {
    return reinterpret_cast<ImgBuf*>(impl)->stride();
}

int32_t ts::TSImgData::channels() const {
    return reinterpret_cast<ImgBuf*>(impl)->channel();
}

bool ts::TSImgData::empty() const {
    return reinterpret_cast<ImgBuf*>(impl)->empty();
}

int ts::TSImgData::format() const {
    return reinterpret_cast<ImgBuf*>(impl)->format();
}
