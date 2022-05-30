/*
 * Copyright(c) 2012-2022
 * All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 * You may not use, copy, distribute, modify, transmit in any form this file
 * except in compliance with THUNDERCOMM in writing by applicable law.
 * 
 * @Description: Initilize version of encapsulation of object-deteciton algorithm developed on SNPE.
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-05-19 11:08:17
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-05-30 13:30:50
 */

//
// headers included
//
#include <memory>
#include <algorithm>
#include <fstream>

#include "TSYolov5s.h"
#include "AlgYolov5s.h"

//
// algorithm arguments
//
typedef struct _AlgConfig {
    std::string       modelPath{ "/opt/thundersoft/algs/model/yolov5s.dlc" };
    ts::TSRect_T<int> roi { ts::TSRect_T<int>(100, 100, 1720, 880)};
    float             nmsThresh{ 0.5 };
    float             confThresh{ 0.5 };
    runtime_t         runtime{ DSP };
    std::string       labelPath{"/opt/thundersoft/configs/yolov5s.txt"};
} AlgConfig;

//
// AlgCore
//} 
typedef struct _AlgCore {
    AlgConfig                cfg_;
    ts::TSObjectDetection*   alg_{ NULL };
    std::vector<std::string> labels_ {};
    TsPutResult              cb_put_result_ { nullptr };
    TsPutResults             cb_put_results_{ nullptr };
    void*                    cb_user_data_  { nullptr };
} AlgCore;

static runtime_t string2runtime(std::string& device)
{
    std::transform(device.begin(), device.end(), device.begin(),
        [](unsigned char ch){ return tolower(ch); }
    );

    if (0 == device.compare("cpu")) {
        return CPU;
    } else if (0 == device.compare("gpu")) {
        return GPU;
    } else if (0 == device.compare("dsp")) {
        return DSP;
    } else if (0 == device.compare("aip")) {
        return AIP;
    } else { 
        return DSP;
    }
}

static bool parse_args(AlgConfig& config, const std::string& data)
{
    JsonParser* parser = NULL;
    JsonNode*   root   = NULL;
    JsonObject* object = NULL;
    GError*     error  = NULL;
    bool        ret    = FALSE;
    
    if (!(parser = json_parser_new())) {
        TS_ERR_MSG_V("Failed to new a object with type JsonParser");
        return FALSE;
    }

    if (json_parser_load_from_data(parser,(const gchar *) data.data(), 
        data.length(), &error)) {
        if (!(root = json_parser_get_root(parser))) {
            TS_ERR_MSG_V("Failed to get root node from JsonParser");
            goto done;
        }

        if (JSON_NODE_HOLDS_OBJECT(root)) {
            if (!(object = json_node_get_object(root))) {
                TS_ERR_MSG_V("Failed to get object from JsonNode");
                goto done;
            }

            if (json_object_has_member(object, "model-path")) {
                std::string p((const char*)json_object_get_string_member(
                    object, "model-path"));
                TS_INFO_MSG_V("\tmodel-path:%s", p.c_str());
                config.modelPath = p;                
            }

            if (json_object_has_member(object, "runtime")) {
                std::string r((const char*)json_object_get_string_member(
                    object, "runtime"));
                TS_INFO_MSG_V("\truntime:%s", r.c_str());
                config.runtime = string2runtime(r);                
            }

            if (json_object_has_member(object, "label-path")) {
                std::string r((const char*)json_object_get_string_member(
                    object, "label-path"));
                TS_INFO_MSG_V("\tlabel-path:%s", r.c_str());
                config.runtime = string2runtime(r);                
            }

            if (json_object_has_member(object, "nms-thresh")) {
                gdouble n = json_object_get_double_member(object, "nms-thresh");
                TS_INFO_MSG_V("\tnms-thresh:%f", n);
                config.nmsThresh = (float)n;
            }

            if (json_object_has_member(object, "conf-thresh")) {
                gdouble c = json_object_get_double_member(object, "conf-thresh");
                TS_INFO_MSG_V("\tconf-thresh:%f", c);
                config.confThresh = (float)c;
            }

            if (json_object_has_member(object, "roi")) {
                JsonObject* r = json_object_get_object_member(object, "roi");

                if (json_object_has_member(r, "x")) {
                    int x = json_object_get_int_member(r, "x");
                    TS_INFO_MSG_V("\tx:%d", x);
                    config.roi.x = x;
                }

                if (json_object_has_member(r, "y")) {
                    int y = json_object_get_int_member(r, "y");
                    TS_INFO_MSG_V("\ty:%d", y);
                    config.roi.y = y;
                }

                if (json_object_has_member(r, "w")) {
                    int w = json_object_get_int_member(r, "w");
                    TS_INFO_MSG_V("\tw:%d", w);
                    config.roi.width = w;
                }

                if (json_object_has_member(r, "h")) {
                    int h = json_object_get_int_member(r, "h");
                    TS_INFO_MSG_V("\th:%d", h);
                    config.roi.height = h;
                }
            }
        }
    } else {
        TS_ERR_MSG_V("Failed to parse json string %s(%s)\n",
            error->message, data.c_str());
        g_error_free(error);
        goto done;
    }

    ret = TRUE;

done:
    g_object_unref(parser);

    return ret;
}

//
// results_to_json_object
//
static JsonObject* results_to_json_object(const std::vector<ts::ObjectData>& results, void* alg)
{
    AlgCore* a = (AlgCore*)alg;
    JsonObject* result = json_object_new();
    JsonArray*  jarray = json_array_new();
    JsonObject* jobject = NULL;

    if (!result || !jarray) {
        TS_ERR_MSG_V("Failed to new a object with type JsonXyz");
        return NULL;
    }

    for (int i = 0; i <(int)results.size(); i ++) {
        if (!(jobject = json_object_new())) {
            TS_ERR_MSG_V("Failed to new a object with type JsonObject");
            json_array_unref(jarray);
            return NULL;
        }

        json_object_set_string_member(jobject, "name",
            a->labels_[results[i].label].c_str());
        json_object_set_string_member(jobject, "score",
            std::to_string(results[i].confidence).c_str());
        json_object_set_string_member(jobject, "x",
            std::to_string(results[i].x).c_str());
        json_object_set_string_member(jobject, "y",
            std::to_string(results[i].y).c_str());
        json_object_set_string_member(jobject, "width",
            std::to_string(results[i].width).c_str());
        json_object_set_string_member(jobject, "height",
            std::to_string(results[i].height).c_str());
        json_array_add_object_element(jarray, jobject);
    }
    
    json_object_set_string_member(result, "alg-name", "yolov5s");
    json_object_set_array_member (result, "alg-result", jarray);
    
    return result;
}

// 
// results_to_osd_object
//
static void results_to_osd_object(const std::vector<ts::ObjectData>& results,
    std::vector<TsOsdObject>& osd, int x, int y, int width, int height)
{
    bool alarm = false;

    for (int i = 0; i <(int)results.size(); i ++) {
        std::string text("yolov5s");
        osd.push_back(TsOsdObject((int)results[i].x,
            (int)results[i].y,(int)results[i].width,
            (int)results[i].height, 0, 255, 0,
            0, text, TsObjectType::OBJECT));
    }
}


//
// algInit
//
void* algInit(const std::string& args)
{
    TS_INFO_MSG_V("algInit called");
    
    AlgCore* a = new AlgCore();

    if (!a) {
        TS_ERR_MSG_V("Failed to new a object with type AlgCore");
        return NULL;
    }

    if (0 != args.compare("")) parse_args(a->cfg_, args);

    std::ifstream in(a->cfg_.labelPath);
    std::string line;
    while (getline(in, line)){
        a->labels_.push_back(line);
    }           

    if (!(a->alg_ = new ts::TSObjectDetection())) {
        TS_ERR_MSG_V("Failed to new a object with type TSFaceDetection");
        goto done;
    }

    a->alg_->Init(a->cfg_.modelPath, a->cfg_.runtime);

    if (!a->alg_->SetScoreThreshold(a->cfg_.confThresh, a->cfg_.nmsThresh)) {
        TS_ERR_MSG_V("Failed to set score thresh(%f, %f)",
            a->cfg_.confThresh, a->cfg_.nmsThresh);
        goto done;
    }

    return (void*)a;

done:
    if (a->alg_) {
        delete a->alg_;
    }

    delete a;

    return NULL;
}

//
// algStart
//
bool algStart(void* alg)
{
    TS_INFO_MSG_V("algStart called");

    return TRUE;
}

//
// algProc
//
std::shared_ptr<TsJsonObject> algProc(
    void* alg, const std::shared_ptr<TsGstSample>& data)
{
    AlgCore* a = (AlgCore*)alg;

    //TS_INFO_MSG_V("algProc called");

    GstSample* sample = data->GetSample();

    gint width, height, type = TYPE_RGB_U8;
    GstCaps* caps = gst_sample_get_caps(sample);
    GstStructure* structure = gst_caps_get_structure(caps, 0);
    gst_structure_get_int(structure, "width", &width);
    gst_structure_get_int(structure, "height", &height);
    std::string format((char*)gst_structure_get_string(
        structure, "format"));
    if (0 != format.compare("RGB")) {
        TS_ERR_MSG_V("Invalid format(%s!=RGB)", format.c_str());
        gst_sample_unref(sample);
        return NULL;
    }

    GstMapInfo map;
    GstBuffer* buf = gst_sample_get_buffer(sample);
    gst_buffer_map(buf, &map, GST_MAP_READ);
    ts::TSImgData image(width, height, TYPE_RGB_U8, map.data);
    gst_buffer_unmap(buf, &map);

    std::vector<ts::ObjectData> results;
    if (!a->alg_->Detect(image, results)) {
        TS_WARN_MSG_V("Failed to detect face in the image");
        //return NULL;
    }

    std::shared_ptr<TsJsonObject> jo = std::make_shared<
        TsJsonObject>(results_to_json_object(results, a));
    results_to_osd_object(results, jo->GetOsdObject(), a->cfg_.roi.x,
        a->cfg_.roi.y, a->cfg_.roi.width, a->cfg_.roi.height);

    a->cb_put_result_(jo, data, a->cb_user_data_);
    return nullptr;
}

std::shared_ptr<std::vector<std::shared_ptr<TsJsonObject>>> algProc2(void* alg,
    const std::shared_ptr<std::vector<std::shared_ptr<TsGstSample>>>& datas)
{
    // TS_INFO_MSG_V("algProc2 called");

    AlgCore* a = (AlgCore*)alg;

    //TS_INFO_MSG_V("algProc called");

    std::shared_ptr<std::vector<std::shared_ptr<TsJsonObject>>> jos;
    if (!(jos = std::make_shared<std::vector<std::shared_ptr<TsJsonObject>>>())) {
        TS_ERR_MSG_V("Failed to create a new object with type std::vector");
        return nullptr;
    }

    // for (size_t i = 0; i < datas->size(); i++) {
    //     GstSample* sample = (*datas)[i]->GetSample();

    //     gint width, height, type = TYPE_RGB_U8;
    //     GstCaps* caps = gst_sample_get_caps(sample);
    //     GstStructure* structure = gst_caps_get_structure(caps, 0);
    //     gst_structure_get_int(structure, "width", &width);
    //     gst_structure_get_int(structure, "height", &height);
    //     std::string format((char*)gst_structure_get_string(
    //         structure, "format"));
    //     if (0 != format.compare("RGB")) {
    //         TS_ERR_MSG_V("Invalid format(%s!=RGB)", format.c_str());
    //         gst_sample_unref(sample);
    //         return NULL;
    //     }

    //     GstMapInfo map;
    //     GstBuffer* buf = gst_sample_get_buffer(sample);
    //     gst_buffer_map(buf, &map, GST_MAP_READ);
    //     ts::TSImgData image(width, height, TYPE_RGB_U8, map.data);
    //     std::vector<ts::TSImgData> images = {image};
    //     gst_buffer_unmap(buf, &map);

    //     std::vector<ts::ObjectData> results;
    //     if (!a->alg_->Detect(images, results)) {
    //         TS_WARN_MSG_V("Failed to detect person in the image");
    //         return NULL;
    //     }

    //     std::shared_ptr<TsJsonObject> jo = std::make_shared<
    //         TsJsonObject>(results_to_json_object(results, a));
    //     results_to_osd_object(results, jo->GetOsdObject(), a->cfg_.roi.x,
    //         a->cfg_.roi.y, a->cfg_.roi.width, a->cfg_.roi.height);

    //     jos->push_back(jo);
    // }

    a->cb_put_results_(jos, datas, a->cb_user_data_);

    return nullptr;
}

//
// algCtrl
//
bool algCtrl(void* alg, const std::string& cmd)
{
    TS_INFO_MSG_V("algCtrl called");

    return FALSE;
}

//
// algStop
//
void algStop(void* alg)
{
    TS_INFO_MSG_V("algStop called");
}

//
// algFina
//
void algFina(void* alg)
{
    AlgCore* a = (AlgCore*)alg;

    TS_INFO_MSG_V("algFina called");

    delete a->alg_;

    delete a;
}

//
// algSetCb
//
bool algSetCb(void* alg, TsPutResult cb, void* args)
{
    // TS_INFO_MSG_V("algSetCb called");

    AlgCore* a = (AlgCore*)alg;
    assert(a);

    if (cb) {
        a->cb_put_result_ = cb;
        a->cb_user_data_ = args;
    }

    return true;
}

//
// algSetCb2
//
bool algSetCb2(void* alg, TsPutResults cb, void* args)
{
    // TS_INFO_MSG_V("algSetCb2 called");

    AlgCore* a = (AlgCore*)alg;
    assert(a);

    if (cb) {
        a->cb_put_results_ = cb;
        a->cb_user_data_ = args;
    }

    return true;
}