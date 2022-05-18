/*
 * @Description: Test program of yolov5s. 
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-05-18 16:51:10
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-05-18 10:10:25
 */

/*
 * @Description: Test porgram of TSYolov5s.
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-03-27 15:26:46
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-05-12 10:50:13
 */

#include <string>
#include <vector>
#include <sys/stat.h>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>

#include "TSYolov5s.h"
#include "TSYolov5sImpl.h"
#include "TSStruct.h"

static bool validateInput(const char* name, const std::string& value) 
{ 
    if (!value.compare ("")) {
        TS_ERROR_LOG("You must specify an input file!");
        return false;
    }

    struct stat statbuf;
    if (0 == stat(value.c_str(), &statbuf)) {
        return true;
    }

    TS_ERROR_LOG("Can't stat input image file: %s", value.c_str());
    return false;
}

DEFINE_string(input, "./image.jpg", "Input image file for this test program.");
DEFINE_validator(input, &validateInput);

static bool validateLabels(const char* name, const std::string& value) 
{ 
    if (!value.compare ("")) {
        TS_ERROR_LOG("You must specify a labels file!");
        return false;
    }

    struct stat statbuf;
    if (0 == stat(value.c_str(), &statbuf)) {
        return true;
    }

    TS_ERROR_LOG("Can't stat labels file: %s", value.c_str());
    return false;
}

DEFINE_string(labels, "./labels.txt", "Labels file for the yolov5s model.");
DEFINE_validator(labels, &validateLabels);

static bool validateModelPath(const char* name, const std::string& value) 
{ 
    if (0 == value.compare ("")) {
        TS_ERROR_LOG("You must specify a dlc file!");
        return false;
    }

    struct stat statbuf;
    if (0 == stat(value.c_str(), &statbuf)) {
        return true;
    }

    TS_ERROR_LOG("Can't stat model file: %s", value.c_str());
    return false;
}

DEFINE_string(model_path, "./yolov5s.dlc", "DLC file path.");
DEFINE_validator(model_path, &validateModelPath);

int main(int argc, char* argv[])
{
    google::ParseCommandLineFlags (&argc, &argv, true);

    std::vector<std::string> labels;
    std::ifstream in(FLAGS_labels);
    std::string line;
    while (getline(in,line)){
        labels.push_back(line);
    }

    std::vector<std::shared_ptr<ts::TSObjectDetection> > vec_alg;
    for (int i = 0; i < 6; i++) {
        std::shared_ptr<ts::TSObjectDetection> alg = std::shared_ptr<ts::TSObjectDetection>(new ts::TSObjectDetection());
        alg->Init(FLAGS_model_path, DSP);
        vec_alg.push_back(alg);
    }

    for (int i = 0; i < 6; i++) {
        cv::Mat img = cv::imread(FLAGS_input);
        cv::Mat rgb_img;

        cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

        ts::TSImgData ts_img(img.cols, img.rows, TYPE_RGB_U8, rgb_img.data);
        std::vector<ts::ObjectData> vec_res;

        vec_alg[i]->Detect(ts_img, vec_res);

        TS_INFO_LOG("result size: %ld", vec_res.size());

        for (size_t i = 0; i < vec_res.size(); i++) {
            ts::ObjectData rect = vec_res[i];
            TS_INFO_LOG("[%d, %d, %d, %d, %f, %d]", rect.x, rect.y, rect.width, rect.height, rect.confidence, rect.label);
            cv::rectangle(img, cv::Rect(rect.x, rect.y, rect.width, rect.height), cv::Scalar(0, 255, 0), 3);
            cv::Point position = cv::Point(rect.x, rect.y - 10);
            cv::putText(img, labels[rect.label], position, cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, 0.3);
        }

        std::string output_path = "./object_detection_result_" + std::to_string(i) + ".jpg";

        cv::imwrite(output_path, img);
    }
    google::ShutDownCommandLineFlags ();
    return 0;
}
