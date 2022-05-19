/*
 * Copyright (c) 2012-2022
 * All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 * You may not use, copy, distribute, modify, transmit in any form this file
 * except in compliance with THUNDERCOMM in writing by applicable law.
 * 
 * @Description: Initilize version of encapsulation of object-deteciton algorithm developed on SNPE.
 * @version: 1.0
 * @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 * @Date: 2022-05-19 11:08:17
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-05-19 11:13:16
 */

#ifndef __TS_OBJECT_FACE_DETECTION_H__
#define __TS_OBJECT_FACE_DETECTION_H__  

#include "Common.h"

extern "C" void* algInit  (const std::string& args                );//**V1**//
extern "C" void* algInit2 (void* userdata, const std::string& args);//**V2**//
extern "C" bool  algStart (void* alg                              );//**V1**//
extern "C" std::shared_ptr<TsJsonObject> algProc (void* alg, 
     const std::shared_ptr<TsGstSample>& data                     );//**V1**//
extern "C" std::shared_ptr<std::vector<std::shared_ptr<TsJsonObject>>> 
                 algProc2 (void* alg, const std::shared_ptr<std::vector<
                           std::shared_ptr<TsGstSample>>>& datas  );//**V2**//
extern "C" bool  algCtrl  (void* alg, const std::string& cmd      );//**V1**//
extern "C" void  algStop  (void* alg                              );//**V1**//
extern "C" void  algFina  (void* alg                              );//**V1**//
extern "C" bool  algSetCb (void* alg, TsPutResult cb,  void* args );//**V1**//
extern "C" bool  algSetCb2(void* alg, TsPutResults cb, void* args );//**V2**//

#endif //__TS_OBJECT_FACE_DETECTION_H__