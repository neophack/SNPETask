#!/bin/bash
###
 # Copyright (c) 2012-2022
 # All Rights Reserved by Thundercomm Technology Co., Ltd. and its affiliates.
 # You may not use, copy, distribute, modify, transmit in any form this file
 # except in compliance with THUNDERCOMM in writing by applicable law.
 #
 # @Description: Build script of AlgHelmetWearingMonitoring.
 # @version: 1.0
 # @Author: Ricardo Lu<sheng.lu@thundercomm.com>
 # @Date: 2022-07-01 10:12:47
 # @LastEditors: Ricardo Lu
 # @LastEditTime: 2022-07-05 02:41:04
### 
#set -x

ALGNAME=yolov5s
CURUSER=`whoami`
SUDO=""
[ "$CURUSER" == "root" ] || SUDO="sudo"

#--------------------------------------------------------------------------------------------------
# calculate version
#--------------------------------------------------------------------------------------------------
VERSION=1.1

#------------------------------------------------------------------------------
# prepare
#------------------------------------------------------------------------------
pushd ${0%/*}  > /dev/null 2>&1
rm -rf build   > /dev/null 2>&1
mkdir -p build > /dev/null 2>&1
pushd build    > /dev/null 2>&1

#------------------------------------------------------------------------------
# build & install
#------------------------------------------------------------------------------
echo -n -e "build and install ALG $ALGNAME \t... "
cmake .. > /dev/null 2>&1
if [ $? != 0 ]; then
  echo "[failed]"
  popd -n -2 > /dev/null 2>&1
  exit 1
fi

make -j4 > /dev/null 2>&1
if [ $? != 0 ]; then
  echo "[failed]"
  popd -n -2 > /dev/null 2>&1
  exit 1
fi

$SUDO make install > /dev/null 2>&1
if [ $? != 0 ]; then
  echo "[failed]"
  popd -n -2 > /dev/null 2>&1
  exit 1
fi

echo "[ok]"

#------------------------------------------------------------------------------
# build deb package
#------------------------------------------------------------------------------
echo -n -e "build ALG $ALGNAME deb package \t... "
checkinstall -D -y --install=no  --maintainer=THUNDERCOMM \
  --pkgversion=$VERSION --pkgrelease=rel --pkggroup=algorithm \
  --pkgname=Alg-${ALGNAME}-01-Linux > /dev/null 2>&1
if [ $? != 0 ]; then
  echo "[failed]"
  popd -n -2 > /dev/null 2>&1
  exit 1
fi

echo "[ok]"

#------------------------------------------------------------------------------
# end
#------------------------------------------------------------------------------
popd -n -2 > /dev/null 2>&1
exit 0

