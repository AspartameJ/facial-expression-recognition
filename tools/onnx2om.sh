#! /bin/bash

ONNX_PATH=../model/vgg19.onnx
SOC_VERSION=`python3 -c 'import acl; print(acl.get_soc_name())'`
ONNX_NAME=`basename $ONNX_PATH`
MODEL_NAME=${ONNX_NAME%%.*}
PRECISION_MODE=force_fp16

atc --model $ONNX_PATH --framework 5 --output ${MODEL_NAME}_${PRECISION_MODE} --soc_version $SOC_VERSION --precision_mode $PRECISION_MODE