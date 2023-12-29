#! /bin/bash

ONNX_DIR=../model
SOC_VERSION=`python3 -c 'import acl; print(acl.get_soc_name())'`
ONNX_NAME=vgg19.onnx
MODEL_NAME=${ONNX_NAME%%.*}
PRECISION_MODE=force_fp16

atc --model $ONNX_DIR/$ONNX_NAME --framework 5 --output $ONNX_DIR/${MODEL_NAME}_${PRECISION_MODE} --soc_version $SOC_VERSION --precision_mode $PRECISION_MODE
