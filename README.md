# facial-expression-recognition
 facial-expression-recognition

### 参考链接
[cv_vgg19_facial-expression-recognition_fer](https://modelscope.cn/models/damo/cv_vgg19_facial-expression-recognition_fer/summary)

### pt转onnx
```
cd tools
python3 exporter.py
```
### onnx转om
```
bash onnx2om.sh
```
### onnx推理
```
cd ../
python3 onnx_infer.py
```
### onm推理
```
cd ../
python3 onm_infer.py
```