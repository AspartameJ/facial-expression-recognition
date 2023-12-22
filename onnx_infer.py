import sys
sys.path.append("./")
import onnxruntime as rt
import cv2
import numpy as np
from PIL import Image
from model import transforms


def softmax(x):
    if len(x.shape) > 1:
        x = np.exp(x) / np.sum(np.exp(x),axis=1).reshape(-1,1)
    else:
        x = np.exp(x) / np.sum(np.exp(x))
    return x

def test_onnx():
    transform_test = transforms.Compose([
                transforms.TenCrop(44),
                transforms.Lambda(lambda crops: np.stack([transforms.ToNdarray()(crop) for crop in crops]))
            ])
    img_paths = ["./imgs/1.jpg","./imgs/2.jpg"]

    sess = rt.InferenceSession("./model/vgg19.onnx", providers=['CPUExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # 打印输入节点的名字，以及输入节点的shape
    for i in range(len(sess.get_inputs())):
        print(sess.get_inputs()[i].name, sess.get_inputs()[i].shape)

    print("----------------")
    # 打印输出节点的名字，以及输出节点的shape
    for i in range(len(sess.get_outputs())):
        print(sess.get_outputs()[i].name, sess.get_outputs()[i].shape)
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

        img = Image.fromarray(np.uint8(img))
        inputs = transform_test(img)

        outputs = sess.run([out_name], {input_name:inputs})
        outputs_avg = outputs[0].mean(0)    # avg over crops

        score = softmax(outputs_avg)
        predicted = np.argmax(outputs_avg)
        print(predicted, score)

if __name__ == '__main__':
    test_onnx()

