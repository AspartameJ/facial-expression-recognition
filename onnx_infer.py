import torch
import onnxruntime as rt
import torch.onnx
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from .model import transforms


def test_onnx():
    transform_test = transforms.Compose([
                transforms.TenCrop(44),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
            ])
    img_paths = ["./imgs/facial_expression_recognition.jpg"]

    sess = rt.InferenceSession("./model/vgg19.onnx", providers=['CPUExecutionProvider'])
    #  sess = rt.InferenceSession(onnx_sim, providers=['CUDAExecutionProvider'])  # providers=[CPUExecutionProvider,'CUDAExecutionProvider']
    #  providers=[CPUExecutionProvider,'CUDAExecutionProvider']

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
        img = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

        img = Image.fromarray(np.uint8(img))
        inputs = transform_test(img)

        ncrops, c, h, w = inputs.shape

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.to('cpu')
        inputs = Variable(inputs, volatile=True)
        outputs = sess(inputs)
        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 0)
        return predicted, score

if __name__ == '__main__':
    print(test_onnx())
