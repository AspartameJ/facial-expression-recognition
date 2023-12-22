import torch
from ..model.vgg import VGG


if __name__ == "__main__":
     model = VGG("VGG19")
     model.load_state_dict(torch.load("../model/pytorch_model.pt"))
     model.eval()
     dummy_input = torch.randn(1, 3, 48, 48)
     torch.onnx.export(model, dummy_input, "../model/vgg19.onnx", opset_version=11, input_names=["input"], output_names=["output"])
