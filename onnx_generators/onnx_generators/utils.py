from string import Formatter

import torch
from onnxconverter_common.data_types import (DictionaryType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType,
                                             StringTensorType)
from torch import nn


class NeuralTorchModel(nn.Module):
    def __init__(self,main_block,softmax):
        super().__init__()
        self.main_block = main_block
        self.softmax = softmax

    def forward(self, x):
        return self.softmax(self.main_block(x))

class FastAIModel(nn.Module):
    def __init__(self,bn_cont,layers,softmax):
        super().__init__()
        self.bn_cont = bn_cont
        self.layers = layers
        self.softmax = softmax

    def forward(self, x):
        return self.softmax(self.layers(self.bn_cont(x)))

def convert_dataframe_schema(input_format, drop=None):
    inputs = []
    for k, v in input_format:
        if drop is not None and k in drop:
            continue
        if k == 'float':
            inputs.extend([(v[i], FloatTensorType([1, 1])) for i in range(len(v))])
        elif k == 'int':
            inputs.extend([(v[i], Int64TensorType([1, 1])) for i in range(len(v))])
        elif k == 'string':
            inputs.extend([(v[i], StringTensorType([1, 1])) for i in range(len(v))])
        else:
            raise NotImplementedError
    return inputs

def check_format(string, format_string):
    formatter = Formatter()
    parsed_format = formatter.parse(format_string)
    for _, field_name, _, _ in parsed_format:
        if field_name is not None and field_name not in string:
            return False
    return True

class Sentiment_score_torch_model(nn.Module):
    def forward(self, x, y):
        product = x * y
        positive_case = (product > 0).type(torch.float32)
        zero_case = (product == 0).type(torch.float32)
        negative_case = (product < 0).type(torch.float32)
        return positive_case - negative_case  # This way, positive_case -> 1, zero_case -> 0, negative_case -> -1

def get_sentiment_score_onnx():
    x = torch.randn(1, 1).to(torch.float32)
    y = torch.randn(1, 1).to(torch.float32)
    Sentiment_score_model = Sentiment_score_torch_model()
    Sentiment_score_model.eval()
    torch.onnx.export(Sentiment_score_model, (x, y), "model.onnx")
    sentimient_onnx = torch.onnx._export(Sentiment_score_torch_model, (x, y))
    return sentimient_onnx.graph.node, sentimient_onnx.graph.initializer

if __name__ == "__main__":
    get_sentiment_score_onnx()