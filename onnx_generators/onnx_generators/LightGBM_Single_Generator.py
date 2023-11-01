import os
import pickle

import numpy as np
import onnx
import onnxmltools
from onnxconverter_common.data_types import (DictionaryType, FloatTensorType,
                                             Int64TensorType)


class LightGBM_single_onnx_generator():
    """
    This class is used to convert a single lightgbm model to onnx model.
    """
    def __init__(self, model_dir) -> None:
        self.model_dir = model_dir
        with open(os.path.join(self.model_dir,'model.pkl'), 'rb') as f:
                self.model = pickle.load(f)
    
    def get_initial_types(self):
        input_dim = 0 
        input_format = self.model.feature_metadata.get_type_group_map_raw().items()
        for input in input_format:
            if input[0] == 'int' or input[0] == 'float':
                input_dim += len(input[1])
        self.initial_types = [('input', FloatTensorType([-1,input_dim]))]
    def get_onnx_model(self):
        self.get_initial_types()
        self.onnx_model = onnxmltools.convert_lightgbm(self.model.model, initial_types = self.initial_types,zipmap=False,target_opset=15)
        del self.onnx_model.graph.output[1]
        self.save_onnx_model(os.path.join(self.model_dir,'model.onnx'))
    
    def save_onnx_model(self, save_path):
        with open (save_path, "wb") as f:
            f.write(self.onnx_model.SerializeToString())

if __name__ == '__main__':
    model_dir = r'E:\Bagging2onnx\AutogluonOnnxGenerator_1.0\onnx_generators\LightGBMLarge'
    generator = LightGBM_single_onnx_generator(model_dir)
    generator.get_onnx_model()    