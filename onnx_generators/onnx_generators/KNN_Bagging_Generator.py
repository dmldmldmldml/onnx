import os
from os.path import isdir, isfile, join

import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
from onnxconverter_common.data_types import (DictionaryType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType)
from skl2onnx import convert_sklearn

from .Abstract_onnx_generator import Abstract_ONNX_Generator
from .operators import (argmax_operator, mean_operator, softmax_operator,
                       subtract_operator)


class KNN_bagging_onnx_generator(Abstract_ONNX_Generator):

    def __init__(self, model_dir) -> None:
        super().__init__(model_dir)
        self.initial_types = self.get_initial_types(self.input_format)
        self.child_to_onnx()
        self.onnx_models = super().load_onnx_models(self.children_path)
    def child_to_onnx(self):
         if self.children:
            for i in range(len(self.children)):
                self.onnx_model = convert_sklearn(self.children[i].model, initial_types=self.initial_types)
                export_path = join(os.path.dirname(self.children_path[i]),'onnx_model.onnx')
                with open (export_path,'wb') as f:
                    f.write(self.onnx_model.SerializeToString())

    def get_initial_types(self,input_format):
        input_dim = 0 
        for i in range(len(input_format)):
            input_dim += len(input_format[i][1])
        initial_types = [('input', FloatTensorType([1,input_dim]))]
        return initial_types
    
    def test(self, test_data):
        for i in range(len(self.children)):
            sess = ort.InferenceSession(self.onnx_model.SerializeToString())
            input_name = sess.get_inputs()[0].name
            # output_name = sess.get_outputs()[0].name
            pred_onx = sess.run(None, {input_name: test_data.astype(np.float32)})[0]
            print(pred_onx)
            return pred_onx
    
if __name__ == "__main__":
    model_dir = r'E:\Bagging2onnx\AutogluonOnnxGenerator_1.0\autogluon_BOC_first_cls\models\KNeighborsDist_BAG_L1'
    model = KNN_bagging_onnx_generator(model_dir=model_dir)