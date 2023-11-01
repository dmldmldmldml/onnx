import os
from os.path import isdir, isfile, join

import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from onnxconverter_common.data_types import (DictionaryType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType)
from skl2onnx import convert_sklearn

from .Abstract_onnx_generator import Abstract_ONNX_Generator


class RandomForest_Bagging_Onnx(Abstract_ONNX_Generator):
    def __init__(self, model_dir) -> None:
        super().__init__(model_dir)
        self.initial_types = self.get_initial_types(self.input_format)

    def transform(self):
         if self.children:
            for i in range(len(self.children)):
                self.onnx_model = convert_sklearn(self.children[i].model, initial_types=self.initial_types, options={type(self.children[i]): {'zipmap':False}})
                export_path = join(os.path.dirname(self.children_path[i]),'onnx_model.onnx')
                del self.onnx_model.graph.output[1]
                self.save(self.onnx_model)

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
    model_dir = r'E:\Bagging2onnx\AutogluonOnnxGenerator_1.0\autogluon_NF_first_cls\models\RandomForestEntr_BAG_L1'
    model = RandomForest_Bagging_Onnx(model_dir=model_dir)
    df = pd.read_csv(r'E:\NF\NF_onnx_input.csv')
    df = df[['now_1', 'predict_1', 'previous_1', 'correction_1', 'now_2',
       'predict_2', 'previous_2', 'correction_2','now_predict_sub_1', 'now_previous_sub_1',
       'now_correction_sub_1', 'now_predict_sub_2', 'now_previous_sub_2',
       'now_correction_sub_2']]
    test_data = df.to_numpy().astype('float32')
    predictor = TabularPredictor.load(r'E:\NF\autogluon_NF_first_cls',require_version_match=False)
    auto_labels = predictor.predict(df,model = 'RandomForestGini_BAG_L1')
    onnx_labels = []
    for i in test_data:
        onnx_labels.append(model.test(i.reshape(1, -1))[0][0])
    auto_labels = np.where(auto_labels==-1,0,1).reshape([1,-1])
    onnx_labels = np.array(onnx_labels).reshape([1,-1])
    print(auto_labels-onnx_labels)

    # test = np.random.rand(1,15).astype(np.float32)
    # model.transform()
    # model.test(test)