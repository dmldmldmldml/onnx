import os

import numpy as np
import onnx
import onnxmltools
import onnxruntime as ort
import pandas as pd
from autogluon.tabular import TabularPredictor
from onnxconverter_common.data_types import FloatTensorType

from .Abstract_onnx_generator import Abstract_ONNX_Generator
from .operators import argmax_operator, mean_operator


class XGboost_bagging_onnx_generator(Abstract_ONNX_Generator):
    
    def __init__(self, model_dir):
        super().__init__(model_dir)
        self.initial_types = self.get_initial_types(self.input_format)
        self.child_to_onnx()
        self.onnx_models = super().load_onnx_models(self.children_path)

    def get_initial_types(self,input_format):
        input_dim = 0 
        for i in range(len(input_format)):
            #TODO: xgboost only support float type
            # if input_format[i][0] == 'int':
            #     initial_types.append(('input'+str(i),Int64TensorType([-1,len(input_format[i][1])])))
            if input_format[i][0] == 'float' or input_format[i][0] == 'int':
                input_dim += len(input_format[i][1])
        initial_types = [('input', FloatTensorType([-1,input_dim]))]
        return initial_types

    def child_to_onnx(self):
        if self.children:
            for i in range(len(self.children)):
                onnx_model = onnxmltools.convert.convert_xgboost(self.children[i].model, initial_types = self.initial_types,target_opset=15)
                export_path = os.path.join(os.path.dirname(self.children_path[i]),'onnx_model.onnx')
                with open (export_path,'wb') as f:
                    f.write(onnx_model.SerializeToString())
    
    def rename_node_name(self):
            """
            This function is used to rename the node name of onnx models.
            """
            for i in range(len(self.onnx_models)):
                graph = self.onnx_models[i].graph
                for initializer in graph.initializer:
                    initializer.name = initializer.name + str(i)
                for z in range(len(graph.output)):
                    graph.output[z].name = graph.output[z].name + str(i)
                nodes = graph.node
                for j in range(len(nodes)):
                    nodes[j].name = nodes[j].name + str(i)
                    if j == 0:
                        for k in range(len(nodes[j].output)):
                            nodes[j].output[k] = nodes[j].output[k] + str(i)
                    else:
                        for k in range(len(nodes[j].input)):
                            nodes[j].input[k] = nodes[j].input[k] + str(i)
                        for k in range(len(nodes[j].output)):
                            nodes[j].output[k] = nodes[j].output[k] + str(i)
    
    def merge_onnx_models(self):
        merged_model = onnx.ModelProto(ir_version=self.onnx_models[0].ir_version,
                        producer_name="ZhengLi",
                        producer_version=self.onnx_models[0].producer_version,
                        opset_import= self.onnx_models[0].opset_import)
        self.rename_node_name()
        merged_model = super().merge_graphs(merged_model = merged_model, onnx_models = self.onnx_models)
        operators = self.making_nodes(len(self.children))
        merged_model.graph.node.extend(operators)
        output = onnx.helper.make_tensor_value_info("final_output", 7, [1])
        output1 = [onnx.helper.make_tensor_value_info(f"label{i}", 7, [1]) for i in range(len(self.children))] 
        merged_model.graph.output.extend([output])
        merged_model.graph.output.extend(output1)
        # merged_model.opset_import[0].version = 17
        return merged_model
    
    @staticmethod
    def making_nodes(num_of_children):
        operators = [mean_operator(output_name='result', inputs_name='probabilities',num_of_children = num_of_children), argmax_operator(output_name='final_output', inputs_name='result')]
        return operators

    def transform(self):
        
        self.merged_onnx_model = self.merge_onnx_models()
        self.save(self.merged_onnx_model)
    
    def test(self,test_data):
        """
        This function is used to test the onnx model.
        """
        sess = ort.InferenceSession(self.merged_onnx_model.SerializeToString())
        test = {}
        test = {'input':test_data.astype(np.float32)}
        res = sess.run(None,test)
        return res


if __name__ == "__main__":
    model_dir = r'E:\Bagging2onnx\Autogluon2onnx\autogluon_FGPMI_label_cls\models\XGBoost_BAG_L1'
    model = XGboost_bagging_onnx_generator(model_dir=model_dir)
    model.transform()
    # test_data = np.random.randn(1,28).astype('float32')
    df = pd.read_csv(r"E:\PMI\法德PMI_label.csv")
    label = np.array(df['label'].tolist()).reshape([1,-1])
    label = np.where(label==-1,0,1)
 
    df = df.drop(columns=['date','label','time'])

    test_data = df.to_numpy().astype('float32')
    predictor = TabularPredictor.load(r'E:\Bagging2onnx\Autogluon2onnx\autogluon_FGPMI_label_cls',require_version_match=False)
    autogluon_prediction = predictor.predict(df,model='XGBoost_BAG_L1')
    
    onnx_output = model.test(test_data)
    autogluon_prediction= np.where(autogluon_prediction==-1,0,1).reshape([1,-1])
    
    onnx_output = np.array(onnx_output).reshape([1,-1])
    print(onnx_output - label)
    print(onnx_output- autogluon_prediction)