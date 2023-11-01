import os

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from onnx.helper import make_tensor
from sclblonnx import concat

from .Abstract_onnx_generator import Abstract_ONNX_Generator
from .operators import (argmax_operator, div_operator, mean_operator,
                        subtract_operator)
from .utils import FastAIModel


class fastai_onnx_generator(Abstract_ONNX_Generator):
    def __init__(self,model_dir):
        super().__init__(model_dir)
        # self.cont_columns = [child.cont_columns for child in self.children]
        # self.cont_normalization = [(cont_mean,cont_std) for child._cont_normalization in self.children]
        #TODO:X[self.cont_columns] = (X[self.cont_columns] - cont_mean) / cont_std
        # self.initial_types = self.get_inputs_type(self.input_format)
        self.preprocess_onnx = [self.preprocess_onnx_graph(self.children[i]) for i in range(len(self.children))]
        self.fastai_torch_to_onnx()
        self.load_torch_onnx
        self.concat_onnx_graph
        self.concat_onnx_model = self.concat_graphs2models(self.concat_graphs)
        
    def preprocess_onnx_graph(self,model):

        cont_mean, cont_std = model._cont_normalization
        input_data = [onnx.helper.make_tensor_value_info(name="value_input", elem_type=onnx.TensorProto.FLOAT, shape=(1,len(model.cont_columns)))]
        output_data = [onnx.helper.make_tensor_value_info(name="div_output", elem_type=onnx.TensorProto.FLOAT, shape=(1,len(model.cont_columns)))]
        cont_mean = make_tensor(name="cont_mean", data_type=onnx.TensorProto.FLOAT, dims=[1,len(model.cont_columns)], vals=cont_mean.reshape([1,-1]).astype(np.float32).tobytes(), raw=True)
        cont_std = make_tensor(name="cont_std", data_type=onnx.TensorProto.FLOAT, dims=[1,len(model.cont_columns)], vals=cont_std.reshape([1,-1]).astype(np.float32).tobytes(), raw=True)
        initializer = [cont_mean,cont_std]
        sub = subtract_operator(('value_input',"cont_mean"), 'sub_out')
        div = div_operator(['sub_out','cont_std'], 'div_output')
        nodes = [sub,div]
        graph = onnx.helper.make_graph(nodes=nodes, name="preprocess", inputs=input_data,
                               outputs=output_data, initializer=initializer)
        return graph
    
    def fastai_torch_to_onnx(self):
        
        for i in range(len(self.children)):
            bn_cont = self.children[i].model.bn_cont
            layers = self.children[i].model.layers
            torch_model = FastAIModel(bn_cont=bn_cont,layers=layers,softmax=nn.Softmax(dim=1))
            torch.onnx.export(torch_model,
                            args= torch.rand(1,len(self.children[i].cont_columns),dtype=torch.float32),
                            f =os.path.join(self.children[i].path,'torch_model.onnx'),
                            opset_version=15,
                            export_params=True,
                            do_constant_folding=True)
    
    @property
    def concat_onnx_graph(self):
        self.concat_graphs = [concat(self.preprocess_onnx[i],self.torch_onnx[i].graph,io_match=[('div_output','input.1')],rename_nodes= False) for i in range(len(self.torch_onnx))]
    

    def concat_graphs2models(self,merge_graphs):
        
        merged_child_model = onnx.ModelProto(ir_version=self.torch_onnx[0].ir_version,
                        producer_name="ZhengLi",
                        producer_version=self.torch_onnx[0].producer_version,
                        opset_import=self.torch_onnx[0].opset_import)
        merged_child_models = [onnx.helper.make_model(merge_graphs[i]) for i in range (len(merge_graphs))]
        for i in range(len(merged_child_models)):
            with open (os.path.join(self.children[i].path,'merged_child_model.onnx'),'wb') as f:
                f.write(merged_child_models[i].SerializeToString())
        return merged_child_models

    def rename_node_name(self,onnx_models):
        """
        This function is used to rename the node name of onnx models.
        """
        for i in range(len(onnx_models)):
            graph = onnx_models[i].graph
            for initializer in graph.initializer:
                initializer.name = initializer.name + str(i)
            for z in range(len(graph.output)):
                graph.output[z].name = graph.output[z].name + str(i)
            nodes = graph.node
            for j in range(len(nodes)):
                nodes[j].name = nodes[j].name + str(i)
                for k in range(len(nodes[j].input)):
                    if nodes[j].input[k] != 'value_input':
                        nodes[j].input[k] = nodes[j].input[k] + str(i)
                for k in range(len(nodes[j].output)):
                    nodes[j].output[k] = nodes[j].output[k] + str(i)
        return onnx_models

    def making_nodes(self,num_of_children):
        operators = [mean_operator(output_name='result', inputs_name='30',num_of_children = num_of_children), argmax_operator(output_name='final_output', inputs_name='result')]
        return operators
    
    
    def merge_onnx_models(self):
        merged_model = onnx.ModelProto(ir_version=self.concat_onnx_model[0].ir_version,
                        producer_name="ZhengLi",
                        producer_version=self.concat_onnx_model[0].producer_version,
                        opset_import= self.concat_onnx_model[0].opset_import)
        self.rename_node_name(self.concat_onnx_model)
        merged_model = super().merge_graphs(merged_model = merged_model, onnx_models = self.concat_onnx_model)
        operators = self.making_nodes(5)
        merged_model.graph.node.extend(operators)
        output = onnx.helper.make_tensor_value_info("final_output", 7, [1])
        merged_model.graph.output.extend([output])
        # merged_model.opset_import[0].version = 17
        return merged_model
    

    def transform(self):
        self.merged_onnx_model = self.merge_onnx_models()
        self.save(self.merged_onnx_model)


        
if __name__ == "__main__":
    model_dir = r'E:\Bagging2onnx\Autogluon2onnx\autogluon_USRS_first_cls\models\NeuralNetFastAI_BAG_L1'
    model = fastai_onnx_generator(model_dir=model_dir)
    model.transform()
    df = pd.read_csv('USRS_train.csv')
    df = df.drop(columns=['date','label'])
    onnx_df = df.to_numpy().astype(np.float32)
    predictor = TabularPredictor.load(r'E:\Bagging2onnx\Autogluon2onnx\autogluon_USRS_first_cls',require_version_match=False)
    auto_labels = predictor.predict(df,model = 'NeuralNetFastAI_BAG_L1')
    onnx_session = ort.InferenceSession(r'E:\Bagging2onnx\Autogluon2onnx\autogluon_USRS_first_cls\models\NeuralNetFastAI_BAG_L1\onnx_model.onnx')
    onnx_labels = [] 
    for i in onnx_df:
        onnx_input = {onnx_session.get_inputs()[0].name: i.reshape(1, -1)}
        onnx_output = onnx_session.run(None, onnx_input)
        onnx_labels.append(onnx_output[0][0])
    auto_labels = np.where(auto_labels==-1,0,1).reshape([1,-1])
    onnx_labels = np.array(onnx_labels).reshape([1,-1])
    print(auto_labels-onnx_labels)
    