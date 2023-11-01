import os

import numpy as np
import onnx
import torch
from autogluon.tabular.models.tabular_nn.compilers.onnx import \
    TabularNeuralNetTorchOnnxCompiler
from sclblonnx import concat
from skl2onnx import convert_sklearn
from sklearn.compose import ColumnTransformer

from .Abstract_onnx_generator import Abstract_ONNX_Generator
from .quantile_transformer import quantile_transformer_onnx_generator
from .utils import NeuralTorchModel, convert_dataframe_schema


class NeuralTorch_BAG_onnx_generator(Abstract_ONNX_Generator):

    def __init__(self,model_dir):
        super().__init__(model_dir)
        self.torch_models = [NeuralTorchModel(child.model.main_block,child.model.softmax) for child in self.children]
        self.processors = [child.processor for child in self.children]
        self.modify_processor
        self.initial_types = self.get_inputs_type(self.input_format)
        self.feature_list = [feature[0] for feature in self.initial_types]

    def get_inputs_type(self,input_format):
        return convert_dataframe_schema(input_format)
    
    @property
    def modify_processor(self):
        
        self.quantile_transformers = []
        for i in range(len(self.processors)):
            self.quantile_transformers.append(self.processors[i].transformers_[1][1].steps.pop()[1])
            transformers = [
                        self.processors[i].transformers_[0],
                        self.processors[i].transformers_[1]
                        ]
            new_processor = ColumnTransformer(transformers=transformers,remainder='passthrough')
            new_processor.transformers_ = transformers
            self.processors[i] = new_processor
    
    @property
    def torch_to_onnx(self):
        for i in range(len(self.children)):
            torch.onnx.export(self.torch_models[i],
                            args= torch.zeros(size=(1, len(self.initial_types)),dtype=torch.float32),
                            f=os.path.join(self.children[i].path,'torch_model.onnx'),
                            do_constant_folding=True)
        self.load_torch_onnx
    #TODO: the precesion of the onnx is quite different from the sklearn model becasue of the QuantiletTransformer operator.

    @property
    def processor_to_onnx(self):

        self.processors_onnx = [convert_sklearn(self.processors[i],initial_types=self.initial_types) for i in range(len(self.processors))]
    
    @property
    def quantile_transformer_to_onnx(self):
        '''
        将quantile_transformer转换为onnx
        '''
        self.quantile_transformers_onnx = [quantile_transformer_onnx_generator(self.quantile_transformers[i]).merge_graph for i in range(len(self.quantile_transformers))]
        
        for i in range(len(self.quantile_transformers_onnx)):
            with open (os.path.join(os.path.dirname(self.children[i].path),'quantile_Transformer.onnx'),'wb') as f:
                quantile_model = onnx.helper.make_model(self.quantile_transformers_onnx[i], producer_name='ZhengLi')
                f.write(quantile_model.SerializeToString())

    @property
    def merge_quantile_into_graph(self):
        
        self.processor_to_onnx
        self.quantile_transformer_to_onnx

        reshape_dim = onnx.helper.make_tensor(name ='reshape_dim', 
                                    data_type=onnx.TensorProto.INT64, 
                                    dims=(2,), 
                                    vals=np.array([1,1]).astype(np.int64).tobytes(), 
                                    raw=True)

        
        
        reshape_op_list = [onnx.helper.make_node('Reshape',
                                                  name = f'reshape_input_{i}',
                                                  inputs=[f'input{i}','reshape_dim'],
                                                  outputs=[f'input_{i}']
                                                  ) for i in range(self.quantile_transformers[0].quantiles_.shape[1])]
        
        for i in range(len(self.processors_onnx)):
            for j in range(len(self.processors_onnx[i].graph.node)):
                if self.processors_onnx[i].graph.node[j].name == 'Concat2':
                    self.processors_onnx[i].graph.node[j].input[1] = 'concat_result'
            split_op = onnx.helper.make_node('Split',
                                        inputs=['variable2'],
                                        outputs=[f'input{i}' for i in range(self.quantile_transformers[i].quantiles_.shape[1])],
                                        axis=1)
            
            reshape_op_list = [onnx.helper.make_node('Reshape',
                                                  name = f'reshape_input_{i}',
                                                  inputs=[f'input{i}','reshape_dim'],
                                                  outputs=[f'input_{i}']) for i in range(self.quantile_transformers[i].quantiles_.shape[1])]
            
            self.processors_onnx[i].graph.initializer.extend([reshape_dim])
            self.processors_onnx[i].graph.initializer.extend(self.quantile_transformers_onnx[i].initializer)
            self.processors_onnx[i].graph.node.extend(self.quantile_transformers_onnx[i].node)
            self.processors_onnx[i].graph.node.extend(reshape_op_list)
            self.processors_onnx[i].graph.node.extend([split_op])
            
        for i in range(len(self.processors_onnx)):
            with open (os.path.join(os.path.dirname(self.children[i].path),'processor.onnx'),'wb') as f:
                f.write(self.processors_onnx[i].SerializeToString())
    
    
    @property
    def concat_processor_torch_model(self):
        """
        得到column_Transformer+torch_model的onnx,子模型的onnx文件
        """
        self.torch_to_onnx
        self.merge_quantile_into_graph
        self.concat_graphs = [concat(self.processors_onnx[i].graph,self.torch_onnx[i].graph,io_match=[('transformed_column','onnx::Gemm_0')]) for i in range(len(self.processors_onnx))]        
        for i in range(len(self.concat_graphs)):
            with open (os.path.join(os.path.dirname(self.children[i].path),'concat_model.onnx'),'wb') as f:
                concat_graph = onnx.helper.make_model(self.concat_graphs[i], producer_name='ZhengLi')
                f.write(concat_graph.SerializeToString())
    
    
    @property
    def rename_child_onnx_graph(self):
        """
        重命名子模型的onnx图的节点名称
        """

        for i in range(len(self.concat_graphs)):
            graph = self.concat_graphs[i]
            for initializer in graph.initializer:
                if initializer.name not in  ['reference','reshape_dim','half','min_ref','max_ref','min_len','max_len','ex_num','Zero']:
                    initializer.name = initializer.name + f'_child{i}'
            for output in graph.output:
                output.name = output.name+ f'_child{i}'
            nodes = graph.node
            for j in range(len(nodes)):
                nodes[j].name = nodes[j].name + f'_child{i}'
                for k in range(len(nodes[j].input)):
                    feature_list = self.feature_list+['reference','reshape_dim','half','min_ref','max_ref','min_len','max_len','ex_num','Zero']
                    if nodes[j].input[k] not in feature_list:
                        nodes[j].input[k] = nodes[j].input[k] + f'_child{i}'
                for k in range(len(nodes[j].output)):
                    nodes[j].output[k] = nodes[j].output[k] + f'_child{i}'
    
    @property
    def merge_final_graphs(self):
        """
        合并子模型的onnx图
        """
        self.concat_processor_torch_model
        self.rename_child_onnx_graph
        mean_op = onnx.helper.make_node('Mean',
                                        inputs=[self.concat_graphs[i].output[0].name for i in range(len(self.concat_graphs))],
                                        outputs=['mean_result'])
        argmax_operator = onnx.helper.make_node('ArgMax',
                                                inputs=['mean_result'],
                                                outputs=['final_result'], axis=1, keepdims=0)
        for i in range(len(self.concat_graphs)):
            graph = self.concat_graphs[i]
            if i == 0:
                merged_graph = graph
            else:
                merged_graph.initializer.extend(graph.initializer)
                merged_graph.node.extend(graph.node)
        merged_graph.node.extend([mean_op,argmax_operator])
        del merged_graph.output[0]
        output = onnx.helper.make_tensor_value_info("final_result", 7, [1])
        merged_graph.output.extend([output])
        merged_model = onnx.helper.make_model(merged_graph, producer_name='ZhengLi')
        return merged_model
    
    def transform(self):
        self.merged_onnx_model = self.merge_final_graphs
        self.save(self.merged_onnx_model)

if __name__ == '__main__':
    data = np.random.rand(1,8).astype(np.float32)
    model_dir = r'E:\Bagging2onnx\AutogluonOnnxGenerator_1.0\autogluon_USCPI_reg\models\NeuralNetTorch_BAG_L1'
    model = NeuralTorch_BAG_onnx_generator(model_dir)
    model.merge_final_graphs
    