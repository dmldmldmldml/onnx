import os

import numpy as np
import onnx
from onnx.helper import make_tensor

from .Abstract_onnx_generator import Abstract_ONNX_Generator
from .operators import add_operator, argmax_operator, mul_operator


class Weighted_ensemble_onnx_generator(Abstract_ONNX_Generator):
    '''
    weighted_ensemble模型转onnx

    Parameters
    ----------
    model_path : str
        Path to the bagged_model in autogulon directory. e.g. 'AutogluonModels/ag_model_2021-05-20_15-00-00/models/Weighted_Ensemble_L2'
    
    '''
    def __init__(self, model_dir):
        '''
        初始化    
        '''
        super().__init__(model_dir)
        self.sub_model_list = self.get_sub_models()
        self.weighted_onnx_model_graphs = self.getting_weighted_model_onnx()
        self.get_the_weigheted_model_weight
    
    def get_sub_models(self):
        '''
        获取子模型的名称
        '''
        stack_column_prefix = self.bagged_model.stack_column_prefix_lst
        return stack_column_prefix
    
    @property
    def get_the_weigheted_model_weight(self):
        '''
        得到子模型的权重
        TODO:目前仅遇到一个子模型的情况，后续需要考虑多个子模型的情况
        '''
        if len(self.children) ==1:
            self.weights = self.children[0].model.weights_
        else:
            raise NotImplementedError('The number of children is not 1')
    
    def getting_weighted_model_onnx(self):
        '''
        得到子模型的onnx模型
        TODO:目前得先生成子模型的onnx模型到对应的模型目录中，后续需要直接生成子模型的onnx文件并进行Weighted_ensemble的onnx生成
        '''
        self.sub_onnx_models = [] 
        for i in range(len(self.sub_model_list)):
            weighted_model_path = os.path.join(os.path.dirname(os.path.dirname(self.bagged_model_path)), self.sub_model_list[i], 'onnx_model.onnx')
            if os.path.isfile(weighted_model_path):
                weighted_model_onnx = onnx.load(weighted_model_path)
                self.sub_onnx_models.append(weighted_model_onnx)
            else:
                raise FileNotFoundError(f'The {self.sub_model_list[i]} onnx model file is not found')
    
    def rename_nodes(self):
        '''
        重命名子模型的算子名称
        '''
        index= 0
        for i in range(len(self.sub_onnx_models)):
            graph = self.sub_onnx_models[i].graph
            for node in graph.node:
                if node.op_type == 'Mean':
                    node.output[0] = node.output[0] + '_' + str(index)
                    index+=1
                elif 'input' in node.input:
                    for i in range(len(node.input)):
                        if node.input[i] == 'input':
                            node.input[i] = 'value_input'
                elif node.op_type == 'ArgMax':
                    graph.node.remove(node)
        
    def making_nodes(self):
        '''
        生成加权融合的算子
        '''
        muls_nodes = [] 
        for i in range(len(self.sub_onnx_models)):
            mul = mul_operator(('result_'+str(i),'weight'+str(i)),'mul_output'+str(i))
            muls_nodes.append(mul)
        add_node = add_operator(['mul_output'+str(i) for i in range(len(self.sub_onnx_models))],'add_output')
        argmax_node = argmax_operator(output_name= 'argmax_output',inputs_name='add_output')
        return [add_node,argmax_node]+muls_nodes
    
    def weighted_ensemble_onnx_generate(self):
        '''
        生成weighted_ensemble的onnx模型
        '''
        weighted_ensemble_onnx = onnx.ModelProto(ir_version=self.sub_onnx_models[0].ir_version,
                        producer_name="ZhengLi",
                        producer_version=self.sub_onnx_models[0].producer_version,
                        opset_import= self.sub_onnx_models[0].opset_import)
        weights =[]
        for i in range(len(self.sub_onnx_models)):
            weight = make_tensor(name="weight"+str(i), data_type=onnx.TensorProto.FLOAT, dims=[1,1], vals=self.weights[i].reshape([1,-1]).astype(np.float32).tobytes(), raw=True)
            weights.append(weight)
        self.rename_nodes()
        weighted_ensemble_onnx = super().merge_graphs(merged_model = weighted_ensemble_onnx, onnx_models = self.sub_onnx_models)
        weighted_ensemble_onnx.graph.initializer.extend(weights)
        added_nodes = self.making_nodes()
        weighted_ensemble_onnx.graph.node.extend(added_nodes)
        output = onnx.helper.make_tensor_value_info("argmax_output", 7, [1])
        weighted_ensemble_onnx.graph.output.extend([output])
        return weighted_ensemble_onnx

    def transform(self):
        '''
        生成onnx模型
        '''
        self.weighted_ensemble_onnx = self.weighted_ensemble_onnx_generate()
        self.save(self.weighted_ensemble_onnx)

        

if __name__ == "__main__":
    model_dir = r'E:\Bagging2onnx\Autogluon2onnx\autogluon_USRS_first_cls\models\WeightedEnsemble_L2'
    model = Weighted_ensemble_onnx(model_dir=model_dir)
    model.transform()