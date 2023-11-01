import os
import pickle
import re
from os.path import join

import numpy as np
import onnx
from autogluon.tabular.models import *


class Abstract_ONNX_Generator():

    """
    This class is used to load the bagged model and convert it to onnx format.
    Warning: The outputs of LightGBM onnx model are not always as same as the original model.

    Parameters
    ----------
    model_path : str
        Path to the bagged_model in autogulon directory. e.g. 'AutogluonModels/ag_model_2021-05-20_15-00-00/models/LightGBMLarge'
    """
    def __init__(self,model_dir) -> None:

        self.bagged_model_path = join(model_dir,'model.pkl')
        self.bagged_model = self.load_model(self.bagged_model_path)
        self.problem_type = self.bagged_model.problem_type
        self.children = [self.bagged_model.load_child(child) for child in self.bagged_model.models]
        self.children_path = [child.path for child in self.children]
        self.input_format = list(self.bagged_model.feature_metadata.get_type_group_map_raw().items())
    
    
    @property
    def load_torch_onnx(self):
        self.torch_onnx = [onnx.load(os.path.join(child.path,'torch_model.onnx')) for child in self.children]


    def load_model(self,model_path):
        """
        This function is used to load models.
        """
        if isinstance(model_path, str):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif isinstance(model_path, list):
            model_list = []
            for dir in list:
                with open(dir, 'rb') as f:
                    model_list.append(pickle.load(f)) 
            return model_list
    

    def load_onnx_models(self,children_path):
        """
        This function is used to load the onnx models.
        """
        if children_path:
            if isinstance(children_path[0],str):
                onnx_models = [onnx.load(join(os.path.dirname(children_path[i]),'onnx_model.onnx')) for i in range(len(children_path))]
        return onnx_models
    
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
                if j == 0:
                    for k in range(len(nodes[j].output)):
                        nodes[j].output[k] = nodes[j].output[k] + str(i)
                else:
                    for k in range(len(nodes[j].input)):
                        nodes[j].input[k] = nodes[j].input[k] + str(i)
                    for k in range(len(nodes[j].output)):
                        nodes[j].output[k] = nodes[j].output[k] + str(i)
        return onnx_models
   

    def merge_graphs(self,merged_model,onnx_models):
        """
        This function is used to merge the onnx models.
        """
        if isinstance(onnx_models,list):
            merged_model.graph.input.extend([input for input in onnx_models[0].graph.input])
            for onnx_model in onnx_models:
                merged_model.graph.initializer.extend([initializer for initializer in onnx_model.graph.initializer])
                merged_model.graph.node.extend([node for node in onnx_model.graph.node])
            return merged_model
        
        else:
            raise NotImplementedError
        


    def save(self,onnx_model):
        onnx_model = self.onnx_input_transformation(onnx_model)
        with open (join(os.path.dirname(self.bagged_model_path),'onnx_model.onnx'),'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        model_type = self.bagged_model_path.split('\\')[0]
        with open (join(os.path.dirname(os.path.dirname(os.path.dirname(self.bagged_model_path))),f'{model_type}_onnx_model.onnx'),'wb') as f:
            f.write(onnx_model.SerializeToString())
        with open(join('onnx_repository',f'{model_type}_onnx_model.onnx'),'wb') as f:
            f.write(onnx_model.SerializeToString())
        
    
    def onnx_input_transformation(self,onnx_model):
        """
        This function is used to transform the input format of onnx model.
        """
        del onnx_model.graph.input[:]
        new_input_list = []
        all_feat_list = []
        new_initializer_list = []
        for data_type, feature_list in self.input_format:
            all_feat_list.extend(feature_list)
            for feature in feature_list:
                if bool(re.match(r'^\s*.+_sub_.+_is_up$', feature)) == True:
                    feat_split = feature[:-6].split('_sub_')
                    prior = feat_split[0]
                    sub = feat_split[1]
                    zero_tensor = onnx.helper.make_tensor(name ='zero_tensor', 
                                    data_type=onnx.TensorProto.INT64, 
                                    dims=(1,), 
                                    vals=np.array([0]).astype(np.int64).tobytes(), 
                                    raw=True)
                    
                    sub_operator = onnx.helper.make_node('Sub',
                                                        name = f'{prior}_sub_{sub}',
                                                        inputs = [prior,sub],
                                                        outputs = [f'{prior}_sub_{sub}'])
                    
                    greater_operator = onnx.helper.make_node('Greater',
                                                        name = f'{prior}_sub_{sub}_bool',
                                                        inputs = [f'{prior}_sub_{sub}','zero_tensor'],
                                                        outputs = f'{prior}_sub_{sub}_bool')
                    
                    cast_operator = onnx.helper.make_node('Cast',
                                                        name = f'{prior}_sub_{sub}_bool_cast',
                                                        inputs = [f'{prior}_sub_{sub}_bool'],
                                                        outputs = [feature],
                                                        to = onnx.TensorProto.INT64)
                    for node in [sub_operator,greater_operator,cast_operator]:
                        if node not in onnx_model.graph.node:
                            onnx_model.graph.node.append(node)
                    new_input_list.extend([prior,sub])
                    new_initializer_list.append(zero_tensor)

                
                elif bool(re.match(r'^\s*.+_abs_sub_.+_abs$', feature)) == True:
                    feat_split = feature[:-4].split('_abs_sub_')
                    prior = feat_split[0]
                    sub = feat_split[1]
                    abs_operator = [onnx.helper.make_node('Abs',
                                                        name = f'abs_{value}',
                                                        inputs = [f'{value}'],
                                                        outputs = [f'abs_{value}']) for value in [prior,sub]]
                    sub_operator = onnx.helper.make_node('Sub',
                                                        name = f'{prior}_sub_{sub}',
                                                        inputs = [f'abs_{prior}',f'abs_{sub}'],
                                                        outputs = [feature])
                    for node in abs_operator+sub_operator:
                        if node not in onnx_model.graph.node:
                            onnx_model.graph.node.append(node)
                    new_input_list.extend([prior,sub])

                elif bool(re.match(r'abs_*.+_sub_.+',feature)) == True:
                    feat_split = feature[4:].split('_sub_')
                    prior = feat_split[0]
                    sub = feat_split[1]
                    sub_operator = onnx.helper.make_node('Sub',
                                                        name = f'{prior}_sub_{sub}',
                                                        inputs = [prior,sub],
                                                        outputs = [f'{prior}_sub_{sub}'])
                    abs_operator = onnx.helper.make_node('Abs',
                                                        name = f'abs_{prior}_sub_{sub}',
                                                        inputs = [f'{prior}_sub_{sub}'],
                                                        outputs = [feature])
                    for node in [sub_operator,abs_operator]:
                        if node not in onnx_model.graph.node:
                            onnx_model.graph.node.append(node)
                    new_input_list.extend([prior,sub])
                
                elif feature.endswith('_relation'):
                    feat_split = feature[:-9].split('_')
                    keyword = feat_split[0]
                    feature_ = '_'.join(feat_split[1:])
                    Mul_operator = onnx.helper.make_node('Mul',
                                                        name = f'{keyword}_{feature_}_mul',
                                                        inputs = [keyword+'_sentiment_score',feature_],
                                                        outputs = [f'{keyword}_{feature_}_mul'])
                    
                    zero_const_node = onnx.helper.make_node(
                                        'Constant',
                                        inputs=[],
                                        outputs=['zero_const'],
                                        value=onnx.helper.make_tensor(
                                            name="const_tensor_zero",
                                            data_type=onnx.TensorProto.FLOAT,
                                            dims=[1, 1],
                                            vals=[0]
                                        ),
                                    )
                    Greater_node = onnx.helper.make_node('Greater',
                                                         name = f'{keyword}_{feature_}_greater',
                                                        inputs = [f'{keyword}_{feature_}_mul','zero_const'],
                                                        outputs = [f'{keyword}_{feature_}_greater'])
                    Less_node = onnx.helper.make_node('Less',
                                                    name = f'{keyword}_{feature_}_less',
                                                    inputs = [f'{keyword}_{feature_}_mul','zero_const'],
                                                    outputs = [f'{keyword}_{feature_}_less'])
                    
                    cast_Greater = onnx.helper.make_node(
                                'Cast',
                                name = f'{keyword}_{feature_}_Cast_Greater',
                                inputs=[f'{keyword}_{feature_}_greater'],
                                outputs=[f'{keyword}_{feature_}_Cast_Greater'],
                                to= getattr(onnx.TensorProto,"FLOAT")  # Cast to FLOAT
                                )
                    cast_Less = onnx.helper.make_node(
                                'Cast',
                                name = f'{keyword}_{feature_}_Cast_Less',
                                inputs=[f'{keyword}_{feature_}_less'],
                                outputs=[f'{keyword}_{feature_}_Cast_Less'],
                                to=getattr(onnx.TensorProto,"FLOAT")  # Cast to FLOAT
                                )
                    
                    sub_node  = onnx.helper.make_node('Sub',
                                                    name = f'{keyword}_{feature_}_sub',
                                                    inputs = [f'{keyword}_{feature_}_Cast_Greater',f'{keyword}_{feature_}_Cast_Less'],
                                                    outputs = [feature])
                    for node in [zero_const_node,Mul_operator,Greater_node,Less_node,cast_Greater,cast_Less,sub_node]:
                        if node not in onnx_model.graph.node:
                            onnx_model.graph.node.append(node)
                    if '_sub_' in feature_:
                        if feature_ not in all_feat_list:
                            feature_list.append(feature_)
                        feat_split = feature_.split('_sub_')
                        prior = feat_split[0]
                        sub = feat_split[1]
                        new_input_list.extend([keyword+'_sentiment_score',prior,sub])
                    else:
                        new_input_list.extend([keyword+'_sentiment_score',feature_])
                
                elif bool(re.match(r'(\w+)_sub_(\w+)', feature)) == True:
                    feat_split = feature.split('_sub_')
                    prior = feat_split[0]
                    sub = feat_split[1]
                    sub_operator = onnx.helper.make_node('Sub',
                                                        name = f'{prior}_sub_{sub}',
                                                        inputs = [prior,sub],
                                                        outputs = [feature])
                    new_input_list.extend([prior,sub])
                    if sub_operator not in onnx_model.graph.node:
                        onnx_model.graph.node.append(sub_operator)
                
                
                else:
                    if data_type == 'int':
                        new_input_list.append(feature)
                    elif data_type == 'float':
                        new_input_list.append(feature)
                    else:
                        raise NotImplementedError
        new_input_list = list(set(new_input_list))
        new_initializer_list = list(set(new_initializer_list))
        onnx_model.graph.input.extend([onnx.helper.make_tensor_value_info(name= input_node,
                                                                          elem_type= onnx.TensorProto.FLOAT, 
                                                                          shape = (1,1)) for input_node in new_input_list])
        onnx_model.graph.initializer.extend(new_initializer_list)
        if self.bagged_model._child_type != TabularNeuralNetTorchModel: 
            if self.bagged_model._child_type == NNFastAiTabularModel:
                input_name = 'value_input'
            elif self.bagged_model._child_type == CatBoostModel:
                input_name = "features"
            else:
                input_name = 'input'
            concat_operator = onnx.helper.make_node('Concat',
                                                    name = 'concat_input',
                                                    inputs = [feature for feature in all_feat_list],
                                                    outputs = [input_name],
                                                    axis = 1)
            onnx_model.graph.node.append(concat_operator)


        return onnx_model





# if __name__ == "__main__":
#     # args = get_args()
#     # bagged_model = Abstract_ONNX_Generator(args.model_dir)
#     # bagged_model.transform()
#     bagged_model = Abstract_ONNX_Generator(args.model_dir)