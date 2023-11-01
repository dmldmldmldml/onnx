import pickle

import numpy as np
import onnx
import pandas as pd
import sclblonnx as so
from mlinsights.mltree import digitize2tree
from skl2onnx import to_onnx

from .quantile_transformer_operators import np_interp_graph


class quantile_transformer_onnx_generator():
    '''
    这是用来将sklearn的quantile_transformer转换为onnx的类以契合autogluon中torch_nn的columntransformer预处理器
    '''

    def __init__(self,quantile_transformer):

        self.quantile_transformer = quantile_transformer
        self.n_quantiles = self.quantile_transformer.n_quantiles
        self.references = self.quantile_transformer.references_
        self.quantiles = self.quantile_transformer.quantiles_
    
    def get_tree_node(self):
        '''
        这是用来获取mlinsights中的digitize2tree的tree_node的方法
        '''
        pos_tree_node = []
        neg_tree_node = [] 
        x = np.random.random(1).astype(np.float32).reshape(1,1)
        for feature_idx in range(self.quantiles.shape[1]):
            pos_tree = digitize2tree(self.quantiles[:,feature_idx],right=True)
            neg_tree = digitize2tree(-(self.quantiles[::-1,feature_idx]),right=True)
            pos_onx = to_onnx(pos_tree, x.reshape((-1, 1)),
                          target_opset=15)
            neg_onx = to_onnx(neg_tree, x.reshape((-1, 1)),
                            target_opset=15)
            pos_node = pos_onx.graph.node[0]
            neg_node = neg_onx.graph.node[0]
            pos_tree_node.append(pos_node)
            neg_tree_node.append(neg_node)
        
        return pos_tree_node,neg_tree_node
    
    @property
    def making_graph(self):
        '''
        这是用来获取onnx的方法
        '''
        pos_tree_node,neg_tree_node = self.get_tree_node()
        self.np_graphs = []
        self.neg_np_graphs = [] 
        for i in range(len(pos_tree_node)):
            np_graph = np_interp_graph(quantile= self.quantiles[:,i],tree_node=pos_tree_node[i], references=self.references)            
            neg_np_graph = np_interp_graph(quantile= self.quantiles[:,i],tree_node=neg_tree_node[i], references=self.references,mode = 'negative')
            # np_model = onnx.helper.make_model(np_graph, producer_name='Zheng')
            # with open('quantile.onnx','wb') as f:
            #     f.write(np_model.SerializeToString())
            # break
            self.np_graphs.append(np_graph)
            self.neg_np_graphs.append(neg_np_graph)
    

    def rename_graph(self,graph_type):
        '''
        重命名graph的输入输出以及中间算子的名字
        '''
        if graph_type == 'positive':
            suffix = 'pos'
            graphs = self.np_graphs
        elif graph_type == 'negative':
            suffix = 'neg'
            graphs= self.neg_np_graphs

        for i in range(len(graphs)):
            graph = graphs[i]
            for initializer in graph.initializer:
                if initializer.name not in ['reference','min_ref','max_ref','min_len','max_len','ex_num','Zero']:
                    initializer.name = initializer.name + "_"+ suffix +"_"+str(i)
            for input in graph.input:
                input.name = input.name + "_" +str(i)
            for output in graph.output:
                output.name = output.name+ "_"+ suffix +"_"+str(i)
            nodes = graph.node
            for j in range(len(nodes)):
                nodes[j].name = nodes[j].name + "_"+ suffix +"_"+str(i)

                for k in range(len(nodes[j].input)):
                    if nodes[j].input[k] == 'input':
                        nodes[j].input[k] = nodes[j].input[k] + "_" +str(i)
                    elif nodes[j].input[k] not in ['reference','min_ref','max_ref','min_len','max_len','ex_num','Zero']:
                        nodes[j].input[k] = nodes[j].input[k] + "_"+ suffix +"_"+str(i)
                
                for k in range(len(nodes[j].output)):
                    nodes[j].output[k] = nodes[j].output[k] + "_"+ suffix +"_"+ str(i)
    
    @property
    def merge_graph(self):
        '''
        将不同quantiles的graph合并为一个graph
        '''
        self.making_graph
        self.rename_graph(graph_type='positive')
        self.rename_graph(graph_type='negative')
        half_initializer = onnx.helper.make_tensor(name ='half', 
                                    data_type=onnx.TensorProto.FLOAT, 
                                    dims=(1,1),
                                    vals=np.array([[2]]).astype(np.float32).tobytes(), 
                                    raw=True)

        pos_concat = onnx.helper.make_node('Concat',name = 'Concat_pos',inputs=[np_graph.output[0].name for np_graph in self.np_graphs],outputs=['pos_concat_result'],axis=1)
        neg_concat = onnx.helper.make_node('Concat',name = 'Concat_neg',inputs=[np_graph.output[0].name for np_graph in self.neg_np_graphs],outputs=['neg_concat_result'],axis=1)
        Sub = onnx.helper.make_node('Sub',name = 'pos_neg_Sub',inputs=['pos_concat_result','neg_concat_result'],outputs=['pos_neg_Sub'])
        div = onnx.helper.make_node('Div',name = 'div',inputs=['pos_neg_Sub','half'],outputs=['div_result'])
        Clip = onnx.helper.make_node('Clip',name = 'div_result_Clip',inputs=['div_result', "min_ref", "max_ref"],outputs=['concat_result'])
        for i in range(len(self.np_graphs)):
            graph = self.np_graphs[i]
            neg_graph = self.neg_np_graphs[i]
            if i == 0:
                merged_graph = graph
                merged_graph.initializer.extend(neg_graph.initializer)
                merged_graph.node.extend(neg_graph.node)
            else:
                merged_graph.initializer.extend(graph.initializer)
                merged_graph.initializer.extend(neg_graph.initializer)
                merged_graph.input.extend(graph.input)
                merged_graph.node.extend(graph.node)
                merged_graph.node.extend(neg_graph.node)   
        
        merged_graph.initializer.extend([half_initializer]) 
        merged_graph.node.extend([pos_concat,neg_concat,Sub,div,Clip])
        # merged_graph.node.extend([pos_concat])
        merged_graph.output.extend([onnx.helper.make_tensor_value_info('concat_result',onnx.TensorProto.FLOAT,[1,-1])])
                                    # onnx.helper.make_tensor_value_info('denom_is_zero_pos_9',onnx.TensorProto.BOOL,[1,-1]),
                                    # onnx.helper.make_tensor_value_info('input_qua_div_pos_9',onnx.TensorProto.FLOAT,[1,-1])])
                                    # onnx.helper.make_tensor_value_info('input_qua_mul_neg_18',onnx.TensorProto.FLOAT,[1,-1])])
        del merged_graph.output[0]
        with open (r'quantile_graph.onnx','wb') as f:
            f.write(merged_graph.SerializeToString())
        return merged_graph
    
if __name__ == '__main__':
    with open(r'E:\USCPI_V2\autogluon_USCPI_first_cls\models\NeuralNetTorch_BAG_L1\S1F4\model.pkl','rb') as f:
        child = pickle.load(f)
    processor =child.processor
    quantile_transformer= processor.transformers_[1][1][1]
    quantile_transformer_onnx_generator(quantile_transformer).merge_graph
