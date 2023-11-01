import os
from os.path import join

import onnx
import onnxmltools
from onnxconverter_common.data_types import (DictionaryType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType)

from .Abstract_onnx_generator import Abstract_ONNX_Generator
from .operators import argmax_operator, mean_operator


class LightGBM_Bagging_onnx_generator(Abstract_ONNX_Generator):

    def __init__(self, model_dir) -> None:
        super().__init__(model_dir)
        self.initial_types = self.get_initial_types(self.input_format)
        self.child_to_onnx()
        self.onnx_models = super().load_onnx_models(self.children_path)

    def get_initial_types(self,input_format):
        input_dim = 0 
        for i in range(len(input_format)):
            #TODO: lightgbm only support float type, with int type it needs to be casted after model built.
            # if input_format[i][0] == 'int':
            #     initial_types.append(('input'+str(i),Int64TensorType([-1,len(input_format[i][1])])))
            if input_format[i][0] == 'float' or input_format[i][0] == 'int':
                input_dim += len(input_format[i][1])
        initial_types = [('input', FloatTensorType([-1,input_dim]))]
        return initial_types

    def child_to_onnx(self):
        if self.children:
            for i in range(len(self.children)):
                onnx_model = onnxmltools.convert_lightgbm(self.children[i].model, initial_types = self.initial_types,target_opset=15,zipmap=False)
                export_path = join(os.path.dirname(self.children_path[i]),'onnx_model.onnx')
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
        if self.problem_type == 'regression':
            output = onnx.helper.make_tensor_value_info("result", 1, [1])
        else:
            output = onnx.helper.make_tensor_value_info("final_output", 7, [1])
        merged_model.graph.output.extend([output])
        return merged_model
    
    def making_nodes(self,num_of_children):
        if self.problem_type == 'regression':
            operators = [mean_operator(output_name='result', inputs_name='variable',num_of_children = num_of_children)]
        else:
            operators = [mean_operator(output_name='result', inputs_name='probabilities',num_of_children = num_of_children), 
                        argmax_operator(output_name='final_output', inputs_name='result')]
        return operators
    

    def transform(self):
        self.merged_onnx_model = self.merge_onnx_models()
        self.save(self.merged_onnx_model)



if __name__ == "__main__":
    model_dir = r'E:\Bagging2onnx\AutogluonOnnxGenerator_1.0\autogluon_USCPI_reg\models\LightGBMLarge_BAG_L1'
    model = LightGBM_bagging_onnx_generator(model_dir=model_dir)
    # test = np.random.rand(1,15).astype(np.float32)
    model.transform()
    # model.test(test)