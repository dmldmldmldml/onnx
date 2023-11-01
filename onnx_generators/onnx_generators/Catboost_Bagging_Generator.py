import os

import onnx

from .Abstract_onnx_generator import Abstract_ONNX_Generator
from .operators import argmax_operator, mean_operator


class Catboost_Bagging_onnx_generator(Abstract_ONNX_Generator):

    def __init__(self, model_dir) -> None:
        super().__init__(model_dir)
        
    
    def child_to_onnx(self):
        for i in range(len(self.children)):
            self.children[i].model.save_model(
                        os.path.join(os.path.dirname(self.children_path[i]),"onnx_model.onnx"),
                        format="onnx",
                        export_parameters={
                            'onnx_domain': 'ai.catboost',
                            'onnx_model_version': 1,
                            'onnx_doc_string': 'test model for BinaryClassification',
                            'onnx_graph_name': 'CatBoostModel_for_BinaryClassification'
                        }
                    )
    
    def merge_onnx_models(self):
        merged_model = onnx.ModelProto(ir_version=self.onnx_models[0].ir_version,
                        producer_name="ZhengLi",
                        producer_version=self.onnx_models[0].producer_version,
                        opset_import= self.onnx_models[0].opset_import)
        merged_model = super().merge_graphs(merged_model = merged_model, onnx_models = self.onnx_models)
        operators = self.making_nodes
        merged_model.graph.node.extend(operators)
        if self.problem_type == 'regression':
            output = onnx.helper.make_tensor_value_info("result", 1, [1])
        else:
            output = onnx.helper.make_tensor_value_info("final_output", 7, [1])
        merged_model.graph.output.extend([output])
        return merged_model
    
    @property
    def making_nodes(self):
        if self.problem_type == 'regression':
            operators = [mean_operator(output_name='result', inputs_name='variable',num_of_children = len(self.children))]
        else:
            operators = [mean_operator(output_name='result', inputs_name='probability_tensor',num_of_children = len(self.children)), 
                        argmax_operator(output_name='final_output', inputs_name='result')]
        return operators
    
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
                if nodes[j].op_type in ['TreeEnsembleClassifier','ZipMap','TreeEnsembleRegressor']:
                    nodes[j].name = nodes[j].op_type + str(i)
                if j == 0:
                    for k in range(len(nodes[j].output)):
                        nodes[j].output[k] = nodes[j].output[k] + str(i)
                else:
                    for k in range(len(nodes[j].input)):
                        nodes[j].input[k] = nodes[j].input[k] + str(i)
                    for k in range(len(nodes[j].output)):
                        nodes[j].output[k] = nodes[j].output[k] + str(i)
        return onnx_models
   
    def transform(self):
        self.child_to_onnx() 
        self.onnx_models =self.load_onnx_models(self.children_path)
        
        self.onnx_models = self.rename_node_name(self.onnx_models)
        self.onnx_models = self.merge_onnx_models()
        self.save(self.onnx_models)


if __name__ == '__main__':
    generator = Catboost_Bagging_onnx_generator(r'E:\Bagging2onnx\AutogluonOnnxGenerator_1.0\UKCPPI_3\models\CatBoost_BAG_L1')
    generator.transform()