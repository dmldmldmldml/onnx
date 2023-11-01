import onnx


def mean_operator(output_name,inputs_name,num_of_children):
    mean = onnx.helper.make_node(
                        "Mean",
                        inputs=[inputs_name+ str(i) for i in range(num_of_children)],
                        outputs=[output_name])
    return mean

def argmax_operator(output_name,inputs_name):
    argmax = onnx.helper.make_node(
                        "ArgMax", inputs=[inputs_name], outputs=[output_name], axis=1, keepdims=0)
    return argmax

def softmax_operator(output_name,inputs_name):
    softmax = onnx.helper.make_node(
                        "Softmax", inputs=[inputs_name], outputs=[output_name])
    return softmax

def subtract_operator(subtract_set,output_name):
    subtract = onnx.helper.make_node(
                        "Sub", name = 'Subtraction',inputs=list(subtract_set), outputs=[output_name])
    return subtract

def concat_operator(input_set,output_name):
    concat = onnx.helper.make_node(
                        "Concat",input = list(input_set),outputs =[output_name],axis = 0
    )
def div_operator(input_set,output_name):
    div = onnx.helper.make_node(
                        "Div", name = 'div',inputs=list(input_set), outputs=[output_name])
    return div

def mul_operator(input_set,output_name):
    mul = onnx.helper.make_node(
                        "Mul",inputs=list(input_set), outputs=[output_name])
    return mul

def add_operator(input_set,output_name):
    add = onnx.helper.make_node(
                        "Add", name = 'add',inputs=list(input_set), outputs=[output_name])
    return add