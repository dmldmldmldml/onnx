import numpy as np
import onnx
from scipy.stats import norm


def np_interp_graph(references,quantile,tree_node,mode = 'positive'):
    
    if mode == 'positive':
        quantile = quantile.reshape(1,-1)
        input_name = 'input'
        
    elif mode == 'negative':
        quantile = -quantile[::-1].reshape(1,-1)
        neg_op = onnx.helper.make_node(
            "Neg",
            name = 'neg_input',
            inputs=["input"],
            outputs=["neg_input"],

        )
        input_name = 'neg_input'
        

    references = np.clip(norm.ppf(references), -5.199337582605575, 5.199337582605575).reshape(1,-1)
    
    reference_initializer = onnx.helper.make_tensor(name ='reference', 
                                    data_type=onnx.TensorProto.FLOAT, 
                                    dims=references.shape,
                                    vals=references.astype(np.float32).tobytes(), 
                                    raw=True)

    quantiles_initializer = onnx.helper.make_tensor(name ='quantile', 
                                    data_type=onnx.TensorProto.FLOAT, 
                                    dims=quantile.shape, 
                                    vals=quantile.astype(np.float32).tobytes(), 
                                    raw=True)


    output = onnx.helper.make_tensor_value_info(name= 'y',elem_type= onnx.TensorProto.FLOAT, shape = (1,1))    
    input = onnx.helper.make_tensor_value_info(name= 'input',elem_type= onnx.TensorProto.FLOAT, shape = (1,1))
    
    
    min_len = onnx.helper.make_tensor(name ='min_len', data_type=onnx.TensorProto.FLOAT,
                                        dims=(1,),
                                        vals=np.array([0]).astype(np.float32).tobytes(),
                                        raw=True)
    
    max_len = onnx.helper.make_tensor(name ='max_len', data_type=onnx.TensorProto.FLOAT,
                                        dims=(1,),
                                        vals=np.array([90]).astype(np.float32).tobytes(),
                                        raw=True)

    min_ref = onnx.helper.make_tensor(name ='min_ref', data_type=onnx.TensorProto.FLOAT,
                                        dims=(1,),
                                        vals=np.array([np.min(references)]).astype(np.float32).tobytes(),
                                        raw=True)
    
    max_ref = onnx.helper.make_tensor(name ='max_ref', data_type=onnx.TensorProto.FLOAT,
                                        dims=(1,),
                                        vals=np.array([np.max(references)]).astype(np.float32).tobytes(),
                                        raw=True)

    ex_num = onnx.helper.make_tensor(name ='ex_num', 
                                     data_type=onnx.TensorProto.FLOAT, 
                                     dims=(1,1), 
                                     vals=np.array([[1]]).astype(np.float32).tobytes(), 
                                     raw=True)
    
    min_quantile = onnx.helper.make_tensor(name ='min_quantile', data_type=onnx.TensorProto.FLOAT,
                                        dims=(1,),
                                        vals=np.array([np.min(quantile)]).astype(np.float32).tobytes(),
                                        raw=True)
    max_quantile = onnx.helper.make_tensor(name ='max_quantile', data_type=onnx.TensorProto.FLOAT,
                                        dims=(1,),
                                        vals=np.array([np.max(quantile)]).astype(np.float32).tobytes(),
                                        raw=True)
    zero = onnx.helper.make_tensor(name ='Zero', data_type=onnx.TensorProto.FLOAT,
                                        dims=(1,1),
                                        vals=np.array([[0]]).astype(np.float32).tobytes(),
                                        raw=True)
    
    #对输入做clip
    input_clip = onnx.helper.make_node('Clip',
                                        name = 'input_clip',
                                        inputs=[input_name,'min_quantile','max_quantile'],
                                        outputs=['input_clip_output'])

    # 树节点
    tree_node_0 = tree_node

    tree_node_0.input[0] = 'input_clip_output'
    tree_node_0.output[0] = 'tree_output'

    variable = onnx.helper.make_node('Clip',
                                     name = 'variable_clip',
                                     inputs=['tree_output','min_len','max_len'],
                                     outputs=['variable'])

    #tree_ix的前一个index
    tree_ix = onnx.helper.make_node("Sub",
                                    name = "tree_ix",
                                    inputs=['variable','ex_num'],outputs=['tree_ix'])
    
    tree_ix_cast = onnx.helper.make_node(
                    "Cast",
                    name = "tree_ix_cast",
                    inputs=["tree_ix"],
                    outputs=["tree_ix_cast"],
                    to=getattr(onnx.TensorProto,"INT32")

                )
    variable_cast = onnx.helper.make_node(
                    "Cast",
                    name = "variable_cast",
                    inputs=["variable"],
                    outputs=["variable_cast"],
                    to=getattr(onnx.TensorProto,"INT32")
                )

    #对tree_Ix做clip
    # tree_ix_clip = onnx.helper.make_node('Clip',
    #                                     name = 'tree_ix_clip',
    #                                     inputs=['tree_ix_output','min_len','max_len'],
    #                                     outputs=['tree_ix'])

    #输入所在quantile和reference的对应的值
    ref_segment =  onnx.helper.make_node("GatherElements",name = 'ref_segment',inputs=['reference','variable_cast'],outputs=['ref_seg'],axis=1)
    qua_segment=  onnx.helper.make_node("GatherElements",name = 'qua_segment',inputs=['quantile','variable_cast'],outputs=['qua_seg'],axis=1)
    ref_ex_segment = onnx.helper.make_node("GatherElements",name = 'ref_ex_segment',inputs=['reference','tree_ix_cast'],outputs=['ref_seg_ex'],axis=1)
    qua_ex_segment = onnx.helper.make_node("GatherElements",name = 'qua_ex_segment', inputs=['quantile','tree_ix_cast'],outputs=['qua_seg_ex'],axis=1)
    

    # 求出所在区间的差值
    ref_seg_len = onnx.helper.make_node("Sub",name='ref_seg_len',inputs=['ref_seg','ref_seg_ex'],outputs=['ref_seg_sub'])
    qua_seg_len = onnx.helper.make_node("Sub",name = 'qua_seg_len', inputs=['qua_seg','qua_seg_ex'],outputs=['qua_seg_sub'])



    # 输入与所在区间的差值及占比
    ref_seg_sub = onnx.helper.make_node("Sub",name = 'ref_seg_sub',inputs=['input_clip_output','qua_seg_ex'],outputs=['qua_input_sub'])
    input_qua_div = onnx.helper.make_node("Div",name = 'input_qua_div ',inputs=['qua_input_sub','qua_seg_sub'],outputs=['input_qua_div'])
    
    #创建 Equal 节点来检查 denom 是否等于 0
    equal_node = onnx.helper.make_node(
                        'Equal', 
                        inputs=['qua_seg_sub', 'Zero'], 
                        outputs=['denom_is_zero'], 
                        name='denom_is_zero_node'
                    )

    # 求出所在区间的差值与所在区间长度的乘积
    ref_seg_sub_mul = onnx.helper.make_node("Mul",name = 'ref_seg_sub_mul',inputs=['ref_seg_sub','input_qua_div'],outputs=['ref_seg_sub_mul'])

    #得到最终输出 
    ref_seg_sub_mul_add = onnx.helper.make_node("Add",name = 'ref_seg_sub_mul_add', inputs=['ref_seg_sub_mul','ref_seg_ex'],outputs=['output'])

    # 创建 Where 节点来选择输出
    where_node = onnx.helper.make_node(
                'Where', 
                inputs=['denom_is_zero', 'ref_seg_ex', 'output'], 
                outputs=['result'], 
                name='where_node'
    )
    #最终输出做clip取值
    clip = onnx.helper.make_node(
        "Clip", 
        name = 'output_clip',
        inputs=["result", "min_ref", "max_ref"],
        outputs=["y"],
        )
    
    nodes = [variable,ref_segment,tree_ix,tree_ix_cast,variable_cast,
             tree_node_0,qua_segment,ref_ex_segment,qua_ex_segment, clip,
             ref_seg_len,qua_seg_len,ref_seg_sub,input_qua_div,ref_seg_sub_mul,ref_seg_sub_mul_add,input_clip,where_node,equal_node]
    

    if mode == 'negative':
       
        nodes.append(neg_op)

    initializer = [ex_num,reference_initializer,quantiles_initializer,min_ref,max_ref,min_len,max_len,min_quantile,max_quantile,zero]
    graph = onnx.helper.make_graph(nodes = nodes, name = 'np.interp', inputs= [input], outputs = [output], initializer=initializer)
    
    return graph


    

