import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
import re
file_name = 'models/alexNetv5_alt8_branched.hdf5'
log_path = './profiles/cpm_352_profile/'

if not os.path.exists(log_path):
    os.makedirs(log_path)

from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

def get_flops(model):
    model.summary()
    #first up, get layer names and order.
    modelLayers ={}
    for i, layer in enumerate(model.layers):
        modelLayers[layer.name] = {"depth":i,"flops":0, "cumulativeFlops":0}
    # print(modelLayers)
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    print(tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        # opts.order_by("depth") ## so this order_by option has been removed from TF, but remains in the docs. so we have to make it ourselves.
        opts["output"]="file:outfile=models/_flops_2.txt"
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="scope", options=opts)
        print(flops)
        # for i, layer in enumerate(flops):
        #     print(layer)
        print("outputfile: ")

        # f = open("models/_flops.txt", "r")
        # #skip to the data we want
        # for i in range(0,8):
        #     f.readline()
        # #process through each line of the profiler output
        # for x in f:
        #     # print(x)    
        #     #NOTE, these regex should work fine as long as you haven't used backslashes / or parenthesis in your model and layer names.
        #     name = (re.search("(\/[0-z]*\/)", x).group().replace("/","")) #find the layer name
        #     layerflops = (re.search("(\([0-z.]*\/)", x).group().replace("/","").replace("(","")) #find the flop count
        #     modelLayers[name]["flops"] = layerflops
        # f.close()
        print(modelLayers)
        return flops.total_float_ops
        # model.summary()
        # #first up, get layer names and order.
        # modelLayers ={}
        # for i, layer in enumerate(model.layers):
        #     modelLayers[layer.name] = {"depth":i,"flops":0, "cumulativeFlops":0}
        # # print(modelLayers)
        # concrete = tf.function(lambda inputs: model(inputs))
        # concrete_func = concrete.get_concrete_function(
        #     [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
        # frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        # print(tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
        # with tf.Graph().as_default() as graph:
        #     tf.graph_util.import_graph_def(graph_def, name='')
        #     run_meta = tf.compat.v1.RunMetadata()
        #     opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        #     # opts.order_by("depth") ## so this order_by option has been removed from TF, but remains in the docs. so we have to make it ourselves.
        #     opts["output"]="file:outfile=models/_flops.txt"
        #     flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="scope", options=opts)
        #     print(flops)
        #     # for i, layer in enumerate(flops):
        #     #     print(layer)
        #     print("outputfile: ")

        #     f = open("models/_flops.txt", "r")
        #     #skip to the data we want
        #     for i in range(0,8):
        #         f.readline()
        #     #process through each line of the profiler output
        #     for x in f:
        #         # print(x)    
        #         #NOTE, these regex should work fine as long as you haven't used backslashes / or parenthesis in your model and layer names.
        #         name = (re.search("(\/[0-z]*\/)", x).group().replace("/","")) #find the layer name
        #         layerflops = (re.search("(\([0-z.]*\/)", x).group().replace("/","").replace("(","")) #find the flop count
        #         modelLayers[name]["flops"] = layerflops



        #     f.close()
        #     print(modelLayers)
        #     return flops.total_float_ops

if __name__ == '__main__':
    model = tf.keras.models.load_model(file_name)
    get_flops(model)