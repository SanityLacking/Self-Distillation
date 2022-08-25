from keras.layers import Input, Dense
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import numpy as np
import tensorflow as tf
from keras.layers import Flatten
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
import os
import types
import tempfile

from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow.compat.v1.keras.backend as K
from time import time

# sess = tf.Session()
# K.set_session(sess)
def loadOriginalModel(filename):
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model(filename)
    return model

import json

def getLayerBytes(model,name = "", saveFile = True,printOutput = True,):
    # from tensorflow.contrib import slim
    layerBytes = []
    for i, layer in enumerate(model.layers):
        op_size = 0 
        inputByteSize = 1
        outputByteSize = 1
        float32Size = 4
        inShapes =[] 
        outShapes =[]
        if layer.output.get_shape():
            outShapes=(layer.output.get_shape().as_list())
        for dim in outShapes:
            if dim == None:
                outputByteSize *= float32Size
            else:
                outputByteSize *= dim
        if type(layer.input) == list:
            for inputs in layer.input:
                print(inputs.get_shape())
                if inputs.get_shape():
                    inShapes=(inputs.get_shape().as_list())
                temp = 1
                for dim in inShapes:
                    if dim == None:
                        temp *= float32Size
                    else:
                        temp *= dim
                inputByteSize += temp
        else:
            if layer.input.get_shape():
                inShapes=(layer.input.get_shape().as_list())
                temp = 1
                for dim in inShapes:
                    if dim == None:
                        temp *= float32Size
                    else:
                        temp *= dim
                inputByteSize += temp

        layerBytes.append({'name':layer.name,'inputSize':inputByteSize,'outputSize':outputByteSize, 'inputShape':inShapes,'outputShape':outShapes})
        if printOutput == True:
            print("name: {}, in shape: {}, out shape{}, output bytes: {}".format(layer.name, inShapes, outShapes, outputByteSize))
        if saveFile == True:
            f = open("models/{}_bytes.txt".format(name), "w")
            f.write(json.dumps(layerBytes))
            f.close()
    return layerBytes


#get the best layer to split on given the following parameters
def getFlopsFromArchitecture(model,name = "",saveFile = True, printOutput = True ):
    layerFlops = []
    config = model.get_config()
    flops = {}
    for i, layer in enumerate(model.layers):
        K.clear_session()
        whole_model = keras.Model.from_config(config)

        new_model = keras.models.Model(inputs=whole_model.inputs, outputs=whole_model.layers[i].get_output_at(0))
        new_model.summary()
        new_config = new_model.get_config()
        K.clear_session()
        run_metadata = tf.RunMetadata()
        new_model = keras.Model.from_config(new_config)
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'file:outfile=log.txt'
        profile = tf.compat.v1.profiler.profile(K.get_session().graph, cmd = 'scope', options = opts,run_meta=run_metadata)
        layerFlops.append({'name':layer.name,'flopsCumulative':profile.total_float_ops})
        flops['{}_{}'.format(i,type(layer))] = profile.total_float_ops
        if printOutput == True:
            print("flops {}".format(flops))
        if saveFile == True:
            f = open("models/{}_flops_old.txt".format(name), "w")
            f.write(json.dumps({flops}))
            f.close()
            f = open("models/{}_flops.txt".format(name), "w")
            f.write(json.dumps(layerFlops))
            f.close()
    return flops

def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
        
            return flops.total_float_ops


def getWholeFlops(filename="",name = "",saveFile = True, printOutput = True):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    # model = tf.keras.models.load_model(filename)
    with graph.as_default():
        with session.as_default():
            flops = {}
            model = tf.keras.models.load_model(filename)
            layerFlops = []
            # for i, layer in enumerate(model.layers):
            # print(type(model))
            model = tf.keras.models.load_model(filename)
            # new_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[i].get_output_at(0))
            model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

            model.summary()
            # new_model.save(tempModelFileName)
            # tf.keras.backend.clear_session()
            # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            # model = tf.keras.models.load_model(tempModelFileName)

            # Print trainable variable parameter statistics to stdout.
            ProfileOptionBuilder = tf.compat.v1.profiler.ProfileOptionBuilder
            opt = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph, options=opt)
            total_flops = flops.total_float_ops

            opt = (tf.compat.v1.profiler.ProfileOptionBuilder(
                            tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
                            .with_node_names()
                            .order_by('depth')
                            .with_file_output('models/'+'enc_log.txt')
                            .build()) 
            flops = tf.compat.v1.profiler.profile(graph, options=opt)
            enc_flops = flops.total_float_ops

            print ("========================================================")
            print ('Total Flops : {}'.format(total_flops))
            print ('Enc. Flops : {}'.format(enc_flops))
    # param_stats = tf.compat.v1.profiler.profile(
    #    tf.compat.v1.get_default_graph(),
    #     options=ProfileOptionBuilder.float_operation())

    # # Use code view to associate statistics with Python codes.
    # opts = ProfileOptionBuilder(
    #     ProfileOptionBuilder.trainable_variables_parameter()
    #     ).with_node_names().build()
    # param_stats = tf.compat.v1.profiler.profile(
    #     tf.compat.v1.get_default_graph(),
    #     cmd='code',
    #     options=opts)
    # # param_stats can be tensorflow.tfprof.GraphNodeProto or
    # # tensorflow.tfprof.MultiGraphNodeProto, depending on the view.
    # # Let's print the root below.
    # print('total_params: %d\n' % param_stats.total_parameters)

    return 
#older method to use if the keras.model.from_config doesn't work for some reason
#get the best layer to split on given the following parameters
def getLayerFlops(filename,name = "",saveFile = True, printOutput = True ):
    flops = {}
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with session.as_default():
            new_model = filename
            new_model.summary()
            new_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            opts['output'] = 'file:outfile=log.txt'
            profile = tf.compat.v1.profiler.profile(graph, cmd = 'scope', options = opts)
            print(profile)
    return flops


def getLayerFlops_old(filename,name = "",saveFile = True, printOutput = True ):
    flops = {}
    tempModelFileName= 'models/tempmodel.hdf5'
    tf.compat.v1.reset_default_graph()
    # if type(filename) != str:
    model = filename
    # else:
        # model = tf.keras.models.load_model(filename)
        
    layerFlops = []
    model.summary()
   
    for i, layer in enumerate(model.layers):
        if any(s in layer.name for s in ('softmax', 'dense_2','branch_flatten','max_pooling2d')):
        # if any(s in layer.name for s in ("input_1","conv2d_1","batch_normalization_1","activation_1","conv2d_2","batch_normalization_2","activation_2","conv2d_3","batch_normalization_3","activation_3","max_pooling2d_1","conv2d_4","batch_normalization_4","activation_4","conv2d_5","batch_normalization_5","activation_5","max_pooling2d_2","mixed0","mixed1","mixed2","mixed3","mixed4","mixed5","mixed6","mixed7","mixed8","mixed9","mixed10","dense_1","dense_2","global_average_pooling2d_1")):
        # if any(s in layer.name for s in ("input_1","conv2d_1","batch_normalization_1","activation_1","")):
            graph = tf.compat.v1.get_default_graph()
            session = tf.compat.v1.Session()
            with graph.as_default():
                with session.as_default():
                    # if (i>5):
                        # break
                    # tf.compat.v1.reset_default_graph()
                    # tf.keras.backend.clear_session()
                    print(type(model))
                    model = tf.keras.models.load_model(filename)
                    # model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
                    new_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[i].get_output_at(0))
                    new_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
                    new_model.summary()
                    new_model.save(tempModelFileName)
                    # tf.compat.v1.reset_default_graph() #model is profiling including the original model size, this is a bug TODO fix this

                    # K.clear_session()
                    # tf.keras.backend.clear_session()
                    # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
                    # tf.keras.backend.clear_session()
                    model = tf.keras.models.load_model(tempModelFileName)
                    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                    opts['output'] = 'file:outfile=log.txt'
                    profile = tf.compat.v1.profiler.profile(graph, cmd = 'scope', options = opts)
                    layerFlops.append({'name':layer.name,'flopsCumulative':profile.total_float_ops})
                    flops['{}_{}'.format(i,type(layer))] = profile.total_float_ops
                    if printOutput == True:
                        print("flops {}".format(flops))
                    if saveFile == True:
                        f = open("models/{}_flops_old.txt".format(name), "w")
                        f.write(json.dumps(flops))
                        f.close()
                        f = open("models/{}_flops.txt".format(name), "w")
                        f.write(json.dumps(layerFlops))
                        f.close()
            tf.compat.v1.reset_default_graph()
            # K.clear_session()
            # tf.keras.backend.clear_session()
            # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            # tf.keras.backend.clear_session()
           
    return flops


    
def getLayerFlops_new(filename="",name = "",saveFile = True, printOutput = True ):
    flops = {}
    tempModelFileName= 'models/tempmodel.hdf5'
    model = tf.keras.models.load_model(filename)
    layerFlops = []
    # import tensorflow as tf
    from keras_flops import get_flops
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    return flops.total_float_ops
    # return flops

if __name__ == "__main__":
    model = tf.keras.models.load_model('models/alexNetv5_alt8_branched.hdf5')
    # model.summary()
    # model.save("models/InceptionV3.h5")
    tf.keras.utils.plot_model(model, to_file='alexnetplot.png', show_shapes=True, show_layer_names=True)
    flops = getLayerFlops_old('models/alexNetv5_alt8_branched.hdf5',printOutput=True,name="alexnetv5")
    # print(flops)


# from tensorflow.python.framework.ops import get_stats_for_node_def
# graph_def = sess.graph.as_graph_def()
# for node in graph_def.node:
#     try:
#         stats = get_stats_for_node_def(sess.graph, node, 'flops')
#     except ValueError:
#         # Catch Exception When shape is incomplete. Skip it.
#         stats = None
# print("stats{}".format(stats))

# for i in range(2):
#     # new_model = Sequential()
#     # a = Input(shape=(28,28))
#     # new_model.Input(shape=(28,28))
#     inputs = Input(shape=(784,))

#     # a layer instance is callable on a tensor, and returns a tensor
#     x = Dense(64, activation='relu')(inputs)
#     x = Dense(64, activation='relu')(x)
#     predictions = Dense(10, activation='softmax')(x)

#     # This creates a model that includes
#     # the Input layer and three Dense layers
#     new_model = keras.models.Model(inputs=inputs, outputs=predictions)
#     new_model.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])

#     new_model.summary()
    # for j in range(i):
    #     # print("layer {}".format(model.layers[0]))
    #     # new_model.add(model.layers[j])
    #    new_model.add(Dense(32, input_dim=784))
    # new_model.add(Activation('relu'))
    # # new_model.compile(loss='categorical_crossentropy',
    # #         optimizer='sgd',
    # #         metrics=['accuracy']
    # #         )
    # # new_model.build((224,224,3))
    # # # new_model.set_in
    # new_model.summary()

    # # fn = K.function([model.layers[0].get_input_at(0)],[model.layers[i-1].get_output_at(0)])
    # x = model.layers[i-1].get_output_at(0)
    # new_model = keras.Model(inputs= model.input,outputs=x)
    # new_model.input_tensor = tf.constant('float32', shape=(1,224,224,3))
    # # var = K.variable(np.zeros((1,224,224,3)))
    # new_model.compile(loss='categorical_crossentropy',
    #             optimizer='sgd',
    #             metrics=['accuracy']
    #             )
    # # new_model.predict()
    # fn([x])
    # new_model.summary()
    # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # opts['output'] = 'file:outfile=log.txt'
    # profile = tf.compat.v1.profiler.profile(K.get_session().graph, cmd = 'scope', options = opts)
    # flopsOutput[i] = 'FLOP of model from 0 to layer {} = {}'.format(i, profile.total_float_ops)


    # color = 3 if RGB else 1
    # base_model = VGG19(weights='imagenet', include_top=False, pooling=None, input_shape=(img_rows, img_cols, color),
    # base_model.summary()
    # for layer in base_model.layers:
        # layer.trainable = False

    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    # predictions = Dense(nb_classes, activation='softmax')(x)

    # model = Model(inputs=base_model.input, outputs=predictions)
    # sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    # new_model_function = K.function([model.layers[0].get_input_at(0)],[model.layers[i].get_output_at(0)])
    # new_model = K.function([model.layers[0].get_input_at(0)],[model.layers[i].get_output_at(0)])
    # print(type(new_model))
    # profile = tf.compat.v1.profiler.profile(keras.backend.get_session().graph, cmd = 'scope', options = opts, run_meta=run_metadata)
    # print('FLOP of model from 0 to layer {} = {}'.format(i, profile.total_float_ops))


# print(model.layers[6].get_output_at(0))
# print(model.layers[7].get_input_at(0))
# def stats_graph(graph):
#     flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
#     params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
#     print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

# sess = K.get_session()
# graph = sess.graph
# stats_graph(graph)
# writer = tf.summary.FileWriter(logdir="logs/{}".format(time()), graph=keras.backend.get_session().graph)
# writer.flush()


# print(model.get_input_shape_at(0))
# run_metadata = tf.RunMetadata()
# # metadata = tf.RunMetadata()
# opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
# print(opts)
# opts['output'] = 'file:outfile=log.txt'
# # print("training params:{}".format(tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()))
# opts['select'] = ['float_ops','peak_bytes','input_shapes']
# # opts['run_metadata'] = run_metadata
# # opts['max_depth'] = 10
# # opts['account_type_regexes'] = ['']
# # opts['trim_name_regexes'] = ['.*batch_normalization*.']
# opts['order_by'] ='depth'
# # opts['hide_name_regexes']=['.*training.*','.*loss.*','.*dropout.*']
# # opts['show_name_regexes']=['.*batch.*']

# profile = tf.compat.v1.profiler.profile(keras.backend.get_session().graph, cmd = 'scope', options = opts, run_meta=run_metadata)
# # print("type {}".format(type(profile)))

# from google.protobuf.json_format import MessageToDict
# text = MessageToDict(profile)
# # print("dict:{}".format(text))

# print(flopsOutput)

# tf.compat.v1.profiler.write_op_log(keras.backend.get_session().graph,"")
# prof = tf.profiler(graph=keras.backend.get_session().graph)
# print(dir(prof))
# prof.profile(options=tf.profiler.ProfileOptionBuilder.float_operation())
# print(dir(profile))

# f = open("log.txt", "r")
# print(f.read())




# print("\n".join(profile.DESCRIPTOR.fields_by_name.keys()))

# compare_table = [(i, getattr(profile,i)) for i in profile.DESCRIPTOR.fields_by_name.keys()]
# print(tabulate.tabulate(compare_table, headers=["Name","Value"]))
# print(type(profile.children))
# response = {}
# print(profile.children[0][0])

# print('FLOP = ', profile.total_float_ops)
# print(' = ', profile.total_float_ops)
# results = model.evaluate(x_test, test_labels)
# model.save('model.h5')

# flops = tf.profiler.profile(graph,\
#      options=tf.profiler.ProfileOptionBuilder.float_operation())
# print('FLOP = ', flops.total_float_ops)
# # model.fit(data, labels)  # starts training
# idx = 4  # index of desired layer

# edge_model = Model(model.layers[0].input, model.layers[2].get_output_at(0))
# edge_model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# edge_model.summary()

# # print(model2.predict())

# # # input_NN2 = Input(64,)
# # input_shape = model.layers[3].get_input_shape_at(0) # get the input shape of desired layer
# # layer_input = Input(shape=input_shape) # a new input tensor to be able to feed the desired layer


# # x = layer_input
# # for layer in model.layers[3:]:
# #     x = layer(x)


# cloud_model = Model(model.layers[3].get_input_at(0), model.layers[6].get_output_at(0))
# cloud_model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# cloud_model.summary()


# import keras.backend as K
# # print(model.layers[0].input_shape)
# # newData = np.random.rand(1,784)



# newData = x_test[:1]
# print(newData.shape)

# get_edge_model_result = K.function([model.layers[0].input],[model.layers[2].get_output_at(0)])
# # print(dir(get_edge_model_result))
# # with a Sequential model
# get_cloud_model_result = K.function([model.layers[3].get_input_at(0)],[model.layers[6].get_output_at(0)])


# output = get_edge_model_result([newData])[0]
# print("output at Edge Node: {}".format(output.shape))
# result_output = get_cloud_model_result([output])[0]
# print("output is: {}".format(result_output))
# print("label is: {}".format(y_test[:1]))

# edge_output = edge_model.predict(newData)
# print("output at Edge Node: {}".format(edge_output.shape))
# result_output = cloud_model.predict(edge_output)
# print("output is: {}".format(result_output))
# print("label is: {}".format(y_test[:1]))

# from sklearn.utils.extmath import softmax
# print(softmax(result_output))