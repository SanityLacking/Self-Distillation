import os
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import tensorflow as tf
# mnist = tf.keras.datasets.mnist
import keras.backend as K
from keras.layers import Flatten

###
#split a model file found at modelInput into subsections. 
# endLayer defaults to the final layer of the model
# startLayer defaults to the first layer of the model
# without paramaters will save a identical copy of the original model.
def save(modelInput, startLayer = 0, endLayer =-1, newName =""):
    assert type(endLayer) == int
    assert type(startLayer) == int

    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            modelInput.save('models/{}.hdf5'.format(os.path.basename(newName)))
        
            # model_slice = get_edge_model_result = K.function([KerasModel.layers[0].input],[KerasModel.layers[6].get_output_at(0)])
            # model_slice.save('model_slice.hdf5')
            # model_slice.summary()

    # def buildSplitModels(self, model, modelName = '', layerNames = [],saveFile = True ):
    #     layerFlops = []
    #     config = model.get_config()
    #     flops = {}
    #     for i, layer in enumerate(model.layers):
    #         print(layer.name)
    #         if layer.name in layerNames:
                
    #             K.clear_session()
    #             whole_model = keras.Model.from_config(config)

    #             new_model = keras.models.Model(inputs=whole_model.inputs, outputs=whole_model.layers[i].get_output_at(0))
    #             #new_model.summary()
    #             new_config = new_model.get_config()
    #             K.clear_session()
    #             run_metadata = tf.RunMetadata()
    #             new_model = keras.Model.from_config(new_config)

    #             if saveFile == True:
    #                 print("savingmodel to : models/{}_{}".format(modelName,layer.name))
    #                 new_model.save('models/{}_{}.hdf5'.format(modelName,layer.name))


if __name__ == '__main__':
    import sys
    modelName = sys.argv[1]
    startLayer = sys.argv[2]
    endLayer = sys.argv[3]
    newName = ""
    try:
        newName = sys.argv[4]
    except Exception as identifier:
        pass

    
    saveModel = SaveModel()
    # node1.loadModel('models_old/saved-model-alexnet-03-0.80.hdf5')
    saveModel.split(modelName, int(startLayer), int(endLayer), newName)
    print("Task Complete")