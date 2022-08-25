import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

import keras.backend as K
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu",input_shape=(3, 3)))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))

# Call model on a test input
x = tf.ones((3, 3))
y = model(x)

startLayer = 0
endLayer = 2


#get_edge_model_result = K.function([model.layers[startLayer].input],[model.layers[endLayer].get_output_at(0)]) #TODO maybe store this in the modeldict? 
print([model.layers[startLayer].input])
print([model.layers[endLayer].get_output_at(0)])

print(model.layers[startLayer].input)
print(model.layers[endLayer].get_output_at(0))
#print(get_edge_model_result(x))