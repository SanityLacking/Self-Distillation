

# import the necessary packages
import brevis

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import time
import json
import math
import pydot
import os

from brevis.utils import *
from brevis.branches import branch
from brevis.dataset import prepare
from brevis.initNeptune import Neptune

# import branchyNet
#class for building a seflDistilation branching model.

class SelfDistilation(brevis.BranchModel):

     # initialize an implementation of alexnet branching that uses the selfdistil methodology
    def alexnet(numEpocs = 2, modelName="", saveName ="",transfer = True,customOptions=""):
        x = tf.keras.models.load_model("models/{}".format(modelName))

        x.summary()
        if saveName =="":
            saveName = modelName
        tf.keras.utils.plot_model(x, to_file="{}.png".format(saveName), show_shapes=True, show_layer_names=True)
        # funcModel = models.Model([input_layer], [prev_layer])
        # funcModel = branch.add(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],branch.newBranch)
        # ["max_pooling2d","max_pooling2d_1","dense"]
        funcModel = branch.add_distil(x,["max_pooling2d","max_pooling2d_1","dense"],[branch.newBranch_flatten_evidence],exact=True)
        #so to self distil, I have to pipe the loss from the main exit back to the branches.
        funcModel.summary()
        # funcModel.save("models/{}".format(saveName))
        dataset = prepare.dataset_distil(tf.keras.datasets.cifar10.load_data(),64,5000,22500,(227,227))
        # funcModel = branchingdnn.models.trainModelTransfer(funcModel,
        #                                                     dataset,
        #                                                     epocs = numEpocs,
        #                                                     save = False,
        #                                                     transfer = transfer,
        #                                                     saveName = saveName,
        #                                                     customOptions=customOptions,
        #                                                     tags =["v6","drt"])
        # funcModel.save("models/{}".format(saveName))
        # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
        return None


    def normal(student_model,data, student_output=[]):  
        epocs = 10
        student_model = branch.add(student_model,["max_pooling2d"],branch.newBranch_bottleneck2)
        if len(student_output)>0:
            student_model = keras.Model(inputs=student_model.inputs, outputs=student_model.get_layer(student_output[0]).output, name="student_model")
        
        for i in range(len(student_model.layers)):
            print(student_model.layers[i].name)
            if "branch" in student_model.layers[i].name:
                print("setting branch layer training to true")
                student_model.layers[i].trainable = True
            else: 
                print("setting main layer training to false")
                student_model.layers[i].trainable = False               
        student_model.compile(optimizer=tf.optimizers.SGD(lr=0.001,clipvalue=0.5), loss='SparseCategoricalCrossentropy', metrics=['accuracy',confidenceDifference],run_eagerly=True)
        

        neptune_cbk = Neptune.getcallback(name = "feature_distill_example", tags =["knowledge_distill","example"])
        train_ds, test_ds, validation_ds = prepare.dataset(data,32,5000,22500,(227,227))


        # Initialize and compile distiller
        student_model.compile(
            optimizer=tf.optimizers.SGD(lr=0.001,clipvalue=0.5),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
            loss='SparseCategoricalCrossentropy', 
            # student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            # distillation_loss_fn=keras.losses.KLDivergence(),
            # alpha=0.1,
            # temperature=10,
        )

        # Distill teacher to student
        student_model.fit(train_ds,
            epochs=epocs,
            validation_data=validation_ds,
            validation_freq=1,
            # batch_size=1,
            callbacks=[neptune_cbk])

        # Evaluate student on test dataset
        student_model.evaluate(test_ds, verbose=2)
        student_model.save("models/{}".format("normal_student.hdf5"))
        return student_model.student




    ''' An implementation of the standard distillation method for comparision of my self built version
        you can compare self distillation as well by just using two instances of the same model for the teacher and the student.
        Just have the student exits be the branches.
    '''
    def normal_distillation(student_model, teaching_model, data, student_output=[], teacher_output=[] ):
        epocs = 10
        student_model = branch.add(student_model,["max_pooling2d"],branch.newBranch_bottleneck)
        if len(student_output)>0:
            student_model = keras.Model(inputs=student_model.inputs, outputs=student_model.get_layer(student_output[0]).output, name="student_model")
        
        if len(teacher_output)>0:
            teaching_model = keras.Model(inputs=teaching_model.inputs, outputs=teacher_output, name="teacher_model")
        for i in range(len(student_model.layers)):
            print(student_model.layers[i].name)
            if "branch" in student_model.layers[i].name:
                print("setting branch layer training to true")
                student_model.layers[i].trainable = True
            else: 
                print("setting main layer training to false")
                student_model.layers[i].trainable = False               
        student_model.compile(optimizer=tf.optimizers.SGD(lr=0.001,clipvalue=0.5), loss='SparseCategoricalCrossentropy', metrics=['accuracy',confidenceDifference],run_eagerly=True)
        
        teaching_model.compile(optimizer=tf.optimizers.SGD(lr=0.001,clipvalue=0.5), loss='SparseCategoricalCrossentropy', metrics=['accuracy',confidenceDifference],run_eagerly=True)

        neptune_cbk = Neptune.getcallback(name = "feature_distill_example", tags =["knowledge_distill","example"])
        train_ds, test_ds, validation_ds = prepare.dataset(data,32,5000,22500,(227,227))


        # Initialize and compile distiller
        distiller = SelfDistilation.Distiller(student=student_model, teacher=teaching_model)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=10,
        )

        # Distill teacher to student
        distiller.fit(train_ds,
            epochs=epocs,
            validation_data=validation_ds,
            validation_freq=1,
            # batch_size=1,
            callbacks=[neptune_cbk])

        # Evaluate student on test dataset
        distiller.evaluate(test_ds, verbose=2)
        distiller.student.save("models/{}".format("distiller_student.hdf5"))
        return distiller.student


    class Distiller(keras.Model):
        def __init__(self, student, teacher):
            super(SelfDistilation.Distiller, self).__init__()
            self.teacher = teacher
            self.student = student

        def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3,
        ):
            """ Configure the distiller.

            Args:
                optimizer: Keras optimizer for the student weights
                metrics: Keras metrics for evaluation
                student_loss_fn: Loss function of difference between student
                    predictions and ground-truth
                distillation_loss_fn: Loss function of difference between soft
                    student predictions and soft teacher predictions
                alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
                temperature: Temperature for softening probability distributions.
                    Larger temperature gives softer distributions.
            """
            super(SelfDistilation.Distiller, self).compile(optimizer=optimizer, metrics=metrics)
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
            self.alpha = alpha
            self.temperature = temperature

        def train_step(self, data):
            # Unpack data
            x, y = data

            # Forward pass of teacher
            teacher_predictions = self.teacher(x, training=False)

            with tf.GradientTape() as tape:
                # Forward pass of student
                student_predictions = self.student(x, training=True)

                # Compute losses
                student_loss = self.student_loss_fn(y, student_predictions)
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics configured in `compile()`.
            self.compiled_metrics.update_state(y, student_predictions)

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update(
                {"student_loss": student_loss, "distillation_loss": distillation_loss}
            )
            return results

        def test_step(self, data):
            # Unpack the data
            x, y = data

            # Compute predictions
            y_prediction = self.student(x, training=False)

            # Calculate the loss
            student_loss = self.student_loss_fn(y, y_prediction)

            # Update the metrics.
            self.compiled_metrics.update_state(y, y_prediction)

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update({"student_loss": student_loss})
            return results