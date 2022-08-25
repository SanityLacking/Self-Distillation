# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import branching
from branching.utils import *

class branch:
    def add(model,  branch_layers = [], identifier =[""],exact = True, target_input= True, compact = False, num_outputs=10):
        """ add branches to the provided model, aka modifying an existing model to include branches.
            identifier: takes a list of names of layers to branch on is blank, branches will be added to all layers except the input and final layer. Can be a list of layer numbers, following the numbering format of model.layers[]
            If identifier is not blank, a branch will be added to each layer with identifier in its name. (identifier = "dense", all dense layers will be branched.)
            Warning! individual layers are defined according to how TF defines them. this means that for layers that would be normally grouped, they will be treated as individual layers (conv2d, pooling, flatten, etc)
            customBranch: optional function that can be passed to provide a custom branch to be inserted. Check "newBranch" function for default shape of branches and how to build custom branching function. Can be provided as a list and each branch will iterate through provided customBranches, repeating last the last branch until function completes
        """
        outputs = []
        for i in model.outputs:
            outputs.append(i)
        
        inputs = []
        ready = False
        targets= None
        for i in model.inputs:
            if i.name == "targets":
                ready = True
            inputs.append(i)
        if target_input:
            print("targets already present? ",ready)
            if not ready:
                print("added targets")
                targets = keras.Input(shape=(num_outputs,), name="targets")
                inputs.append(targets) #shape is (1,) for sparse_categorical_crossentropy
            else:
                targets = model.get_layer('targets').output

        #add targets as an input to the model so it can be used for the custom losses.
        #   input size is the size of the     
        #add target input 
        new_model = branching.BranchModel(inputs=inputs, outputs=outputs,name = model.name, freeze=model.freeze, custom_objects=model.custom_objects)

        if type(identifier) != list:
            identifier = [identifier]

        if type(branch_layers) != list:
            branch_layers = [branch_layers]
        if len(branch_layers) == 0:
            return new_model    
        branches = 0
        if len(identifier) > 0:
            print("Matching Branchpoint by id number")
            if type(identifier[0]) == int:
                for i in identifier: 
                    try:
                        outputs.append(branch_layers[min(branches, len(branch_layers))-1](model.layers[i].output,targets = targets))
                        branches=branches+1
                    except:
                        pass
            else:
                print("Matching Branchpoint by name")
                for i in range(len(model.layers)):
                    if exact == True:
                        if model.layers[i].name in identifier:
                            print("add Branch to branch point ",model.layers[i].name)
                            outputs.append(branch_layers[min(branches, len(branch_layers)-1)](model.layers[i].output,targets = targets))
                            branches=branches+1
                    else:
                        if any(id in model.layers[i].name for id in identifier):
                            print("add Branch to branch point ",model.layers[i].name)
                            outputs.append(branch_layers[min(branches, len(branch_layers)-1)](model.layers[i].output,targets = targets))
                            branches=branches+1
        else: #if identifier is blank or empty
            # print("nothing")
            for i in range(1-len(model.layers)-1):
                # print(model.layers[i].name)
                # if "dense" in model.layers[i].name:
                # outputs = newBranch(model.layers[i].output,outputs)
                outputs = branch_layers[min(branches, len(branch_layers))-1](model.layers[i].output,outputs,targets = targets)
                branches=branches+1
            # for j in range(len(model.layers[i].inbound_nodes)):
            #     print(dir(model.layers[i].inbound_nodes[j]))
            #     print("inboundNode: " + model.layers[i].inbound_nodes[j].name)
            #     print("outboundNode: " + model.layers[i].outbound_nodes[j].name)
        # print(outputs)
        # print(new_model.input)
        # outputs.pop(0)
        # print(outputs)
        # input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
        print(new_model.input)
        print(outputs)
        new_model = branching.BranchModel([new_model.input], [outputs], name = new_model.name, custom_objects=new_model.custom_objects)
        return new_model
    
   
    def add_distil(model, teacher=None,   branch_layers = [], identifier =[""], teaching_features=None, exact = True):
        """ 
            Add branches, self distilation style.
            teacher_softmax and teaching_features are expected to be a string and list of strings respectively. these strings are matched as names of layers to use.
        """
        outputs = []
        for i in model.outputs:
            outputs.append(i)

        targets= None
        if teacher:
            teacher = model.get_layer(teacher).output
        ### teaching_features 
        
        # if teaching_features:
        #     if type(teaching_features) != list:
        #         teaching_features = [teaching_features]

        #     for i, teacher_name in enumerate(teaching_features):
        #         teaching_features[i]=model.get_layer(teacher_name).output
        #         print("teaching",teaching_features[i])
        # teaching_features = [model.get_layer('max_pooling2d_1').output, model.get_layer('max_pooling2d_2').output, model.get_layer('max_pooling2d_2').output]
        # print("teaching Feature:", teaching_features)
        # teacher_softmax = [None]
        #get the loss from the main exit and combine it with the loss of the 
        if type(identifier) != list:
            identifier = [identifier]
        if type(branch_layers) != list:
            branch_layers = [branch_layers]
        if len(branch_layers) == 0:
            branch_layers = [branch.newBranch_flatten]
        branches = 0
        if len(identifier) > 0:
            if type(identifier[0]) == int:
                print("Matching Branchpoint by id number")
                for i in identifier: 
                    try:
                        outputs.append(branch_layers[min(branches, len(branch_layers))-1](model.layers[i].output,
                                                                                            teacher = teacher, 
                                                                                            # teaching_features = teaching_features[min(branches, len(teaching_features))-1])
                                                                                            ))
                        branches=branches+1
                    except:
                        pass
            else:
                print("Matching Branchpoint by name")
                for i in range(len(model.layers)):
                    if exact == True:
                        if model.layers[i].name in identifier:
                            print("add Branch to branch point ",model.layers[i].name)
                            # print(teaching_features[min(branches, len(teaching_features))-1])
                            outputs.append(branch_layers[min(branches, len(branch_layers)-1)](model.layers[i].output,
                                                                                            teacher = teacher, 
                                                                                            # teaching_features = teaching_features[min(branches, len(teaching_features))-1])
                                                                                            ))
                            branches=branches+1
                    else:
                        if any(id in model.layers[i].name for id in identifier):
                            print("add Branch to branch point ",model.layers[i].name)
                            outputs.append(branch_layers[min(branches, len(branch_layers)-1)](model.layers[i].output,
                                                                                            teacher = teacher, 
                                                                                            # teaching_features = teaching_features[min(branches, len(teaching_features))-1])
                                                                                            ))
                            branches=branches+1
        else: #if identifier is blank or empty
            # print("nothing")
            for i in range(1-len(model.layers)-1):
                # outputs = newBranch(model.layers[i].output,outputs)
                outputs = branch_layers[min(branches, len(branch_layers))-1](model.layers[i].output,
                                                                                            teacher = teacher, 
                                                                                            # teaching_features = teaching_features[min(branches, len(teaching_features))-1]
                                                                                            )
                branches=branches+1
        model = models.Model([model.input], [outputs], name="{}_branched".format(model.name))
            
        return model

    def add_ensemble(model, identifier =[""], customBranch = [],exact = True, target_input= True, compact = False, num_outputs=10):
        """ add branches to the provided model organizing the exits as an ensemble of exits.
            identifier: takes a list of names of layers to branch on is blank, branches will be added to all layers except the input and final layer. Can be a list of layer numbers, following the numbering format of model.layers[]
            If identifier is not blank, a branch will be added to each layer with identifier in its name. (identifier = "dense", all dense layers will be branched.)
            Warning! individual layers are defined according to how TF defines them. this means that for layers that would be normally grouped, they will be treated as individual layers (conv2d, pooling, flatten, etc)
            customBranch: optional function that can be passed to provide a custom branch to be inserted. Check "newBranch" function for default shape of branches and how to build custom branching function. Can be provided as a list and each branch will iterate through provided customBranches, repeating last the last branch until function completes
        """
        # model = keras.Model([model.input], [model_old.output], name="{}_branched".format(model_old.name))
        # model.summary()

        # outputs = [model.outputs]
        # outputs.append(newBranch(model.layers[6].output))
        # new_model = keras.Model([model.input], outputs, name="{}_branched".format(model.name))
        # new_model.summary()
        outputs = []
        for i in model.outputs:
            outputs.append(i)
        
        inputs = []
        ready = False
        
        targets= None
        
        for i in model.inputs:
            if i.name == "targets":
                ready = True
            inputs.append(i)
        if target_input:
            print("targets already present? ",ready)

            if not ready:
                print("added targets")
                targets = keras.Input(shape=(num_outputs,), name="targets")
                inputs.append(targets) #shape is (1,) for sparse_categorical_crossentropy
            else:
                targets = model.get_layer('targets').output

        #add targets as an input to the model so it can be used for the custom losses.
        #   input size is the size of the     
        #add target input 
        new_model = branching.BranchModel(inputs=inputs, outputs=outputs,name = model.name, freeze=model.freeze, custom_objects=model.custom_objects)

        if type(identifier) != list:
            identifier = [identifier]

        if type(customBranch) != list:
            customBranch = [customBranch]
        if len(customBranch) == 0:
            return new_model    
        branches = 0
        if len(identifier) > 0:
            print("Matching Branchpoint by id number")
            if type(identifier[0]) == int:
                for i in identifier: 
                    try:
                        outputs.append(customBranch[min(branches, len(customBranch))-1](model.layers[i].output,targets = targets))
                        branches=branches+1
                    except:
                        pass
            else:
                print("Matching Branchpoint by name")
                for i in range(len(model.layers)):
                    if exact == True:
                        if model.layers[i].name in identifier:
                            print("add Branch to branch point ",model.layers[i].name)
                            outputs.append(customBranch[min(branches, len(customBranch)-1)](model.layers[i].output,targets = targets))
                            branches=branches+1
                    else:
                        if any(id in model.layers[i].name for id in identifier):
                            print("add Branch to branch point ",model.layers[i].name)
                            outputs.append(customBranch[min(branches, len(customBranch)-1)](model.layers[i].output,targets = targets))
                            branches=branches+1
        else: #if identifier is blank or empty
            # print("nothing")
            for i in range(1-len(model.layers)-1):
                # print(model.layers[i].name)
                # if "dense" in model.layers[i].name:
                # outputs = newBranch(model.layers[i].output,outputs)
                outputs = customBranch[min(branches, len(customBranch))-1](model.layers[i].output,outputs,targets = targets)
                branches=branches+1
            # for j in range(len(model.layers[i].inbound_nodes)):
            #     print(dir(model.layers[i].inbound_nodes[j]))
            #     print("inboundNode: " + model.layers[i].inbound_nodes[j].name)
            #     print("outboundNode: " + model.layers[i].outbound_nodes[j].name)
        # print(outputs)
        # print(new_model.input)
        # outputs.pop(0)
        # print(outputs)
        # input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
        print(new_model.input)
        print(outputs)
        new_model = branching.BranchModel([new_model.input], [outputs], name = new_model.name, custom_objects=new_model.custom_objects)
        return new_model

    class EvidenceEndpoint_Model(tf.keras.Model):

        def __init__(self,name=''):
            super().__init__(name=name)
            self.layer = keras.layers.Layer(name=name)
            # self.layer.name = name
            # self.name=tf.compat.v1.get_default_graph().unique_name("branch")
            self.branchLayer1 = keras.layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))
            self.branchLayer2 = keras.layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))
            self.branchLayer3 = keras.layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))
            self.exit =branch.BranchEndpoint(10,name=tf.compat.v1.get_default_graph().unique_name("branch_exit"))

        def call(self, input_tensor, training=False):
            x = self.branchLayer1(input_tensor)
            x = tf.nn.relu(x)
            x = self.branchLayer2(x)
            x = tf.nn.relu(x)
            x = self.branchLayer3(x)
            x = self.exit(x)
            return x
        
    class branch_finished(keras.layers.Dropout):

        def __init__(self, rate=0, name=''):
            super().__init__(rate =rate, name=name)
            
        def call(self, input_tensor,branch_result):
            result = super().call(input_tensor)
            return result
    
    ## base class for all branching endpoint layers.
    class BranchEndpoint(keras.layers.Layer):
        def __init__(self, num_outputs, name=None, branch_exit = True, confidence_threshold=0, temperature=10, alpha=0.1, loss_coef = 1, feature_loss_coef=1,  **kwargs):
            super(branch.BranchEndpoint, self).__init__(name=name, **kwargs)
            # self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=Tru
            self.num_outputs = num_outputs
            self.branch_exit = branch_exit
            self.confidence_threshold = confidence_threshold
            self.evidence = softplus_evidence
            self.loss_coefficient = loss_coef
            self.feature_loss_coefficient = feature_loss_coef
            self.kl = tf.keras.losses.KLDivergence()
            self.loss_fn = keras.losses.sparse_categorical_crossentropy
            self.temperature = temperature
            self.alpha = alpha
        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'num_outputs': self.num_outputs,
                'name': self.name,
                'branch_exit':self.branch_exit,
                'confidence_threshold': self.confidence_threshold
            })
            return config

        def call(self, inputs, labels, teacher_sm=None, sample_weights=None):
            softmax = tf.nn.softmax(inputs)
            normal_loss = self.loss_fn(labels, softmax, sample_weights)
            total_loss =tf.reduce_mean(normal_loss)
            if teacher_sm is not None:
                kl_loss = self.kl( tf.nn.softmax(softmax / self.temperature, axis = 1 ),
                tf.nn.softmax(teacher_sm /self.temperature,axis=1))
                # print("KL_LOSS", kl_loss)
                # self.add_loss(kl_loss)
                total_loss += self.alpha * total_loss + (1- self.alpha) * kl_loss
                self.add_metric(kl_loss, name=self.name+"_KL")
            self.add_loss(total_loss)
            self.add_metric(tf.reduce_sum(self.losses), name=self.name+"_losses")
            return softmax

        def calcConfidence(self, outputs):
            return tf.reduce_mean(outputs)


    class EvidenceEndpoint(BranchEndpoint):
            def __init__(self, num_outputs, name=None, **kwargs):
                super(branch.EvidenceEndpoint, self).__init__(name=name,kwargs=kwargs)
                self.num_outputs = num_outputs
    #             self.kl = tf.keras.losses.KLDivergence()
                self.loss_fn = evidence_loss(sparse=False)
    #             self.loss_fn = tf.keras.losses.categorical_crossentropy
                self.evidence = relu_evidence
    #             self.evidence = tf.compat.v1.distributions.Dirichlet
                self.temperature = 10
                self.lmb = 0.005
            def build(self, input_shape):
            # tf.print("inputShape",input_shape)
                self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])
            
            def get_config(self):
                config = super().get_config().copy()
                config.update({
                    'num_outputs': self.num_outputs,
                    'name': self.name
                })
                return config

            def call(self, inputs, labels,learning_rate=1):
                outputs = tf.matmul(inputs,self.kernel)
                softmax = tf.nn.softmax(outputs)
                evidence = softplus_evidence(outputs)
                alpha = evidence + 1
                
                ## convert labels to logits
                loss = tf.reduce_mean(self.loss_fn(labels, outputs))
                    
                u = self.num_outputs / tf.reduce_sum(alpha, axis=1, keepdims=True) #uncertainty
                prob = alpha/tf.reduce_sum(alpha, 1, keepdims=True) 
                l2_loss = tf.nn.l2_loss(self.weights) * self.lmb
                total_loss = loss + l2_loss
                self.add_loss(total_loss)
                pred = tf.argmax(outputs,1)
                truth = tf.argmax(labels,1)
                match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
                total_evidence = tf.reduce_sum(evidence,1, keepdims=True)
                mean_avg = tf.reduce_mean(total_evidence)
                mean_succ = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*match) / tf.reduce_sum(match+1e-20)
                mean_fail = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) )

                self.add_metric(evidence, name=self.name+"_evidence")
                # self.add_metric(u, name=self.name+"_uncertainty")
                self.add_metric(mean_avg, name=self.name+"_mean_ev_avg")
                self.add_metric(mean_succ, name=self.name+"_mean_ev_succ")
                self.add_metric(mean_fail, name=self.name+"_mean_ev_fail")
                
                return softmax


    class LogisticEndpoint(BranchEndpoint):
        def __init__(self, name=None, **kwargs):
            super(branch.LogisticEndpoint, self).__init__(name=name)
            self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            self.accuracy_fn = keras.metrics.BinaryAccuracy()

        def call(self, targets, logits, sample_weights=None):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            loss = self.loss_fn(targets, logits, sample_weights)
            self.add_loss(loss)

            # Log accuracy as a metric and add it
            # to the layer using `self.add_metric()`.
            acc = self.accuracy_fn(targets, logits, sample_weights)
            self.add_metric(acc, name="accuracy")

            # Return the inference-time prediction tensor (for `.predict()`).
            return tf.nn.softmax(logits)         
                

    ''' Third version, doesn't have its own loss, so it just provides the evidence as a metric
        Doesn't apply softmax, as the custom loss does that itself.
    '''
    class CrossEntropyEndpoint(BranchEndpoint):
        def __init__(self, num_outputs, name=None, **kwargs):
            super(branch.CrossEntropyEndpoint, self).__init__(name=name, num_outputs=num_outputs)
            self.num_outputs = num_outputs
#             self.kl = tf.keras.losses.KLDivergence()
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
#             self.loss_fn = tf.keras.losses.categorical_crossentropy
            self.evidence = softplus_evidence
#             self.evidence = tf.compat.v1.distributions.Dirichlet
            self.temperature = 10
            self.lmb = 0.005
            self.branch_exit = True

        def build(self, input_shape):
            # tf.print("inputShape",input_shape)
            self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])
        
        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'num_outputs': self.num_outputs,
                'name': self.name
            })
            return config

        def call(self, inputs, labels,learning_rate=1):
            outputs = tf.matmul(inputs,self.kernel)
            softmax = tf.nn.softmax(outputs)
            evidence = self.evidence (outputs)
            alpha = evidence + 1
            u = self.num_outputs / tf.reduce_sum(alpha, axis=1, keepdims=True) #uncertainty
          
            # prob = alpha/tf.reduce_sum(alpha, 1, keepdims=True) 
            pred = tf.argmax(outputs,1)
            truth = tf.argmax(labels,1)
            match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
            # total_evidence = tf.reduce_sum(evidence,1, keepdims=True)
            mean_succ = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*match) / tf.reduce_sum(match+1e-20)
            mean_fail = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) )
            
            self.add_metric(evidence, name=self.name+"_evidence",aggregation='mean')
            self.add_metric(mean_succ, name=self.name+"_mean_ev_succ",aggregation='mean')
            self.add_metric(mean_fail, name=self.name+"_mean_ev_fail",aggregation='mean')
            
            return outputs




    class SelfDistilEndpoint(BranchEndpoint):
        def __init__(self, num_outputs, loss_coef=.3, temperature=10, name=None, **kwargs):
            super(branch.SelfDistilEndpoint, self).__init__(num_outputs=num_outputs, name=name)
            self.num_outputs = num_outputs
            self.loss_coef = loss_coef
            self.temperature = temperature 
            self.distillation_loss_fn=keras.losses.KLDivergence()

        def build(self, input_shape):
            tf.print("inputShape",input_shape)
            self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])
        
        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'name': self.name
            })
            return config

        def call(self, inputs, teaching_distill=None):
            ''' do the normal kernel operations, then compare the difference between the teacher and this.
            '''
            outputs = tf.matmul(inputs,self.kernel)
            outputs = tf.nn.softmax(outputs)
            # tf.print("outputs",outputs)
            # tf.print("teaching",teaching_distill)
            if teaching_distill is not None:
                distil_loss = self.distillation_loss_fn(outputs/self.temperature, teaching_distill/self.temperature)
                distil_loss = distil_loss * self.loss_coef
                # print("KL_LOSS", kl_loss)
                # self.add_loss(kl_loss)
                self.add_loss(distil_loss)
                self.add_metric(distil_loss, aggregation='mean',name=self.name+"_distil") # metric so this loss value can be monitored.
            return outputs
  




