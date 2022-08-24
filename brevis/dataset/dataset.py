# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import print_tensor



class prepare:
    def augment_images(image, label,input_size=None, channel_first = False):
            # Normalize images to have a mean of 0 and standard deviation of 1
            # image = tf.image.per_image_standardization(image)
            # Resize images from 32x32 to 277x277
            image = tf.image.resize(image,input_size)
            if channel_first:
                image = tf.transpose(image, [2, 0, 1])
            
            return image, label
    
    
    #dataset for knowledge distillation, includes a second input of labels to input "targets"
    def dataset_distil(dataset,batch_size=32, validation_size = 0, shuffle_size = 0, input_size=None,channel_first= False):
        (train_images, train_labels), (test_images, test_labels) = dataset

        #hack to get around the limitation of providing additional parameters to the map function for the datasets below 
        def augment_images(image, label,input_size=input_size, channel_first = channel_first):
            return prepare.augment_images(image, label, input_size, channel_first)
        
        validation_images, validation_labels = train_images[:validation_size], train_labels[:validation_size] #get the first 5k training samples as validation set
        train_images, train_labels = train_images[validation_size:], train_labels[validation_size:] # now remove the validation set from the training set.
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
        
        if input_size is not None:
            train_ds = (train_ds.map(augment_images))
            validation_ds = (validation_ds.map(augment_images))
            test_ds = (test_ds.map(augment_images))
        
        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))

        target = tf.data.Dataset.from_tensor_slices((train_labels))
        train_ds = tf.data.Dataset.zip((train_ds,target))

        v_target = tf.data.Dataset.from_tensor_slices((validation_labels))
        validation_ds = tf.data.Dataset.zip((validation_ds,v_target))

        t_target = tf.data.Dataset.from_tensor_slices((test_labels))
        test_ds = tf.data.Dataset.zip((test_ds,t_target))

        
        print("trainSize {}".format(train_ds_size))
        print("testSize {}".format(test_ds_size))
        train_ds = (train_ds
                        # .map(augment_images)
                        .shuffle(buffer_size=tf.cast(shuffle_size,'int64'))
                        .batch(batch_size=batch_size, drop_remainder=True))

        test_ds = (test_ds
                        # .map(augment_images)
                        #   .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
                        # .map(augment_images)
                        #   .shuffle(buffer_size=validation_ds_size)
                        .batch(batch_size=batch_size, drop_remainder=True))


        print(train_ds)

        print(test_ds)
        return (train_ds, test_ds, validation_ds)


    def dataset(dataset,batch_size=32, validation_size = 0, shuffle_size = 0, input_size=None, channel_first = False, include_targets=False, categorical = True,num_outputs=10,reshuffle=False):
        ''' build the dataset
            Arguments: 
                dataset: the dataset to be used
                batch_size: the batch size
                validation_size: the size of the validation set
                shuffle_size: the size of the shuffle buffer
                input_size: the size of the input image
                channel_first: whether the input image is channel first or not
                include_targets: whether to include the targets as a second input
                categorical: whether the targets are categorical or not
                num_outputs: the number of outputs for the categorical targets                            
        '''
        (train_images, train_labels), (test_images, test_labels) = dataset
        
        if categorical:
            train_labels = tf.keras.utils.to_categorical(train_labels,num_outputs)
            test_labels = tf.keras.utils.to_categorical(test_labels,num_outputs)


        #hack to get around the limitation of providing additional parameters to the map function for the datasets below 
        def augment_images(image, label,input_size=input_size, channel_first= channel_first):
            # if channel_first:
                #swap the channels around.
                # image = tf.transpose(image, [2, 0, 1])
                # print(image)
            return prepare.augment_images(image, label, input_size, channel_first)
        
        validation_images, validation_labels = train_images[:validation_size], train_labels[:validation_size] #get the first 5k training samples as validation set
        train_images, train_labels = train_images[validation_size:], train_labels[validation_size:] # now remove the validation set from the training set.
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))


        
        
        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))
        if input_size is not None:
            print("augment Dataset")
            train_ds = (train_ds.map(augment_images))
            validation_ds = (validation_ds.map(augment_images))
            test_ds = (test_ds.map(augment_images))

        print("targetsis :", include_targets)
        #if include_targets is flagged, add an additional input with the label, this is used by custom loss layers that need a separate label source in the inputs to process.
        if include_targets == True:
            print("adding targets to inputs")
            target = tf.data.Dataset.from_tensor_slices((train_labels))
            train_ds = tf.data.Dataset.zip((train_ds,target))

            v_target = tf.data.Dataset.from_tensor_slices((validation_labels))
            validation_ds = tf.data.Dataset.zip((validation_ds,v_target))

            t_target = tf.data.Dataset.from_tensor_slices((test_labels))
            test_ds = tf.data.Dataset.zip((test_ds,t_target))

        print("trainSize {}".format(train_ds_size))
        print("testSize {}".format(test_ds_size))
        
        train_ds = (train_ds
                        # .map(augment_images)
                        .shuffle(buffer_size=shuffle_size,seed=42,reshuffle_each_iteration=reshuffle)
                        .batch(batch_size=batch_size, drop_remainder=False))

        test_ds = (test_ds
                        # .map(augment_images)
                        #   .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=batch_size, drop_remainder=False))

        validation_ds = (validation_ds
                        # .map(augment_images)
                        #   .shuffle(buffer_size=validation_ds_size)
                        .batch(batch_size=batch_size, drop_remainder=False))

        return (train_ds, test_ds, validation_ds)

    def test_set(dataset,batch_size=32, input_size=(), channel_first = False, include_targets=False, categorical = True, num_outputs=10):
        ''' quick function to get a test set for the model.'''
        (train_images, train_labels), (test_images, test_labels) = dataset
        if categorical:
            test_labels = tf.keras.utils.to_categorical(test_labels,num_outputs)
        #hack to get around the limitation of providing additional parameters to the map function for the datasets below 
        def augment_images(image, label,input_size=input_size, channel_first= channel_first):
            return prepare.augment_images(image, label, input_size, channel_first)
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_ds_size = len(list(test_ds))
        test_ds = (test_ds.map(augment_images))
        print("targetsis :", include_targets)
        #if include_targets is flagged, add an additional input with the label, this is used by custom loss layers that need a separate label source in the inputs to process.
        if include_targets == True:
            print("adding targets to inputs")
            t_target = tf.data.Dataset.from_tensor_slices((test_labels))
            test_ds = tf.data.Dataset.zip((test_ds,t_target))

        test_ds = (test_ds.batch(batch_size=batch_size, drop_remainder=True))
        return test_ds

    def outlierDataset(dataset,batch_size=32, validation_size = 0, shuffle_size = 0, input_size=(), channel_first = False, include_targets=False, categorical = True,num_outputs=10):
            ''' build the dataset with outliers as well
                build the dataset provided, but add in outliers from another datatset. 
                the outliers are not provided labels...
                Arguments: 
                    dataset: the dataset to be used
                    batch_size: the batch size
                    validation_size: the size of the validation set
                    shuffle_size: the size of the shuffle buffer
                    input_size: the size of the input image
                    channel_first: whether the input image is channel first or not
                    include_targets: whether to include the targets as a second input
                    categorical: whether the targets are categorical or not
                    num_outputs: the number of outputs for the categorical targets                            
            '''
            (train_images, train_labels), (test_images, test_labels) = dataset
            
            if categorical:
                train_labels = tf.keras.utils.to_categorical(train_labels,num_outputs)
                test_labels = tf.keras.utils.to_categorical(test_labels,num_outputs)


            #hack to get around the limitation of providing additional parameters to the map function for the datasets below 
            def augment_images(image, label,input_size=input_size, channel_first= channel_first):
                # if channel_first:
                    #swap the channels around.
                    # image = tf.transpose(image, [2, 0, 1])
                    # print(image)
                return prepare.augment_images(image, label, input_size, channel_first)
            
            validation_images, validation_labels = train_images[:validation_size], train_labels[:validation_size] #get the first 5k training samples as validation set
            train_images, train_labels = train_images[validation_size:], train_labels[validation_size:] # now remove the validation set from the training set.
            train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))


            
            
            train_ds_size = len(list(train_ds))
            test_ds_size = len(list(test_ds))
            validation_ds_size = len(list(validation_ds))

            print("augment Dataset")
            train_ds = (train_ds.map(augment_images))
            validation_ds = (validation_ds.map(augment_images))
            test_ds = (test_ds.map(augment_images))

            print("targetsis :", include_targets)
            #if include_targets is flagged, add an additional input with the label, this is used by custom loss layers that need a separate label source in the inputs to process.
            if include_targets == True:
                print("adding targets to inputs")
                target = tf.data.Dataset.from_tensor_slices((train_labels))
                train_ds = tf.data.Dataset.zip((train_ds,target))

                v_target = tf.data.Dataset.from_tensor_slices((validation_labels))
                validation_ds = tf.data.Dataset.zip((validation_ds,v_target))

                t_target = tf.data.Dataset.from_tensor_slices((test_labels))
                test_ds = tf.data.Dataset.zip((test_ds,t_target))

            print("trainSize {}".format(train_ds_size))
            print("testSize {}".format(test_ds_size))
            
            train_ds = (train_ds
                            # .map(augment_images)
                            .shuffle(buffer_size=train_ds_size,seed=42,reshuffle_each_iteration=False)
                            .batch(batch_size=batch_size, drop_remainder=True))

            test_ds = (test_ds
                            # .map(augment_images)
                            #   .shuffle(buffer_size=train_ds_size)
                            .batch(batch_size=batch_size, drop_remainder=True))

            validation_ds = (validation_ds
                            # .map(augment_images)
                            #   .shuffle(buffer_size=validation_ds_size)
                            .batch(batch_size=batch_size, drop_remainder=True))

            return (train_ds, test_ds, validation_ds)


    def dataset_normalized(dataset,batch_size=32, validation_size = 0, shuffle_size = 0, input_size=(), channel_first = False, include_targets=False, categorical = True):
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_images = train_images.reshape(50000, 32,32,3).astype("float32") / 255
        test_images = test_images.reshape(10000, 32,32,3).astype("float32") / 255
        print("labels",train_labels[0])
        if categorical:
            train_labels = tf.keras.utils.to_categorical(train_labels,10)
            test_labels = tf.keras.utils.to_categorical(test_labels,10)
        print(train_labels[0])

        #hack to get around the limitation of providing additional parameters to the map function for the datasets below 
        def augment_images(image, label,input_size=input_size, channel_first= channel_first):
            # if channel_first:
                #swap the channels around.
                # image = tf.transpose(image, [2, 0, 1])
                # print(image)
            return prepare.augment_images(image, label, input_size, channel_first)
        
        validation_images, validation_labels = train_images[:validation_size], train_labels[:validation_size] #get the first 5k training samples as validation set
        train_images, train_labels = train_images[validation_size:], train_labels[validation_size:] # now remove the validation set from the training set.
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        # train_ds.tak
        
        
        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))

        print("augment Dataset")
        # train_ds = (train_ds.map(augment_images))
        # validation_ds = (validation_ds.map(augment_images))
        # test_ds = (test_ds.map(augment_images))

        print("targetsis :", include_targets)
        #if include_targets is flagged, add an additional input with the label, this is used by custom loss layers that need a separate label source in the inputs to process.
        if include_targets == True:
            print("adding targets to inputs")
            target = tf.data.Dataset.from_tensor_slices((train_labels))
            train_ds = tf.data.Dataset.zip((train_ds,target))

            v_target = tf.data.Dataset.from_tensor_slices((validation_labels))
            validation_ds = tf.data.Dataset.zip((validation_ds,v_target))

            t_target = tf.data.Dataset.from_tensor_slices((test_labels))
            test_ds = tf.data.Dataset.zip((test_ds,t_target))

        print("trainSize {}".format(train_ds_size))
        print("testSize {}".format(test_ds_size))
        
        train_ds = (train_ds
                        # .map(augment_images)
                        .shuffle(buffer_size=tf.cast(shuffle_size,'int64'))
                        .batch(batch_size=batch_size, drop_remainder=True))

        test_ds = (test_ds
                        # .map(augment_images)
                        #   .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=1, drop_remainder=True))

        validation_ds = (validation_ds
                        # .map(augment_images)
                        #   .shuffle(buffer_size=validation_ds_size)
                        .batch(batch_size=batch_size, drop_remainder=True))

        return (train_ds, test_ds, validation_ds)



    def prepareMnistDataset(dataset,batch_size=32, validation_size = 0, shuffle_size = 0, input_size=(), channel_first = False, include_targets=False, categorical = True):
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_images = train_images.reshape(60000, 784).astype("float32") / 255
        test_images = test_images.reshape(10000, 784).astype("float32") / 255
        if categorical:
            train_labels = tf.keras.utils.to_categorical(train_labels,10)
            test_labels = tf.keras.utils.to_categorical(test_labels,10)
            
        validation_images, validation_labels = train_images[:validation_size], train_labels[:validation_size] #get the first 5k training samples as validation set
        train_images, train_labels = train_images[validation_size:], train_labels[validation_size:] # now remove the validation set from the training set.
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
       
       
        print("augment Dataset")
        # train_ds = (train_ds.map(augment_images))
        # validation_ds = (validation_ds.map(augment_images))
        # test_ds = (test_ds.map(augment_images))

        print("targetsis :", include_targets)
        #if include_targets is flagged, add an additional input with the label, this is used by custom loss layers that need a separate label source in the inputs to process.
        if include_targets == True:
            print("adding targets to inputs")
            target = tf.data.Dataset.from_tensor_slices((train_labels))
            train_ds = tf.data.Dataset.zip((train_ds,target))

            v_target = tf.data.Dataset.from_tensor_slices((validation_labels))
            validation_ds = tf.data.Dataset.zip((validation_ds,v_target))

            t_target = tf.data.Dataset.from_tensor_slices((test_labels))
            test_ds = tf.data.Dataset.zip((test_ds,t_target))

        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))

        train_ds = (train_ds
            # .map(prepare.augment_images)
            .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=batch_size, drop_remainder=True))
        test_ds = (test_ds
            # .map(prepare.augment_images)
            .shuffle(buffer_size=int(test_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
            # .map(prepare.augment_images)
            .shuffle(buffer_size=int(validation_ds_size))
            .batch(batch_size=batch_size, drop_remainder=True))
        return train_ds, test_ds, validation_ds

    def prepareAlexNetDataset_alt(dataset,batch_size=32):
        import csv
        (train_images, train_labels), (test_images, test_labels) = dataset
        with open('results/altTrain_labels2.csv', newline='') as f:
            reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
            alt_trainLabels = list(reader)
        with open('results/altTest_labels2.csv', newline='') as f:
            reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
            alt_testLabels = list(reader)

        altTraining = tf.data.Dataset.from_tensor_slices((train_images,alt_trainLabels))

        validation_images, validation_labels = train_images[:5000], alt_trainLabels[:5000]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, alt_trainLabels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, alt_testLabels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))
        train_ds = (train_ds
            .map(prepare.augment_images)
            .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=batch_size, drop_remainder=True))
        test_ds = (test_ds
            .map(prepare.augment_images)
            .shuffle(buffer_size=int(test_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
            .map(prepare.augment_images)
            .shuffle(buffer_size=int(validation_ds_size))
            .batch(batch_size=batch_size, drop_remainder=True))
        return train_ds, test_ds, validation_ds


    def prepareAlexNetDataset( dataset, batch_size =32):
        (train_images, train_labels), (test_images, test_labels) = dataset

        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))


        print("trainSize {}".format(train_ds_size))
        print("testSize {}".format(test_ds_size))

        train_ds = (train_ds
            .map(prepare.augment_images)
            .shuffle(buffer_size=int(train_ds_size))
            # .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=batch_size, drop_remainder=True))
        test_ds = (test_ds
            .map(prepare.augment_images)
            # .shuffle(buffer_size=int(train_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
            .map(prepare.augment_images)
            # .shuffle(buffer_size=int(train_ds_size))
            .batch(batch_size=batch_size, drop_remainder=True))
        return train_ds, test_ds, validation_ds

    def prepareAlexNetDataset_old( batchsize=32):
        # tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

        CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # validation_images, validation_labels = train_images[:5000], alt_trainLabels[:5000]
        # train_ds = tf.data.Dataset.from_tensor_slices((train_images, alt_trainLabels))
        # test_ds = tf.data.Dataset.from_tensor_slices((test_images, alt_testLabels))

        ###normal method
        validation_images, validation_labels = train_images[:5000], train_labels[:5000] #get the first 5k training samples as validation set
        train_images, train_labels = train_images[5000:], train_labels[5000:] # now remove the validation set from the training set.
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        def augment_images(image, label):
            # Normalize images to have a mean of 0 and standard deviation of 1
            # image = tf.image.per_image_standardization(image)
            # Resize images from 32x32 to 277x277
            image = tf.image.resize(image, (227,227))
            return image, label
        

        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))

        print("trainSize {}".format(train_ds_size))
        print("testSize {}".format(test_ds_size))

        train_ds = (train_ds
                        .map(prepare.augment_images)
                        .shuffle(buffer_size=tf.cast(train_ds_size/2,'int64'))
                        .batch(batch_size=batchsize, drop_remainder=True))

        test_ds = (test_ds
                        .map(prepare.augment_images)
                        #   .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=batchsize, drop_remainder=True))

        validation_ds = (validation_ds
                        .map(prepare.augment_images)
                        #   .shuffle(buffer_size=validation_ds_size)
                        .batch(batch_size=batchsize, drop_remainder=True))

        print("testSize2 {}".format(len(list(test_ds))))
        return train_ds, test_ds, validation_ds

    def prepareInceptionDataset( dataset, batch_size=32):
        (train_images, train_labels), (test_images, test_labels) = dataset

        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))


        print("trainSize {}".format(train_ds_size))
        print("testSize {}".format(test_ds_size))

        train_ds = (train_ds
            .map(prepare.augment_images)
            .shuffle(buffer_size=int(train_ds_size))
            # .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=batch_size, drop_remainder=True))
        test_ds = (test_ds
            .map(prepare.augment_images)
            # .shuffle(buffer_size=int(train_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
            .map(prepare.augment_images)
            # .shuffle(buffer_size=int(train_ds_size))
            .batch(batch_size=batch_size, drop_remainder=True))
        return train_ds, test_ds, validation_ds


    def datasetStats( dataset):
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset, batch_size = 1)
        # can still use tf.data.Dataset for mnist and numpy models
        # I found a bug where the model couldn't run on the input unless the dataset is batched. so make sure to batch it.
        val_size = int(len(train_images) * 0.2)  #atm I'm making validation sets that are a fifth of the test set. 
        x_val = train_images[-val_size:]
        y_val = train_labels[-val_size:]
        train_images = train_images[:-val_size]
        train_labels = train_labels[:-val_size]
        
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_ds = train_ds.shuffle(buffer_size=1024).batch(64)
        # Reserve 10,000 samples for validation
        
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_ds = test_ds.batch(64)

        # Prepare the validation dataset
        validation_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        validation_ds = validation_ds.batch(64)
        ds_iter = iter(test_ds)
        results = []
        for i in range(len(list(validation_ds))):
            one = ds_iter.get_next()
            results.append(one[1].numpy())
        # print(one)
            # print(one[1])
        results = np.vstack(results)
        unique, counts = np.unique(results, return_counts=True)        
        print(dict(zip(unique, counts)))
        return