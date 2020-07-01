'''
Training the actual model.
Uses a ImageNet pre-trained VGG16 base model to do feature extraction.
Following that, a top model classifier is trained with the features extracted
and given training data.
'''
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
import time, datetime, os

model_path = 'top_model.h5'
train_data_dir = 'train\\train'
validation_data_dir = 'validation'

total_samples = 105000
nb_train_samples = 84000 
nb_validation_samples = 21000 

img_width, img_height = 300, 300
epochs = 50
batch_size = 30

def save_bottleneck_features():
    '''
    Use the pre-trained VGG16 model's convolutional layers to run predictions on training 
    and validation data without providing it with the categories.
    The results are saved into external numpy files to be processed into the top model classifier.
    '''
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network without the fully-connected layers
    model = applications.VGG16(include_top=False, weights='imagenet')

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict(train_generator, steps=(nb_train_samples//batch_size), verbose=1)
    
    np.save(open('train.npy', 'wb'), bottleneck_features_train)

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict(validation_generator, steps=(nb_validation_samples//batch_size), verbose=1)
    np.save(open('validation.npy', 'wb'), bottleneck_features_validation)

def initialise_labels(nb_samples,sample_dir):
    '''
    Create a numpy array of size equivalent to the number of training or validation images,
    with each entry containing the category it belongs to.
    This is for matching the predictions made in save_bottleneck_features().
    '''
    labels = 43 * np.ones((nb_samples))
    category = 0
    index = 0
    print(labels.shape)

    sum = 0
    for folder in os.listdir(sample_dir):
        num_category_images = len(os.listdir(os.path.join(sample_dir,folder)))
        labels[index:index+num_category_images] = category
        category += 1
        index += num_category_images
        sum += num_category_images
    assert(sum == nb_samples)
        
    return labels

def train_top_model():
    '''
    Creates a top model or classifier that contains the fully connected layers to train the weights 
    with the training and validation data and their respective categories.
    Saves the model in an external h5 file for testing in test_model.py.
    '''
    train_data = np.load('train.npy')
    validation_data = np.load('validation.npy')
    
    #Classification of labels
    train_labels = initialise_labels(nb_train_samples,train_data_dir)
    train_unique, train_counts = np.unique(train_labels, return_counts=True)

    validation_labels = initialise_labels(nb_validation_samples,validation_data_dir)
    validation_unique, validation_counts = np.unique(validation_labels, return_counts=True)

    assert(np.all(train_labels) < 42)
    assert(np.all(validation_labels) < 42)
    
    #Check if the fully connected layers output the correct shape for sparse categories
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(42, activation='sigmoid'))

    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save(model_path)
    
start = time.time()

save_bottleneck_features()
#train_top_model()

end = time.time()
print("Time taken: " + str(datetime.timedelta(seconds=(end - start))) + " minutes")