'''
Actual testing of model.
Test images are preprocessed, before pushing them through the 
prediction algorithm.
Results are then parsed and processed into the predicted categories
and saved in a csv file.
'''
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
import os

model_path = 'top_model.h5'
test_data_dir = 'test\\test'
img_width, img_height = 300, 300
batch_size = 32
nb_test_samples = 12186   

def string_format(integer):
    '''Converting categories expressed as integer indexes in the Numpy array of results into the string format required.'''
    if (integer >= 0) and (integer < 10):
        return '0' + str(integer)
    elif (integer >= 10) and (integer < 42):
        return str(integer)
    else:
        return None

def initialise_model(model_path):
    '''
    Initialise and load the classifier created and trained under train_model.py, 
    on top of the ImageNet pre-trained VGG16 base model.
    '''
    base_model = applications.VGG16(include_top=False, weights='imagenet')
    top_model = models.load_model(model_path)
    top_model.summary()

    model = Sequential()
    model.add(base_model)
    model.add(top_model)

    return model

def preprocessing(test_data_dir):
    '''
    Given the test directory, return the Numpy arrays of all test images, as well as their
    file labels.
    Preprocessing includes resizing images and rescaling of pixel values (0-255) into
    floats (0-1).
    '''
    #List of all test filenames
    names = os.listdir(test_data_dir)
    assert(names.size == nb_test_samples)

    #Getting Numpy arrays of test files' pixel values
    test_arrays = []
    for f in names:
        img_path = os.path.join(test_data_dir,f)
        img = load_img(img_path, target_size=(img_height, img_width)) #Resize image
        arr = img_to_array(img) #Get array from image

        rescaled_arr = arr*(1./255) #Rescale array

        assert((rescaled_arr.all() <= 1) and (rescaled_arr.all() >= 0))

        test_arrays.append(rescaled_arr)
    assert(len(test_arrays) == nb_test_samples)  
    test_arrays = np.array(test_arrays)

    return names, test_arrays

def predict_and_output(model, test_arrays, names):
    '''
    Actual prediction using the model initialised and converted into the correct format
    in a CSV file.
    '''
    #Using top model classifier with trained weights to get correct predictions
    results = model.predict(
        test_arrays,
        steps=(nb_test_samples//batch_size),
        verbose=1)

    #For every image entry, identify the category that has the highest probability.
    categories = []
    for row in results:
        row_as_list = list(row)
        max_value = max(row_as_list)
        max_value_index = row_as_list.index(max_value)
        max_value_index_string = string_format(max_value_index)
        categories.append(max_value_index_string)

    #Tag image entries with their respective filenames and converting to CSV file
    filename_category = pd.DataFrame({'filename':names,'category':categories})
    filename_category.to_csv('results.csv',header=['filename','category'],index=False)

model = initialise_model(model_path)
names, test_arrays = preprocessing(test_data_dir)
predict_and_output(model, test_arrays, names)