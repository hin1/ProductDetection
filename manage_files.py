'''
Preprocessing files beforehand. 
As training and validation data must be divisible by batch size, the training and validation
directories had to be trimmed to have 105000 images, with a validation split of 0.2, meaning
84000 training images and 21000 validation images. 
'''
import time, datetime, os
from random import sample
from shutil import move

train_data_dir = 'train\\train'
validation_data_dir = 'validation'

total_samples = 105398
nb_train_samples = 84336 
nb_validation_samples = 21062 

total_nb_to_trim = 398
train_nb_to_trim = nb_train_samples - 84000
validation_nb_to_trim = nb_validation_samples - 21000

nb_categories = 42
validation_split = 0.2


def move_to_validation():
    '''
    Split training data according to a
    given validation split value, to move to the validation data directory.
    ''' 
    for folder in os.listdir(train_data_dir):
        folder_path = os.path.join(train_data_dir,folder)
        folder_list = os.listdir(folder_path)
        folder_num = len(folder_list)
        num_to_move = int(folder_num * validation_split)

        os.makedirs(os.path.join(validation_data_dir,folder))

        for f in folder_list[:num_to_move]:
            og_path_name = os.path.join(folder_path,f)
            new_path_name = os.path.join(validation_data_dir,folder,f)
            move(og_path_name,new_path_name)

def get_nb_images_per_category(data_dir):
    '''
    Find number of images in each category for a given directory with the
    following directory structure:
    
    /data_dir
        /category01
            <images in category 01>
        /category02
            <images in category 02>
        ...
    '''
    category_image_numbers = []
    for folder in os.listdir(data_dir):
        category_num = len(os.listdir(os.path.join(data_dir,folder)))
        category_image_numbers.append(category_num)

        assert(len(category_image_numbers) == nb_categories)

    return category_image_numbers

def get_nb_trim(og_cat_list,num_to_trim):
    '''
    Find the proportion and number of images to trim in each category,
    given the number of images in each category, and the total number of
    images to trim
    '''
    trim_list = []

    #Find proportion to trim in each 
    sum_trim_num = 0
    for num in og_cat_list:
        pc_to_trim = num/total_samples
        category_nb_to_trim = int(pc_to_trim * total_nb_to_trim)
        trim_list.append(category_nb_to_trim)
        sum_trim_num += category_nb_to_trim

    #Manually add or subtract images to trim due to rounding error
    category_sorted_by_num = sorted(range(len(og_cat_list)), key=lambda ind: og_cat_list[ind])
    extra_trim_from_rounding_err = num_to_trim - sum_trim_num

    #If rounding error gives too many images to trim 
    if (extra_trim_from_rounding_err > 0):
        for i in range(extra_trim_from_rounding_err):
            trim_list[category_sorted_by_num[-1]] += 1
            category_sorted_by_num.pop(-1)

    #Or if rounding error gives too little images to trim
    elif (extra_trim_from_rounding_err < 0):
        extra_trim_from_rounding_err = abs(extra_trim_from_rounding_err)
        for i in range(extra_trim_from_rounding_err):
            trim_nb = trim_list[category_sorted_by_num[0]]
            if trim_nb == 0:
                largest_category = og_cat_list.index(max(og_cat_list))
                trim_list[largest_category] -= 1
                og_cat_list[largest_category] = -1
            else:
                trim_nb -= 1
                category_sorted_by_num.pop(0)

    return trim_list

def trim(trim_dir, trim_list):
    '''
    Actual trimming of files from a given directory as well as a list of the number
    of images to trim for each category (subdirectory)
    '''
    category = 0
    for folder in os.listdir(trim_dir):
        folder_path = os.path.join(trim_dir,folder)
        folder_list = os.listdir(folder_path)
        trim_nb = trim_list[category]
        for f in sample(folder_list,trim_nb):
            os.remove(os.path.join(folder_path, f))
        print(str(trim_nb) + ' images removed in Category ' + folder)
        category += 1


#Split training data into training and validation data
move_to_validation()

#Find out the number of images in each category (subdirectory)
train_category_nb = get_nb_images_per_category(train_data_dir)
validation_category_nb = get_nb_images_per_category(validation_data_dir)

#Find number of images to trim in each category
trim_train_category_nb = get_nb_trim(train_category_nb,train_nb_to_trim)
trim_validation_category_nb = get_nb_trim(validation_category_nb,validation_nb_to_trim)

#Trim the images
trim(train_data_dir,trim_train_category_nb)
trim(validation_data_dir,trim_validation_category_nb)

#Check that the correct number of images were trimmed
trimmed_train_num = get_nb_images_per_category(train_data_dir)
trimmed_validation_num = get_nb_images_per_category(validation_data_dir)
    
