# Shopee Code League 2020: Product Detection

## Introduction

My first attempt at Machine Learning. This is a submission for the Product Detection competition hosted by Shopee Code League 2020 on Kaggle. 

## How it works

This model was mainly referenced from a Keras blog guide. Firstly, bottleneck features are extracted using a [ImageNet](http://www.image-net.org) pre-trained [VGG16](https://neurohive.io/en/popular-networks/vgg16/) model. Then, a small classifier on top is trained to output the categories of the products, with the bottleneck features as input.

## Getting Started

### Format

Test images should be in the [test directory](./test/test), training images (to be split into training and validation images) should be in the [train directory](./train/train)

Images in the test and training directories should be in the following format:

```
/test
    /00
        image01.jpg
        image02.jpg
    /01
        image03.jpg
    /02
    ...
```
where image01.jpg and image02.jpg belong to category 00, and so on for 42 categories.

### Files to run

Run manage_files.py to split into training and validation files, then train_model to train the model and finally test_model to produce the predictions for the test directory in a results.csv file.

## Resources and Credits

### Theoretical
* Shopee's workshop on Product Detection
* [3Blue1Brown's amazing and beginner-friendly introduction to Neural Networks and Deep Learning](https://www.3blue1brown.com/neural-networks)
* [Stanford's course on Convolution Neural Networks](https://cs231n.github.io/convolutional-networks/)
* [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way by Sumit Saha](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
* [Explained Visually's intuitive visualisation](https://setosa.io/ev/image-kernels/) of the effect of image kernels to explain why we need convolution to filter images and extract features
* [Understanding different Loss Functions for Neural Networks by Shiva Verma](https://towardsdatascience.com/understanding-different-loss-functions-for-neural-networks-dd1ed0274718)



### Technical
* [Keras blog guide to building image classifiers](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 
* [Tensorflow and Keras documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
* [Pandas documentation](https://pandas.pydata.org)

# Final Thoughts
This project only touches the tip of the iceberg in the very convoluted subject of Machine and Deep Learning. 

Current Machine Learning libraries on Python such as [Tensorflow](https://www.tensorflow.org) (with Keras), [Pytorch](https://pytorch.org) and [scikit-learn](https://scikit-learn.org), in conjunction with data science libraries like [Numpy](https://numpy.org), [Pandas](https://pandas.pydata.org) and [Matplotlib](https://matplotlib.org), has made Deep Learning much more accessible to new beginners like myself. 

However, this mathematical and foundational gap must be filled in order to have better knowledge in tuning hyperparameters to create better models.