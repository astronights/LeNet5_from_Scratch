# COMP3314: Machine Learning Assignment 2

## LeNet5 Neural Network

### Install

This project requires **Python** and the following libraries:

- [Numpy](http://numpy.org)
- [Pandas](http://pandas.pydata.org)

### Code

The relevant files are provided in the 3035345306_A2 directory. 

### Files Needed

The dataset and utils directories are required to be in the same location as the A2.py script. 
Empty directories called layer and model respectively are to be created in the same location as A2.py script to store the layer weights and the trained models.

### Run

The project can be run using the command

```
python A2.py
```

This will run the LeNet5 Classifier for 20 epochs to train on mini-batches, printing out the training and test error after each mini-batch and finally running on the training data to provide the error rate. 