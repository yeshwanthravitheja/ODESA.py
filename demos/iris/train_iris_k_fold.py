'''
Test ODESA on iris dataset converted to population coding using Bohte et al 2002 paper
Author: Yeshwanth Bethi
Email: Y.Bethi@westernsydney.edu.au
'''

#######################################################################
# The First part of the code is to generate a spiking dataset from IRIS dataset using population coding.
# The ODESA part follows this. 
#######################################################################
from sklearn.datasets import load_iris
from scipy.stats import norm
import numpy as np
import random


# load Iris dataset
iris = load_iris()


X = iris.data  # we only take the first two features.

# Seperate each feature
X_0 = X[:,0]
X_1 = X[:,1]
X_2 = X[:,2]
X_3 = X[:,3]

# Find the min and max of each feature
X_0_min = np.min(X_0)
X_1_min = np.min(X_1)
X_2_min = np.min(X_2)
X_3_min = np.min(X_3)

X_0_max = np.max(X_0)
X_1_max = np.max(X_1)
X_2_max = np.max(X_2)
X_3_max = np.max(X_3)

# Number of neurons per feature to encode its value
m = 5

#  Create the means and variances of the gaussian receptive fields. 
mu_0 = [X_0_min+((2*i-3)/2)*((X_0_max-X_0_min)/(m-2)) for i in range(1,m+1)]
mu_1 = [X_1_min+((2*i-3)/2)*((X_1_max-X_1_min)/(m-2)) for i in range(1,m+1)]
mu_2 = [X_2_min+((2*i-3)/2)*((X_2_max-X_2_min)/(m-2)) for i in range(1,m+1)]
mu_3 = [X_3_min+((2*i-3)/2)*((X_3_max-X_3_min)/(m-2)) for i in range(1,m+1)]

beta = 1.5

sigma_0 = (1/beta)*((X_0_max-X_0_min)/(m-2))
sigma_1 = (1/beta)*((X_1_max-X_1_min)/(m-2))
sigma_2 = (1/beta)*((X_2_max-X_2_min)/(m-2))
sigma_3 = (1/beta)*((X_3_max-X_3_min)/(m-2))

y = iris.target

dataset_len = X.shape[0]
num_features = X.shape[1]

mu = []
mu.append(mu_0)
mu.append(mu_1)
mu.append(mu_2)
mu.append(mu_3)

sig = []
sig.append(sigma_0)
sig.append(sigma_1)
sig.append(sigma_2)
sig.append(sigma_3)

dataset = []


for i in range(dataset_len):
    # Create a list of events per example in the dataset
    feature_timestamps = []
    X_i = X[i,:]
    for j in range(num_features):
        population = []
        for k in range(m):
            gaussian_value = norm.pdf(X_i[j],mu[j][k],sig[j])
            max_prob = norm.pdf(mu[j][k],mu[j][k],sig[j])
            gaussian_value /= max_prob
            
            if gaussian_value > 0.1:
                # Each event consists of (Index, timestamp). The Gaussian value is the time stamp. It is between 0-1 for each neuron index. 
                feature_timestamps.append((j*m+k,gaussian_value))
    
    dataset.append(sorted(feature_timestamps,key=lambda x: x[1]))


# np.random.seed(0)

# ###################################################################################################
###################################################################################################
# ODESA TRAINING
#################################################################################################
#####################################################################################################
import sys, os
sys.path.append(os.path.join(sys.path[0], '..','..'))

import numpy as np
from ODESA.FullyConnected import FullyConnected as HiddenLayer
from ODESA.Classifier import Classifier as OutputLayer
from ODESA.FCModel import FCModel as OdesaModel
from tqdm import tqdm



def evaluate(trained_model, input_stream,output_stream,testing_ids):
    correct_count = 0
    wrong_count = 0
    no_count = 0

    # Reset the timesurfaces. 
    trained_model.reset()

    # Reset global time
    t = 0

    # For each example in the test set. 
    for idx in testing_ids:
        example = input_stream[idx]
        num_events = len(example)
        label = output_stream[idx]
        t = t + 5
        for event_idx in range(num_events):
            event = example[event_idx]
            x = 0
            y = event[0]
            ts = t + event[1]
            
            if event_idx < num_events - 1:
                # Infer only does a forward pass without changing the weights and thresholds. 
                winners, output_winner, output_class = trained_model.infer((x,y,ts,1))
            else:
                # Infer only does a forward pass without changing the weights and thresholds. 
                winners, output_winner, output_class = trained_model.infer((x,y,ts,1))

                # Check the metrics. 
                if output_winner > -1:
                    if output_class == label:
                        correct_count += 1
                    else:
                        wrong_count += 1
                else:
                    no_count += 1


    correct_class_percent = correct_count/(correct_count+no_count+wrong_count)
    no_class_percent = no_count/(correct_count+no_count+wrong_count)
    wrong_class_percent = wrong_count/(correct_count+no_count+wrong_count)

    return correct_class_percent, wrong_class_percent, no_class_percent






def run_training(epochs,x_train,y_train,training_order,config):

    # The code is tested on the CPU. I've never got it to run faster than CPU on a GPU because its event-driven
    # device = torch.device('cpu')

    # Number of input dimensions
    np.random.seed(0) 
    dim = num_features*m
    # Flag to use accumulative time surfaces in all layers. 
    cumulative_time_surfaces = True

    # Different config options for each layer
    tau = config['tau']
    tau2 = config['tau2']
    eta = config['eta']
    eta2 = config['eta2']
    n_input_neurons = config['n_input_neurons']

    # Hyper Parameters for the first hidden layer
    # Input rows = 1 for 1D data 
    input_layer_context_rows = 1
    # Input columns is the dimension of input data 
    input_layer_context_cols = dim
    # Number of neurons in the layer
    input_layer_n_neurons = n_input_neurons
    # Learning rate for the weight update
    input_layer_eta = eta
    # Amount by which threshold is opened when a neuron is punished. 
    input_layer_threshold_open = 0.1
    # The timesurface time constant of the layer
    input_layer_tau = tau
    # The timesurface time constant of the next layer, this is used to trace the contribution of current layer in next layer's activity
    input_layer_trace_tau = tau2
    # The learning rate for the adaptive threshold update. Most of the times leaving it to 0.01 just works fine
    input_layer_thresh_eta = input_layer_eta
    # The flag for if the layer has to use a accumulative time surface or not
    input_layer_cumulative_ts = cumulative_time_surfaces



    # Instantiate the first hidden layer
    input_layer = HiddenLayer(input_layer_context_rows, 
                                input_layer_context_cols,
                                input_layer_n_neurons,
                                input_layer_eta,
                                input_layer_threshold_open,
                                input_layer_tau,
                                input_layer_trace_tau,
                                input_layer_thresh_eta,
                                cumulative_ts=input_layer_cumulative_ts)


    # Hyper parameters for constructing the final classifier output layer
    output_layer_context_rows = 1 
    # Context cols is going to be equal to the previous layer's number of neurons.
    output_layer_context_cols = input_layer_n_neurons
    #  Number of classes in the task
    output_layer_n_classes = 3
    # Number of neurons per a class group. Total number of neurons in the final layer = n_classes*n_neurons_per_class. 
    output_layer_n_neurons_per_class = [config['n_per_class'] for i in range(3)]
    # The learning rate for the layer weight update and other parameters just like any other layer
    output_layer_eta = eta2
    output_layer_threshold_open = 0.1
    output_layer_tau = tau2
    output_layer_thresh_eta = output_layer_eta

    output_layer_cumulative_ts = cumulative_time_surfaces



    # Instantiate the output layer
    output_layer = OutputLayer(output_layer_context_rows,
                                    output_layer_context_cols,
                                    output_layer_n_neurons_per_class,
                                    output_layer_n_classes,
                                    output_layer_eta,
                                    output_layer_tau,
                                    output_layer_threshold_open,
                                    thresh_eta = output_layer_thresh_eta,
                                    cumulative_ts=output_layer_cumulative_ts)
        
    # input_layer.initialize()
    # output_layer.initialize()
    # Initalize a Feast Model to have all the layers. 
    model = OdesaModel()
    # Add hidden layers. Copy the statement to add more layers to the network.  
    model.add_hidden_layer(input_layer)
    # Finally add the output layer. Every model has to have an output layer. 
    model.add_output_layer(output_layer)

    # Counters for classifcation metrics
    train_correct_scores = []
    train_wrong_scores = []
    train_no_scores = []
    pbar = tqdm(range(epochs))
    # Epochs is the number of epochs for which you want to go through the training data. 
    for epoch in pbar:
        # This resets the timesurfaces of all the layers.         
        model.reset()

        # Reset the metrics for the epoch
        correct_count = 0
        wrong_count = 0
        no_count = 0
        # Reset the global time.
        t = 0
        
        # For each example in the training set
        for idx in training_order:
            example = x_train[idx]
            # Find the number of events in the example to give the supervisory signal at the last event. 
            num_events = len(example)
            # find the class of the example. 
            label = y_train[idx]
            # Leave a reasonable amount of time between each example, so the timesurfaces can decay back to zero befor getting new inputs. 
            t = t + 5

            # For each event in the training example
            for event_idx in range(num_events):
                # Find the event
                event = example[event_idx]
                # The row of the event is always 0 for 1D datasets. 
                x = 0
                # The col of the event is the input index.
                y = event[0]
                # Because each example starts with time t=0 and end at almost t=1, we have a global clock which carries on the time. 
                ts = t + event[1]
                
                # If the event is not the last event, then the label is sent as -1(which means no class)
                if event_idx < num_events - 1:
                    # This passes the event through all the layers. and gives you a list of winner neurons in each layer for the given event. 
                    # And if the label is non negative, it does the weight and threshold adaptation according to super-feast algorithm. 
                    winners, output_winner, output_class = model.forward((x,y,ts,1),-1)
                
                # If the event is the last one in the example, you send the true label of the example. 
                else:
                    # Pass the label as non negative value for the last event. 
                    winners, output_winner, output_class = model.forward((x,y,ts,1),label)

                    # Note down the performance of the model. 
                    if output_winner > -1:
                        if output_class == label:
                            correct_count += 1
                        else:
                            wrong_count += 1
                    else:
                        no_count += 1


   
        current_correct_count = correct_count/(correct_count+no_count+wrong_count+1e-6)
        current_no_count = no_count/(correct_count+no_count+wrong_count+1e-6)
        current_wrong_count = wrong_count/(correct_count+no_count+wrong_count+1e-6)

        # Find the moving average of the metrics. 
        # correct_class_percent = 0.9*correct_class_percent+0.1*current_correct_count
        # wrong_class_percent = 0.9*wrong_class_percent+0.1*current_wrong_count
        # no_class_percent = 0.9*no_class_percent+0.1*current_no_count
        pbar.set_description("Correct Classification Percentage : {:.2f}".format(current_correct_count))

        # train_correct_scores.append(current_correct_count)
        # train_wrong_scores.append(current_wrong_count)
        # train_no_scores.append(current_no_count)
       
        # test_correct_scores.append(test_correct)
        # test_wrong_scores.append(test_wrong)
        # test_no_scores.append(test_no)
    return  model, train_correct_scores, train_wrong_scores, train_no_scores


epochs = 400

# Set the config of the model. These were picked for the iris data. 
config = {}
config['n_per_class'] = 1
config['n_input_neurons'] = 10 
config['tau'] = 0.464
config['tau2'] = 0.81 
config['eta'] = 0.001
config['eta2'] = 0.01

from random import shuffle

train_idx = list(range(0,150,2))
test_idx = list(range(1,150,2))
shuffle(train_idx)

#Run training and testing
model, train_correct_scores, train_wrong_scores, train_no_scores = run_training(epochs,dataset,y,train_idx,config)
test_correct, test_wrong, test_no = evaluate(model, dataset,y,test_idx)
print("Test :{:.2f}".format(test_correct))

# for train_idx, test_idx in tqdm(rkf.split(dataset)):
#     iris_train, y_train = np.array(dataset)[train_idx], y[train_idx]
#     iris_test, y_test = np.array(dataset)[test_idx], y[test_idx]
#     train_correct_scores, train_wrong_scores, train_no_scores, test_correct_scores, test_wrong_scores, test_no_scores = run_training(epochs,iris_train,y_train,iris_test,y_test,config)
#     if SAVE:
#         exp = "./" 
#         np.save(exp+"iris_train_correct_{}.npy".format(i),train_correct_scores)
#         np.save(exp+"iris_train_wrong_{}.npy".format(i),train_wrong_scores)
#         np.save(exp+"iris_train_no_{}.npy".format(i),train_no_scores)
#         np.save(exp+"iris_test_correct_{}.npy".format(i),test_correct_scores)
#         np.save(exp+"iris_test_wrong_{}.npy".format(i),test_wrong_scores)
#         np.save(exp+"iris_test_no_{}.npy".format(i),test_no_scores)
#     print("Train :{} Test :{}".format(train_correct_scores[-1],test_correct_scores[-1]))
#     i += 1

