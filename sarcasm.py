"""
    title: Course Project - Sarcasm Detection with Using Keras
    author: Sydney Yeargers
    date created: 12/2/21
    date last modified: 12/12/21
    python version: 3.9.7
    description: This application uses Keras software to demonstrate the impact that layering and hyperparameter tuning
    have on the validation accuracy of a deep learning model. This program specifically looks at the model's ability to
    classify a headline as sarcastic or not.
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
tf.random.set_seed(89)
np.random.seed(89)


def plot_history(history, title):
    """
    Creates a figure with two plots, the first with training accuracy and validation accuracy over epochs, and the
    second with training loss and validation loss over epochs.

    :param history: a Keras History object representing the history of a model
    :param title: a string object representing a title of the graph
    :return: a figure object of two plots representing the accuracy and loss history
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(12, 5))  # initialize figure
    # create validation plot
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training accuracy')  # plot training accuracy
    plt.plot(x, val_acc, 'r', label='Validation accuracy')  # plot validation accuracy
    plt.title('Training and validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    # create loss plot
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')  # plot training loss
    plt.plot(x, val_loss, 'r', label='Validation loss')  # plot validation loss
    plt.title('Training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    fig.suptitle(title)  # add figure title
    plt.show()


def logistic(x_train, train_labels, x_test, test_labels):
    """
    Trains and evaluates a classification model using logistic regression

    :param x_train: a csr_matrix object representing the count vector for training data
    :param train_labels: an ndArray object representing the classification of each training observation
    :param x_test: a csr_matrix object representing the count vector for test data
    :param test_labels: an ndArray object representing the classification of each testing observation
    :return: a float object representing the validation accuracy
    """
    classifier = LogisticRegression(max_iter=500, verbose=False)
    classifier.fit(x_train, train_labels)  # train the model
    score = classifier.score(x_test, test_labels)  # get validation accuracy
    print("Accuracy (logistic regression) :", score)
    return score


def basic_NN(x_train, train_labels, x_test, test_labels, batch_size, callbacks, lr=1e-3):
    """
    Trains and evaluates a basic Keras Sequential model using Dense layers

    :param x_train: a csr_matrix object representing the count vector for training data
    :param train_labels: an ndArray object representing the classification of each training observation
    :param x_test: a csr_matrix object representing the count vector for test data
    :param test_labels: an ndArray object representing the classification of each test observation
    :param batch_size: an integer object representing input value for hyperparameter batch_size
    :param callbacks: a list object representing the callbacks for the model to implement
    :param lr: a float object (default = 1e-3) representing input value for hyperparameter learning rate
    :return: a float object representing the validation accuracy, and a Keras History object representing the history of
        a model
    """
    input_dim = x_train.shape[1]
    # create model
    model = Sequential([
        layers.Dense(100, input_dim=input_dim, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(1, activation='sigmoid')], name='sarcasm-detection')
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy'])
    # train model
    history = model.fit(x_train, train_labels, epochs=100, verbose=2, batch_size=batch_size,
                        validation_data=(x_test, test_labels), callbacks=callbacks)
    # evaluate model
    loss, accuracy = model.evaluate(x_test, test_labels, verbose=False)
    print("Testing Accuracy: {:.4f}".format(accuracy))
    return accuracy, history


def nn_dropout(x_train, train_labels, x_test, test_labels, dropout, batch_size, callbacks):
    """
    Trains and evaluates a basic Keras Sequential model using Dense and Dropout layers

    :param x_train: a csr_matrix object representing the count vector for training data
    :param train_labels: an ndArray object representing the classification of each training observation
    :param x_test: a csr_matrix object representing the count vector for test data
    :param test_labels: an ndArray object representing the classification of each test observation
    :param dropout: a float object representing input value for hyperparameter rate
    :param batch_size: an integer object representing input value for hyperparameter batch_size
    :param callbacks: a list object representing the callbacks for the model to implement
    :return: a float object representing the validation accuracy, and a Keras History object representing the history of
        a model
    """
    input_dim = x_train.shape[1]
    # create model
    model = Sequential()
    model.add(layers.Dense(100, input_dim=input_dim, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    # train model
    history = model.fit(x_train, train_labels, epochs=100, verbose=1, batch_size=batch_size,
                        validation_data=(x_test, test_labels), callbacks=callbacks)
    # evaluate model
    loss, accuracy = model.evaluate(x_test, test_labels, verbose=False)
    return accuracy, history


def embedd_param(train_headlines, train_labels, test_headlines, test_labels, embeddim, dropout, batch_size,
                 callbacks):
    """
    Converts data to padded sequences, then trains and evaluates a basic Keras Sequential model using Embedding, Dense,
    and Dropout layers

    :param train_headlines: an ndArray object representing unprocessed training data
    :param train_labels: an ndArray object representing the classification of each unprocessed training observation
    :param test_headlines: an ndArray object representing unprocessed test data
    :param test_labels: an ndArray object representing the classification of each unprocessed test observation
    :param embeddim: an integer object representing input value for hyperparameter output_dim
    :param dropout: a float object representing input value for hyperparameter rate
    :param batch_size: an integer object representing input value for hyperparameter batch_size
    :param callbacks: a list object representing the callbacks for the model to implement
    :return:a float object representing the validation accuracy, and a Keras History object representing the history of
        a model
    """
    # preprocess data
    tokenizer = Tokenizer(num_words=10000, oov_token="<oov>", lower=True)
    tokenizer.fit_on_texts(train_headlines)
    x_train = tokenizer.texts_to_sequences(train_headlines)
    maxlen = len(max(x_train, key=len))
    x_test = tokenizer.texts_to_sequences(test_headlines)
    maxvalstest = len(max(x_test, key=len))
    if maxvalstest > maxlen:
        maxlen = maxvalstest
    x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
    vocab_size = len(tokenizer.word_index) + 1  # plus one for reserved 0 index

    # create model
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embeddim, input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    # train model
    history = model.fit(x_train, train_labels, epochs=100, verbose=1, batch_size=batch_size,
                        validation_data=(x_test, test_labels), callbacks=callbacks)
    # evaluate model
    loss, accuracy = model.evaluate(x_test, test_labels, verbose=False)
    return accuracy, history


def main():
    t0 = time.time()  # start timer
    df = pd.read_json('./Sarcasm_Headlines_Dataset_v2.json', lines=True)  # read in data
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)  # delete duplicated observations
    headlines = df['headline'].values  # isolate predictor data
    labels = df['is_sarcastic'].values  # isolate response data
    # split into training and test set
    train_headlines, test_headlines, train_labels, test_labels = train_test_split(headlines, labels,
                                                                                  test_size=.25, random_state=10)

    #  Using logistic regression:
    vectorizer = CountVectorizer(lowercase=True)
    vectorizer.fit(train_headlines)
    x_train = vectorizer.transform(train_headlines)  # get count vector for training data
    x_test = vectorizer.transform(test_headlines)  # get count vector for test data
    print("Fitting Logistic Model")
    # build logistic regression model and save validation accuracy
    accs = {"Logistic Regression": logistic(x_train, train_labels, x_test, test_labels)}

    # Using a basic artificial neural network:
    # set callbacks and arbitrary batch_size
    batch_size = 200
    callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)]
    print("Fitting Basic Neural Network Model...")
    nn_history = basic_NN(x_train, train_labels, x_test, test_labels, batch_size, callbacks)  # build model
    plot_history(nn_history[1], 'Model 1: Basic Neural Network')  # plot accuracies and losses
    accs["Basic Neural Network"] = nn_history[0]  # save validation accuracy

    # Using an artificial neural network with an adjusted learning rate:
    print("Fitting Basic Neural Network Model with Adjusted LR...")
    nn_history = basic_NN(x_train, train_labels, x_test, test_labels, batch_size, callbacks, lr=1e-4)  # build model
    plot_history(nn_history[1], 'Model 2: Basic Neural Network with Adjusted Learning Rate')  # plot accuracies and losses
    accs["Basic NN with Adjusted LR"] = nn_history[0]  # save validation accuracy

    # Using an artificial neural network with dropout layers:
    print("Finding Best Parameters for Neural Network...")
    # initialize values to test for hyperparameters
    dropouts = np.arange(0.1, 0.9, 0.2)
    batch_sizes = np.arange(100, 350, 50)
    maxacc = 0
    bestdrop = 0.0
    bestbatch = 0
    # for each possible value of batch_sizes
    for size in batch_sizes:
        print("Testing batch_size {:n}...".format(size))
        # for each possible value of rate
        for drop in dropouts:
            print("Testing dropout rate {:1.1f}...".format(drop))
            param_test = nn_dropout(x_train, train_labels, x_test, test_labels, drop, size, callbacks)  # build model
            accuracy = param_test[0]
            # if accuracy is the highest found
            if accuracy > maxacc:
                maxacc = accuracy  # set as max accuracy
                bestdrop = drop  # save value for drop
                bestbatch = size  # save value for batch_size
                besthist = param_test[1]  # save model history
    print("Best model: drop rate = {:1.1f}; batch size = {:n}; testing accuracy = {:.4f} ".format(bestdrop, bestbatch,
                                                                                                  maxacc))
    plot_history(besthist, 'Model 3: Neural Network with Best Drop Rate and Batch size')  # plot accuracies and losses
    accs["Basic NN with Dropout"] = maxacc  # save validation accuracy

    # Using an artificial neural network with embedding layers:
    print("Finding Best Parameters for Neural Network with Embedding...")
    # initialize values to test for hyperparameters
    ed = np.arange(128, 384, 64)
    maxacc = 0
    bestdim = 0
    # for each possible value of output_dim
    for dim in ed:
        print("Testing embedding dimension = {:n}...".format(dim))
        e_param = embedd_param(train_headlines, train_labels, test_headlines, test_labels, dim, bestdrop, bestbatch,
                               callbacks) # build model
        accuracy = e_param[0]
        # if accuracy is the highest found
        if accuracy > maxacc:
            maxacc = accuracy  # set as max accuracy
            bestdim = dim  # save value for output_dim
            besthist1 = e_param[1]  # save model history
    print("Best dim is {:1.1f} with a testing Accuracy of : {:.4f}".format(bestdim, maxacc))
    plot_history(besthist1, 'Model 4: Embedded NN with Best Embedding Output Dimension')  # plot accuracies and losses
    accs["NN with Text Embedding"] = maxacc  # save validation accuracy

    df_acc = pd.DataFrame.from_dict(accs, orient='index')  # put accuracies in a table by model
    print(df_acc)
    runtime = time.time() - t0  # get runtime of program
    print("Program Runtime: ", round(round(runtime) / 60, 2), " minutes.")


main()
