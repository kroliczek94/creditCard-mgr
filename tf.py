import pandas as pd
import numpy as np

# VARIABLES
FILE_NAME = 'creditcard.csv'
LABEL_STR = 'Class'
splitRate = 0.8
# Import and store dataset

df_read = pd.read_csv(FILE_NAME)


# print(credit_card_data)

# Splitting data into 4 sets
# 1. Shuffle/randomize data
# 2. One-hot encoding
# 3. Normalize
# 4. Splitting up X/y values
# 5. Convert data_frames to numpy arrays (float32)
# 6. Splitting the final data into X/y train/test

# Shuffle and randomize data
def shuffle(df):
    return df.sample(frac=1)


def normalize(df):
    return (df - df.min()) / (df.max() - df.min())


def splitVariablesAndLabels(df):
    df_X = df.drop([LABEL_STR + '_0', LABEL_STR + '_1'], axis=1)
    df_y = df[[LABEL_STR + '_0', LABEL_STR + '_1']]

    return df_X, df_y

def convertToNumpyArray(df):
    return np.asarray(df.values, dtype='float32')

def splitToTrainingTestData(x, y):
    train_size = int(splitRate * len(x))
    (raw_X_train, raw_y_train) = (x[:train_size], y[:train_size])
    # print((raw_X_train[1], raw_y_train[1]))

    (raw_X_test, raw_y_test) = (x[train_size:], y[train_size:])
    return raw_X_train, raw_y_train, raw_X_test, raw_y_test

df_shuffled = shuffle(df_read)
# print(shuffled_data[:5])
# print (shuffled_data.show(5))
# Change Class column into Class_0 ([1 0] for legit data) and Class_1 ([0 1] for fraudulent data)

df_converted = pd.get_dummies(df_shuffled, columns=[LABEL_STR])
df_normalized = normalize(df_converted)
df_X, df_y = splitVariablesAndLabels(df_normalized)

# Convert both data_frames into np arrays of float32
ar_X = convertToNumpyArray(df_X)
ar_y = convertToNumpyArray(df_y)

# Allocate first 80% of data into training data and remaining 20% intxo testing data
raw_X_train, raw_y_train, raw_X_test, raw_y_test = splitToTrainingTestData(ar_X, ar_y)
# print((raw_X_test[1], raw_y_test[1]))

# Gets a percent of fraud vs legit transactions (0.0017% of transactions are fraudulent)
count_legit, count_fraud = np.unique(df_read[LABEL_STR], return_counts=True)[1]
# print(count_legit)
# print(count_fraud)

fraud_ratio = float(count_fraud / (count_legit + count_fraud))
print('Percent of fraudulent transactions: ', fraud_ratio)

# Applies a logit weighting of 578 (1/0.0017) to fraudulent transactions to cause model to pay more attention to them
weighting = 1 / fraud_ratio
raw_y_train[:, 1] = raw_y_train[:, 1] * weighting

import tensorflow as tf

# 30 cells for the input
input_dimensions = ar_X.shape[1]
# print(input_dimensions)

# 2 cells for the output
output_dimensions = ar_y.shape[1]
# print(output_dimensions)

# 100 cells for the 1st layer
num_layer_1_cells = 100
# 150 cells for the second layer
num_layer_2_cells = 150

# We will use these as inputs to the model when it comes time to train it (assign values at run time)
X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name='X_train')
y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name='y_train')

# We will use these as inputs to the model once it comes time to test it
X_test_node = tf.constant(raw_X_test, name='X_test')
y_test_node = tf.constant(raw_y_test, name='y_test')

# First layer takes in input and passes output to 2nd layer
weight_1_node = tf.Variable(tf.zeros([input_dimensions, num_layer_1_cells]), name='weight_1')
biases_1_node = tf.Variable(tf.zeros([num_layer_1_cells]), name='biases_1')

# Second layer takes in input from 1st layer and passes output to 3rd layer
weight_2_node = tf.Variable(tf.zeros([num_layer_1_cells, num_layer_2_cells]), name='weight_2')
biases_2_node = tf.Variable(tf.zeros([num_layer_2_cells]), name='biases_2')

# Third layer takes in input from 2nd layer and outputs [1 0] or [0 1] depending on fraud vs legit
weight_3_node = tf.Variable(tf.zeros([num_layer_2_cells, output_dimensions]), name='weight_3')
biases_3_node = tf.Variable(tf.zeros([output_dimensions]), name='biases_3')


# Function to run an input tensor through the 3 layers and output a tensor that will give us a fraud/legit result
# Each layer uses a different function to fit lines through the data and predict whether a given input tensor will \
#   result in a fraudulent or legitimate transaction
def network(input_tensor):
    # Sigmoid fits modified data well
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weight_1_node) + biases_1_node)
    # Dropout prevents model from becoming lazy and over confident
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, weight_2_node) + biases_2_node), 0.85)
    # Softmax works very well with one hot encoding which is how results are outputted
    layer3 = tf.nn.softmax(tf.matmul(layer2, weight_3_node) + biases_3_node)
    return layer3


# Used to predict what results will be given training or testing input data
# Remember, X_train_node is just a placeholder for now. We will enter values at run time
y_train_prediction = network(X_train_node)
y_test_prediction = network(X_test_node)

# Cross entropy loss function measures differences between actual output and predicted output
cross_entropy = tf.losses.softmax_cross_entropy(y_train_node, y_train_prediction)

# Adam optimizer function will try to minimize loss (cross_entropy) but changing the 3 layers' variable values at a
#   learning rate of 0.005
optimizer = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)


# Function to calculate the accuracy of the actual result vs the predicted result
def calculate_accuracy(actual, predicted):
    actual = np.argmax(actual, 1)
    predicted = np.argmax(predicted, 1)
    return (100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0])


num_epochs = 100

import time

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        start_time = time.time()

        _, cross_entropy_score = session.run([optimizer, cross_entropy],
                                             feed_dict={X_train_node: raw_X_train, y_train_node: raw_y_train})

        # if epoch % 10 == 0:
        timer = time.time() - start_time

        print('Epoch: {}'.format(epoch), 'Current loss: {0:.4f}'.format(cross_entropy_score),
              'Elapsed time: {0:.2f} seconds'.format(timer))

        final_y_test = y_test_node.eval()
        final_y_test_prediction = y_test_prediction.eval()
        final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
        print("Current accuracy: {0:.2f}%".format(final_accuracy))

final_y_test = y_test_node.eval()
final_y_test_prediction = y_test_prediction.eval()
final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
print("Final accuracy: {0:.2f}%".format(final_accuracy))

final_fraud_y_test = final_y_test[final_y_test[:, 1] == 1]
final_fraud_y_test_prediction = final_y_test_prediction[final_y_test[:, 1] == 1]
final_fraud_accuracy = calculate_accuracy(final_fraud_y_test, final_fraud_y_test_prediction)
print('Final fraud specific accuracy: {0:.2f}%'.format(final_fraud_accuracy))
