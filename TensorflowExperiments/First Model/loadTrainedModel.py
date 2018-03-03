import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
########################################################################################################################
#
#   DATA LOADING SECTION
#
########################################################################################################################

# Load data into memory from csv
trainingData = pd.read_csv("./data/sales_data_training.csv", dtype=float)

# Pull out columns X is data to train with, Y is the expected result
# axis = 1 is to drop the column, not the row
xTraining = trainingData.drop("total_earnings", axis=1).values
yTraining = trainingData[['total_earnings']].values

# Load test data
testData = pd.read_csv("./data/sales_data_test.csv", dtype=float)

xTest = testData.drop("total_earnings", axis=1).values
yTest = testData[["total_earnings"]].values

# Scale all data down to be in the range of 0 - 1
xScaler = MinMaxScaler(feature_range=(0, 1))
yScaler = MinMaxScaler(feature_range=(0, 1))

# Scale the training inputs and expected results
xScaledTraining = xScaler.fit_transform(xTraining)
yScaledTraining = yScaler.fit_transform(yTraining)

xScaledTest = xScaler.transform(xTest)
yScaledTest = yScaler.transform(yTest)

########################################################################################################################
#
#   MODEL DEFINITION
#
########################################################################################################################

# Define model parameters
runName = "Run 1 with 50 nodes"
learningRate = 0.001
learningEpochs = 100
displayStep = 5

# Define number of inputs/outputs
numberInputs = 9
numberOutputs = 1

# Define how many neurons to have in each hidden layer
layer1Nodes = 50
layer2Nodes = 100
layer3Nodes = 50

########################################################################################################################
#
#   SETTING UP THE MODEL
#
########################################################################################################################
# Inputs
with tf.variable_scope("input"):
    x = tf.placeholder(dtype=tf.float32, shape=(None,numberInputs))

# Layer 1
with tf.variable_scope("layer1"):
    weights = tf.get_variable(name="weights1", shape=[numberInputs, layer1Nodes],\
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="bias1", shape=[layer1Nodes], initializer=tf.zeros_initializer())
    layer1Outputs = tf.nn.relu(tf.matmul(x, weights) + biases)

# Layer 2
with tf.variable_scope("layer2"):
    weights = tf.get_variable(name="weights2", shape=[layer1Nodes, layer2Nodes],\
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="bias2", shape=[layer2Nodes], initializer=tf.zeros_initializer())
    layer2Outputs = tf.nn.relu(tf.matmul(layer1Outputs, weights) + biases)

# Layer 3
with tf.variable_scope("layer3"):
    weights = tf.get_variable(name="weights3", shape=[layer2Nodes, layer3Nodes],\
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="bias3", shape=[layer3Nodes], initializer=tf.zeros_initializer())
    layer3Outputs = tf.nn.relu(tf.matmul(layer2Outputs, weights) + biases)

# Outputs
with tf.variable_scope("output"):
    weights = tf.get_variable(name="weights4", shape=[layer3Nodes, numberOutputs],\
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="bias4", shape=[numberOutputs], initializer=tf.zeros_initializer())
    predictions = tf.nn.relu(tf.matmul(layer3Outputs, weights) + biases)


########################################################################################################################
#
#   TRAINING DEFINITION
#
########################################################################################################################

# Cost function
with tf.variable_scope("cost"):
    # Expected value
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    # Cost = mean squared difference between predictions and expected
    cost = tf.reduce_mean(tf.squared_difference(predictions, y))

with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

########################################################################################################################
#
#   LOGGING/SAVING DEFINITIONS
#
########################################################################################################################

# Summary operation to log progress
with tf.variable_scope("logging"):
    tf.summary.scalar("currentCost", cost)
    summary = tf.summary.merge_all()

# This object will be used to save the model
saver = tf.train.Saver()

########################################################################################################################
#
#   TRAINING LOOP
#
########################################################################################################################

# Initialise the session
with tf.Session() as session:

    # load the pretrained model
    saver.restore(session, "./logs/trainedModel.ckpt")

    finalTrainingCost = session.run(cost, feed_dict={x: xScaledTraining, y: yScaledTraining})
    finalTestingCost = session.run(cost, feed_dict={x: xScaledTest, y: yScaledTest})
    print("Final Training Cost: {}", format(finalTrainingCost))
    print("Final Testing Cost: {}", format(finalTestingCost))

    savePath = saver.save(session, "./logs/{}/trainedModel.ckpt".format(runName))