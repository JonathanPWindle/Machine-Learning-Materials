## Loading data
* Use pandas to read csv and manipulate the data
* By only dropping the earnings, it will use every other column of data to train on
* As it's the earnings trying to predict, the earnings column is used for the expected result
* Data is scaled down using the _MinMaxScaler_ in the _Scikit-learn_ library
* Important that the test data and training data are bot scaled using the ***same*** Scalers otherwise it won't work

## Network Definition
* 9 inputs
* 3 hidden layers
    * 50 Nodes
    * 100 Nodes
    * 50 Nodes
 * 1 Ouput
 * All fully connected for simplicity
 * Learning rate: 0.001
 * Training epochs: 100
 * Display steps: 5
 
 ## Network Creation
 * Keep all layers in their own variable scope for some abstraction
 * Input layer
    * Use ```tf.placeholder``` to store the inputs
         * Pass it the type to expect, in this case ```float32```
         * The shape, in this case, ```None, numberOfInputs``` ```None``` tells the neuron to take in different sized batches of 9 floats
  * Layer 1
    * Contains:
        * Weights
        * Bias
        * Outputs
    * Bias uses ```tf.getvariable``` because the network needs to remember this value over time.
        * Also takes in ```tf.zeros_initializer()``` to as it suggests, initialise the biases to 0
    * Weights are also a variable
        * The shape of this is ```Number of inputs, Number of nodes in layer 1```
        * Pass in the _tensorflow_ initializer ```xavier_initializer()```
    * Outputs are calculated using matrix multiplication and a standard relu function
        * Multiply using ```tf.matmul(x,weights)``` and add the biases
        * pass this result into the relu function using ```tf.nn.relu(tf.matmul(x,weights) + biases)```
  * Layer 2
    * Very similar to layer 1
    * Weight shape changes to be ```layer1Nodes, layer2Nodes```
    * Output is calculated using ```layer1Outputs``` rather than inputs ```x```
  * Layer 3
    * Same again, changing relevant variables
  * Output
    * Same again, changing relevant variables
    
 ## Training Definition
 
 * Cost function
    * Define ```y``` as the expected value to feed in
        * ```y = tf.placeholder(tf.float32, shape = [None,1])```
        * Take different sized batches of size 1
    * Calculate the cost as the _mean squared error_ between the prediction and the expected value.
        * Use ```tf.reduce_mean(tf.squared_difference(prediction, y))``` 
 * Optimizer function
    * Tensorflow calls this to train the network
    * Using the AdamOptimizer, must give it the training rate
        * ```optimizer = tf.train.AdamOptimizer(learningRate)```
    * Need to tell tensorflow which variable to minimize, pass it the cost
        * ```.minimize(cost)```
    * All in all, this tells tensorflow to whenever it executes the optimizer, run one iteration of the Adam Optimizer in an attempt to make the cost smaller
 
## Training Loop
* Create session to run operations on the graph defined above
    * Use ```with tf.Session() as session:``` to create this
    * First have to run the session and set all variables to their defaults
        * ```session.run(tf.global_variables_initializer())```
    * Each epoch is one run through the **entire data set**
    * Feed in the training data to the optimizer
        * ```feed_dict={x: scaledTrainingData}``` will feed the scaled training data into the x variable defined in the graph 
        * ```session.Run(optimizer,feed_dict{x: xScaledTraining, y: yScaledTraining})```
    * Check progress by running the cost function on the data set ever 5 epochs, the cost should decrease each time
        * ```trainingCost = session.run(cost,feed_dict={x: xScaledTraining, y: yScaledTraining})```

## Logging with TensorBoard
* Scalars
    * Log single values over time and view as graphs
    * To add a scalar, use ```tf.summary.scalar("currentCost", cost)```
* To manually run each individual summary code can be tedious
    * ```summary = tf.summary.merge_all()``` When the summary node is called, it will execute all summary nodes in the graph
* During the session, create two log files, one for training, one for testing data.
    * ```trainingWriter = tf.summary.FileWriter("./logs/training", session.graph)```
* To update the summary, the summary node needs to be called in the session
    * Can call multiple nodes in one ```session.run```  statement
        * ```trainingCost, trainingSummary = session.run([cost,summary], feed_dict{x: xScaledTraining, y: yScaledTraining})```
        * This runs the two nodes and stores results in the respective variables
        
## Saving/Loading Models
* Use a ```tf.train.Saver``` object to save the model, defined outside of the session
* To save ```saver.save(session, "./logs/trainedModel.ckpt")``` after training within the session scope.
* When loading a model, it's important not to initialise the variables, must load them instead
* To load ```saver.restore(session, "./logs/trainedModel.ckpt")```
## Side Notes

* Variables vs Placeholders:
    * Use ***Variables*** for trainable parameters
        * Values are derived from training
        * Initial values must be set
    * Use ***Placeholders*** for data feeding
        * Essentially allocates storage you know you will need
        * No initial value needed