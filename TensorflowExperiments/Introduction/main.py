########################################################################################################################
#   First program, simple addition between two tensors
########################################################################################################################
import tensorflow as tf

# define the graph
x = tf.placeholder(tf.float32, name= "x")
y = tf.placeholder(tf.float32, name= "y")

addition = tf.add(x, y, name="addition")

# Create the session
with tf.Session() as session:
    result = session.run(addition, feed_dict={x: [1], y: [4]})

    print(result)
