import tensorflow as tf
import numpy as np

class Estimator:

    """
    Estimator class implements the function approximation for DQN.

    The Estimator class defines a NN that is used by DQN to estimate the Q-function values.
    It takes the classification state as input, followed by a fully connected layer with 
    sigmoid activation of dimensionality 10. The output is then concatenates with action representations followed by 
    a fully connected layers with sigmoid activation of size 5 and then the last linear fully connceted layer with 1 output.

    Attributes:
        classifier_placeholder:         A TF placeholder of shape any x state_length for the classification state.
        action_placeholder:             A TF placeholder of shape any x action_length for the action (datapoint) state.  
        predictions:                    A tensor of size any x 1 that contains predictions of the approximation by Q-network.
        summaries:                      A TF summary object that will contain the stats for result analysis.
    """
    


    def __init__(self, state_length, action_length, is_target_dqn, var_scope_name, bias_average, max_batch):

        """
        Initializes the estimator.
        
        A computational graph that computes Q-values starting from 
        classification state and action state.

        Args:
            state_length:    An integer indicating the number of features in classifier state.
            action_length:        An integer indicating the number of features in action state.
            is_target_dqn:              A boolean indicating if the Estimator is a target network. 
                                        Only normal (not target) network is trained, the other one is a lagging copy.
            var_scope_name:             A string, can be "dqn" or "target_dqn", for example.
            bias_average:               A float that is used to initialize the bias in the last layer.
        """

        self.classifier_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, state_length], name="X_classifier")
        self.action_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, max_batch, action_length), name="X_datapoint")

        with tf.variable_scope(var_scope_name):

            # Ensure the classifier_placeholder is reshaped to 2D.
            # self.classifier_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 3, 20))

            # Flatten the classifier_placeholder for fc1 input.
            flattened_input = tf.reshape(self.classifier_placeholder, [-1, 30])
            
            # A fully connected layers with classifier_placeholder as input.
            fc1 = tf.contrib.layers.fully_connected(
                inputs=flattened_input, 
                num_outputs=10, 
                activation_fn=tf.nn.sigmoid,
                trainable=not is_target_dqn,
                variables_collections=[var_scope_name],
            )

            # Concatenate the output of first fully connected layer with action_placeholder.
            # Ensure action_placeholder is compatible for concatenation.
            fc2concat = tf.concat([fc1, tf.reshape(self.action_placeholder, [-1, action_length * max_batch])], 1)

            # A fully connected layer with fc2concat as input.
            fc3 = tf.contrib.layers.fully_connected(
                inputs=fc2concat, 
                num_outputs=5, 
                activation_fn=tf.nn.sigmoid,
                trainable=not is_target_dqn,
                variables_collections=[var_scope_name]
            )

            # The last linear fully connected layer.
            # The bias on the last layer is initialized to some value.
            # Normally it is the - average episode duriation / 2,
            # like this NN find optimum better even as the mean is not 0.
            self.predictions = tf.contrib.layers.fully_connected(
                inputs=fc3, 
                num_outputs=1, 
                biases_initializer=tf.constant_initializer(bias_average),
                activation_fn=None,
                trainable=not is_target_dqn,
                variables_collections=[var_scope_name],
            )

            # Summaries for tensorboard.
            # Can write weights to check how they are developed, 
            # but it takes more space and used only for debugging.
            #tf.summary.histogram("estimator/fc1", fc1)
            #tf.summary.histogram("estimator/fc2", fc2)
            #tf.summary.histogram("estimator/fc3", fc3)
            tf.compat.v1.summary.histogram("estimator/q_values", self.predictions)
            self.summaries = tf.compat.v1.summary.merge_all()