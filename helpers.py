import numpy as np



class Minibatch:

    """
    Minibatch class that helps for gradient descent training.

    Attributes:
        state:       A numpy.ndarray of size batch_size x # classifier features characterizing the state of classifier at the sampled iterations.
        action:           A numpy.ndarray of size batch_size x # action features characterizing the action that was taken at the sampled iterations.
        reward:                 A numpy.ndarray of size batch_size.
        next_state:  A numpy.ndarray of size batch_size x # classifier features.
        next_action:      A list of size batch_size of numpy.ndarrays characterizing the possible actions that were available at the sampled iterations.
        terminal:               A numpy.ndarray of size batch_size of booleans indicating if the iteration was terminal.
        indices:                A numpy.ndarray of size batch_size that contains indices of samples iterations in the replay buffer.
    """
    


    def __init__(self, state, action, reward, next_state, next_action, terminal, indices):

        # Inits the Minibatch object and initializes the attributes with given values.   
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.next_action = next_action
        self.terminal = terminal
        self.indices = indices
    
    

class ReplayBuffer:

    """
    Replay Buffer is used to store the transactions from episodes.

    Attributes:
        buffer_size:                An integer indicating the maximum number of transactions to be stored in the replay buffer.
        n:                          An integer, the maximum index to be used for sampling. It is useful when the buffer is not filled in fully.
                                    It grows from 0 till the buffer_size-1 and then stops changing.
        write_index:                An integer, the index where the next transaction should be written. 
                                    Goes from 0 till the buffer_size-1 and then starts from 0 again.
        max_td_error:               A float used to initialize the td error of newly added samples.
        prior_exp:                  A float that is used for turning the td error into a probability to be sampled. 
        all_state:       A numpy.ndarray of size batch_size x #classifier features
                                    characterising the state of classifier at the sampled iterations.
        all_actions:          A numpy.ndarray of size batch_size x #action features,
                                    characterizing the action that was taken at the sampled iterations.
        all_rewards:                A numpy.ndarray of size batch_size.
        all_next_states: A numpy.ndarray of size batch_size x #classifier features.
        all_next_action:      A list of size batch_size of numpy.ndarrays,
                                    characterizing the possible actions that were available at the sampled iterations.
        all_terminals:              A numpy.ndarray of size batch_size of booleans indicating if the iteration was terminal.
        all_td_errors:              A numpy.ndarray of size batch_size with td errors of transactions 
                                    when each of them was used in a gradient update.
        max_td_error:               A float with the highest (absolute) value of td error from all transactions stored in the buffer.
    """
    


    def __init__(self, buffer_size=1e4, prior_exp=0.5):

        # Inits a few attributes with 0 or the default values.
        self.buffer_size = int(buffer_size)
        self.n = 0
        self.write_index = 0
        self.max_td_error = 1000.0
        self.prior_exp = prior_exp


    
    def _init_nparray(self, state, action, reward, next_state, next_action, terminal):
        
        # Initialize numpy arrays of all_xxx attributes to one transaction repeated buffer_size times.
        self.all_states = np.array([state] * self.buffer_size)
        self.all_action = [action] * self.buffer_size
        self.all_rewards = np.array([reward] * self.buffer_size)
        self.all_next_states = np.array([next_state] * self.buffer_size)
        self.all_next_actions = [next_action] * self.buffer_size
        self.all_terminals = np.array([terminal] * self.buffer_size)
        self.all_td_errors = np.array([self.max_td_error] * self.buffer_size)

        # Set the counters to 1 as one transaction is stored.
        self.n = 1
        self.write_index = 1


  
    def store_transition(self, state, action, reward, next_state, next_action, terminal):

        # Add a new transaction to a replay buffer.
        # If buffer arrays are not yet initialized, initialize it.
        if self.n == 0:
            self._init_nparray(state, action, reward, next_state, next_action, terminal)
            return
        
        # Write a transaction at a write_index position.
        self.all_states[self.write_index] = state
        self.all_action[self.write_index] = action
        self.all_rewards[self.write_index] = reward
        self.all_next_states[self.write_index] = next_state
        self.all_next_actions[self.write_index] = next_action
        self.all_terminals[self.write_index] = terminal
        self.all_td_errors[self.write_index] = self.max_td_error

        # Keep track of the index for writing.
        self.write_index += 1
        if self.write_index >= self.buffer_size:
            self.write_index = 0

        # Keep track of the max index to be used for sampling.
        if self.n < self.buffer_size:
            self.n += 1



    def sample_minibatch(self, batch_size=32):

        """
        Sample a new minibatch from replay buffer.
        
        Args:
            batch_size:     An integer indicating how many transactions to be sampled from a replay buffer.
            
        Returns:
            minibatch:      An object of class Minibatch with sampled transactions.
        """

        # Get td error of samples that were written in the buffer.
        td_errors_to_consider = self.all_td_errors[:self.n]

        # Scale and normalize the td error to turn it into a probability for sampling.
        p = np.power(td_errors_to_consider, self.prior_exp) / np.sum(np.power(td_errors_to_consider, self.prior_exp))

        # Choose indices to sample according to the computed probability.
        # The higher the td error is, the more likely it is that the sample will be selected.
        minibatch_indices = np.random.choice(range(self.n), size=batch_size, replace=True, p=p)

        minibatch = Minibatch(
            self.all_states[minibatch_indices],
            [self.all_action[i] for i in minibatch_indices],
            self.all_rewards[minibatch_indices],
            self.all_next_states[minibatch_indices],
            [self.all_next_actions[i] for i in minibatch_indices],
            self.all_terminals[minibatch_indices],
            minibatch_indices,
        )

        return minibatch
    


    def update_td_errors(self, td_errors, indices):

        """
        Updates td_errors in replay buffer.
        
        After a gradient step was made, we need to updates 
        td errors to recently calculated errors.
        
        Args:
            td_errors:  A numpy array with new td errors.
            indices:    A numpy array with indices of points which td errors should be updated.
        """

        # Set the values for prioritized replay to the most recent td errors.
        self.all_td_errors[indices] = np.ravel(np.absolute(td_errors))

        # Find the max error from the replay buffer that will be used as a default value for new transactions.
        self.max_td_error = np.max(self.all_td_errors)