import torch # A Python package for tensor computations similar to NumPy but with GPU support.
import torch.nn as nn # A subpackage for building neural networks.
import torch.optim as optim # A subpackage containing optimization algorithms (like Adam, SGD).
import numpy as np # A package for scientific computing, mainly for array manipulations.
import os # A module for interacting with the operating system.
from batch_estimator import Estimator # Importing a custom class Estimator for creating the Q-network. This class is assumed to be defined in a separate file.

"""
The DQN class implements a Deep Q-Network for reinforcement learning.
It initializes a Q-network and a target network,
selects actions based on current Q-values, calculates the target Q-values,
trains the network using experience replay,
and periodically updates the target network to ensure stable training.

The primary goal is to learn a policy that maximizes the expected cumulative reward
by approximating the optimal Q-values for each state-action pair.
"""

class DQN:

    def __init__(self,
                 observation_length=6,
                 learning_rate=1e-3,
                 batch_size=32,
                 target_copy_factor=0.001,
                 bias_average=0):
        
        """
        DQN CLASS INITIALIZATION.

        The DQN class encapsulates the Deep Q-Network (DQN) algorithm.
        This algorithm is used for training a neural network to approximate Q-values for reinforcement learning tasks.

        - observation_length: The length of the observation vector for each state.
        - learning_rate: The learning rate for the optimizer.
        - batch_size: The number of samples per training batch.
        - target_copy_factor: A factor for updating the target network.
        - bias_average: A parameter for the bias in the estimator.
        """

        
        # DEVICE SETUP AND MODEL INITIALIZATION.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Sets the device to GPU if available, otherwise CPU.
        self.i_train = 0 # Counter for training iterations.
        self.i_actions_taken = 0 # Counter for actions taken.
        self.batch_size = batch_size # Stores the batch size for training.
        self.target_copy_factor = target_copy_factor # Stores the target copy factor for updating the target network.

        # TARGET ESTIMATOR.
        # Creates the target network which is used to provide stable target Q-values.
        self.target_estimator = Estimator(observation_length, is_target_dqn=True, bias_average=bias_average).to(
            self.device)

        # ESTIMATOR.
        # Creates the main Q-network that is trained.
        self.estimator = Estimator(observation_length, is_target_dqn=False, bias_average=bias_average).to(self.device)

        # Optimizer.
        # An Adam optimizer to update the weights of the Q-network.
        self.optimizer = optim.Adam(self.estimator.parameters(), lr=learning_rate)

        # Loss function.
        # Mean Squared Error loss function to calculate the difference between predicted Q-values and target Q-values.
        self.loss_fn = nn.MSELoss()

        # Mode flag to initialize model parameters.
        # A flag to check if the model parameters have been initialized.
        self._initialized = False

    # INITIALIZATION CHECK AND WEIGHTS INITIALIZATION.
    def _check_initialized(self): # Ensures that the network weights are initialized only once.
        if not self._initialized:
            self.estimator.apply(self._weights_init) # Applies the _weights_init function to initialize the weights of the network.
            self.target_estimator.load_state_dict(self.estimator.state_dict()) # Copies the weights from the main Q-network to the target network.
            self._initialized = True # Sets the flag to indicate that initialization is complete.

    # WEIGHT INITIALIZATION FUNCTION.
    # Initializes the weights of linear layers using the Xavier uniform initialization and sets the biases to 0.01.
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # ACTION SELECTION.
    # Method to select an action based on the current state and the Q-network's predictions.
    def get_action(self, code_state, dataset, model, state, next_action_batch, next_action_unlabeled_data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Sets the device to GPU if available, otherwise CPU.
        # Counter of how many times this function was called.
        self.i_actions_taken += 1 # Increments the action counter.
        self._check_initialized() # Ensures the network is initialized.

        # STATE AND ACTION PREPARATION.
        # Converts inputs to NumPy arrays if they are not already. This ensures compatibility with subsequent operations.
        if not isinstance(state, np.ndarray):
            state = state.cpu().numpy()
        if not isinstance(next_action_batch, np.ndarray):
            if isinstance(next_action_batch, list):
                next_action_batch = torch.tensor(next_action_batch).float().to(device)
            next_action_batch = next_action_batch.cpu().numpy()
        if not isinstance(next_action_unlabeled_data, np.ndarray):
            next_action_unlabeled_data = next_action_unlabeled_data.cpu().numpy()

        # STATE DUPLICATION AND TESNOR CONVERTION.
        # Repeat classification_state so that we have a copy of classification state for each possible action.
        state = np.repeat([state], len(next_action_batch), axis=0) # Duplicates the state to match the number of possible actions.

        # Convert to torch tensors.
        state_tensor = torch.from_numpy(state).float().to(device) # Converts the state to a PyTorch tensor.
        next_action_batch_tensor = torch.from_numpy(next_action_batch).float().to(device) # Converts the next action batch to a PyTorch tensor.

        # Q-VALUE PREDICTION.
        # Predict q-values with current estimator.
        with torch.no_grad(): # Disables gradient calculation for efficiency.
            predictions = self.estimator(state_tensor, next_action_batch_tensor) # Predicts Q-values using the current Q-network.

        selected_batch = np.random.choice(np.where(predictions.cpu().numpy() == predictions.max().item())[0]) # Chooses the action with the highest Q-value.
        selected_batch = int(next_action_batch[selected_batch])

        # DATA EXTRACTION AND PREPARATION.
        if code_state=="Agent":
            input_data = dataset.agent_data[next_action_unlabeled_data, :] # Extracts the appropriate data based on code_state.
        elif code_state=="Test methods":
            input_data = dataset.test_methods_data[next_action_unlabeled_data, :]

        # Ensure the input data has the correct shape and number of channels.
        if input_data.shape[1] != 3: # Ensures the input data has 3 channels.
            raise ValueError(f"Expected input data to have 3 channels, but got {input_data.shape[1]} channels instead")

        # Convert to torch tensor and send to device.
        input_data_tensor = torch.from_numpy(input_data).float().to(device) # Converts the input data to a PyTorch tensor.

        # CLASS PROBABILITIES AND UNCERTAINTY SAMPLING.
        # Get the class probabilities.
        with torch.no_grad():
            train_predictions = model(input_data_tensor) # Gets class probabilities from the model.

        # Uncertainty sampling.
        uncertainty_scores = -np.abs(train_predictions[:, 0].cpu().numpy()) # Calculates uncertainty scores.
        selected_indices = np.argsort(uncertainty_scores)[:selected_batch] # Selects the indices with the highest uncertainty.

        if isinstance(selected_batch, torch.Tensor):
            selected_batch = selected_batch.cpu()

        return selected_batch, selected_indices

    # TRAINING FUNCTION.
    def train(self, minibatch): # Trains the Q-network using a minibatch of experiences.
        self._check_initialized()
        max_prediction_batch = [] # Stores the maximum target predictions for each minibatch entry.

        for i in range(len(minibatch.next_state)): # Iterates over each minibatch entry.
            next_state = np.repeat([minibatch.next_state[i]], len(minibatch.next_action[i]), axis=0) # Duplicates the next state.
            
            # Converts various components to tensors.
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            next_action_tensor = torch.tensor(minibatch.next_action[i], dtype=torch.float32).to(self.device)
            state_tensor = torch.tensor(minibatch.state, dtype=torch.float32).to(self.device)
            action_tensor = torch.tensor([[x] for x in minibatch.action], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                target_predictions = self.target_estimator(next_state_tensor, next_action_tensor) # Gets target Q-values from the target network.
                predictions = self.estimator(state_tensor, action_tensor) # Gets Q-values from the current network.

            # Find the best action by estimator.
            if len(predictions) == 0:
                # Handle case where predictions is empty (no valid actions).
                best_action_by_estimator = 0  # Choose a default action or handle appropriately.
            else:
                best_action_by_estimator = torch.argmax(predictions).item() # Finds the best action predicted by the estimator.

            # Ensure best_action_by_estimator is within valid range.
            if best_action_by_estimator >= len(target_predictions):
                best_action_by_estimator = len(target_predictions) - 1  # Clamp to the last valid index.

            max_target_prediction_i = target_predictions[best_action_by_estimator].item() # Gets the maximum target prediction.
            max_prediction_batch.append(max_target_prediction_i)

        # Converts various components to tensors for the current minibatch.
        state_tensor = torch.tensor(minibatch.state, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor([[x] for x in minibatch.action], dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(minibatch.reward, dtype=torch.float32).to(self.device)
        terminal_tensor = torch.tensor(minibatch.terminal, dtype=torch.bool).to(self.device)
        max_prediction_tensor = torch.tensor(max_prediction_batch, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad() # Resets gradients.

        predictions = self.estimator(state_tensor, action_tensor) # Predicts Q-values for the current state-action pairs.
        terminal_mask = ~terminal_tensor # Masks terminal states.
        masked_target_predictions = max_prediction_tensor * terminal_mask # Applies the terminal mask to the target predictions.

        actions_taken_targets = reward_tensor + masked_target_predictions # Calculates the targets for actions taken.
        actions_taken_targets = actions_taken_targets.view(self.batch_size, 1)

        td_error = actions_taken_targets - predictions # Computes the temporal difference error.
        loss = torch.sum(td_error ** 2) # Computes the loss as the sum of squared TD errors.

        loss.backward() # Performs backpropagation.
        self.optimizer.step() # Updates the network weights.

        # Update target estimator.
        # Updates the target network parameters.
        for target_param, param in zip(self.target_estimator.parameters(), self.estimator.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.target_copy_factor) + param.data * self.target_copy_factor)

        return td_error.cpu().detach().numpy()
