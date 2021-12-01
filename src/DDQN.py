"""Title: Source code for double deep Q-learning Author: Christoph Metzner
Reference:
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/ddqn_keras.py """

from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

from typing import List, Tuple

# the following code was used from the reference above
class ReplayBuffer(object):
    """
    Class object that stores the transitions
    """

    def __init__(self, max_size: int, input_shape: List[int], n_actions: int):
        """
        Initialize ReplayBuffer object with maximum memory size, shape of input, and number of possible actions.

        Parameters
        ----------
        max_size: int
            Number indicating maximum size of memory, i.e,. maximum number of transitions stored in memory to learn from
        input_shape: List[int]
            Shape of state input of our environment
        n_actions: int
            Number of actions the agent could select in the environment/state
        """
        self.memory_size = max_size  # maximum number of observations in memory
        self.memory_cntr = 0  # index of last memory saved
        self.state_memory = np.zeros((self.memory_size, input_shape))  # init memory of all visited states
        self.next_state_memory = np.zeros((self.memory_size, input_shape))  # init memory of all visited next states

        # actions are later used as indices
        self.action_memory = np.zeros((self.memory_size, n_actions), dtype=np.int8)  #
        self.reward_memory = np.zeros(self.memory_size)  # store all received rewards

        # used to not sample the terminal state in your state_memory array
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

    def store_transition(self,
                         state: List[int],
                         action: int,
                         reward: float,
                         next_state: List[int],
                         terminal: bool):
        """
        Function that stores the most recently visited transition (state, action, reward, next_stat) and whether next
        state was terminal or not.

        Parameters
        ----------
        state: Tuple[int]
            A list containing the x,y,z-positions of the agent (2D: x and y / 3D: x, y, and z) as the state
        action: int
            Integer indicating which action was selected
        reward: float
            Received reward for state-action-pair
        next_state: Tuple[int]
            A list containing the x, y, and z-positions of the agent as the next state of the environment (2D: x and y /
            3D: x, y, and z)
        terminal: bool
            A boolean expression that indicates whether the next state was terminal or not
        """
        index = self.memory_cntr % self.memory_size  # used to keep memory finite; % used to identify current index
        self.state_memory[index] = state  # store state at current index of memory
        self.next_state_memory[index] = next_state  # store next state at current index memory

        # one-hot-encoding for discrete actions
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1.0
        self.action_memory[index] = actions  # store selected at current index memory
        self.reward_memory[index] = reward  # store received reward at current index memory

        # If terminal == true --> The action-value of terminal state is 0, however, int(terminal=true) == 1, therefore,
        # we need to subtract 1 - int(terminal) to make sure that we actually multiply the update by 0 (Updating Q!).
        self.terminal_memory[index] = 1 - terminal  # store if next state was terminal or not
        self.memory_cntr += 1

    def sample_buffer(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Function that samples memorized transitions from the ReplayBuffer object for the given batch_size.

        Parameters
        ----------
        batch_size: int
            Size of the batch used for learning

        Returns
        -------
        states: np.ndarray
            Containing all sampled states of respective transitions
        actions: np.ndarray
            Containing all sampled actions of respective transitions
        rewards: np.ndarray
            Containing all sampled rewards of respective transitions
        next_states: np.ndarray
            Containing all sampled next states of respective transitions
        terminals: np.ndarray
            Containing all sampled information if next state for respective transition was terminal or not
        """
        # get range of memory that is already filled with previous transitions
        max_memory = min(self.memory_cntr, self.memory_size)  # avoid sampling empty indexes
        # Create batch of randomly selected past transitions, do not sample the same transition again
        batch = np.random.choice(max_memory, batch_size, replace=False)

        # fill the batch with states, new_states, actions, rewards, information about the terminal state
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminals


def build_DQN(lr: float,
              n_actions: int,
              input_dims: List[int],
              fc1_dims: int,
              fc2_dims: int) -> Sequential:
    """
    Function that constructs a sequential model using the library keras

    Parameters
    ----------
    lr: float
        Learning rate of the selected optimizer
    n_actions: int
        Number of actions the agent could select in the environment/state
    input_dims: List[int]
        Shape of state input of our environment
    fc1_dims: int
        Number of nodes in first fully connected layer
    fc2_dims: int
        Number of nodes in second fully connected layer

    Returns
    -------
    Sequential
        Keras sequential model - deep neural network
    """
    model = Sequential([
        Dense(fc1_dims, input_shape=(input_dims,)),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(n_actions)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return model


class DDQNAgent(object):
    def __init__(self,
                 alpha: float,
                 gamma: float,
                 n_actions: int,
                 epsilon: float,
                 batch_size: int,
                 input_dims: Tuple[int,],
                 fname: str,
                 epsilon_dec: float = 0.996,
                 epsilon_min: float = 0.01,
                 memory_size: int = 1000000,
                 replace_target: int = 100):

        """
        Initialize double deep Q-learning agent object with parameters

        Parameters
        ----------
        alpha: float
            step size of the temporal difference update
        gamma: float
            discount factor to discount future rewards
        n_actions: int
            Number of actions the agent could select in the environment/state
        epsilon: float
            Variable indicating amount of exploration
        batch_size: int
            Size of the batch used for learning
        input_dims: Tuple[int, ]
            Shape of state input of our environment
        fname: str
            File name of saved model - 'ddqn_model_*D.h5' - replace * with 2 or 3 for dimensions of environment
        epsilon_dec: float
            Value of epsilon at start of learning
        epsilon_min:
            Value of minimum exploration
        memory_size: int
            Number indicating maximum size of memory, i.e,. maximum number of transitions stored in memory to learn from
        replace_target: int
            Variable indicating schedule of updating weights of target network
        """

        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(
            max_size=memory_size,
            input_shape=input_dims,
            n_actions=n_actions)

        # First network
        self.q_eval = build_DQN(
            lr=alpha,
            n_actions=n_actions,
            input_dims=input_dims,
            fc1_dims=256,
            fc2_dims=256)

        # Second network
        self.q_target = build_DQN(
            lr=alpha,
            n_actions=n_actions,
            input_dims=input_dims,
            fc1_dims=256,
            fc2_dims=256)

    def remember(self,
                 state: List[int],
                 action: int,
                 reward: float,
                 next_state: List[int],
                 terminal: bool):
        """
        Function that stores the visited transitions in the memory buffer.

        Parameters
        ----------
        state: List[int]
            A list containing the x,y,z-positions of the agent (2D: x and y / 3D: x, y, and z) as the state
        action: int
            Integer indicating which action was selected
        reward: float
            Received reward for state-action-pair
        next_state: List[int]
            A list containing the x, y, and z-positions of the agent as the next state of the environment (2D: x and y /
            3D: x, y, and z)
        terminal: bool
            A boolean expression that indicates whether the next state was terminal or not
        """

        # store transition in memory
        self.memory.store_transition(state, action, reward, next_state, terminal)

    def choose_action(self, state: List[int]) -> int:
        """

        Parameters
        ----------
        state: List[int]
            A list containing the x,y,z-positions of the agent (2D: x and y / 3D: x, y, and z) as the state

        Returns
        -------
        int
            selected action for given state
        """
        state = np.array(state)[np.newaxis]
        rand = np.random.random()  # select from uniform distribution
        # if random below epsilon do exploration else do exploitation
        if rand < self.epsilon:  # do exploration
            action = np.random.choice(self.action_space)  # select action based on uniform distribution
        else:  # do exploitation
            action_values = self.q_eval.predict(state)  # compute action_values for the current state
            action = np.argmax(action_values)  # select action that maximizes action_value

        return action

    def learn(self):
        # make sure that learning occurs only if the memory is larger than the batch size
        # avoids sampling the same transition over and over again
        if self.memory.memory_cntr > self.batch_size:
            states, actions, rewards, next_states, terminals = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(actions, action_values)

            # The following code handles the updating
            # Get Q-value for target network: Q(s', argmax a Q(s', a; Phi(t)); Phi(t-))
            # Step 1: Compute all possible q-values using target network
            q_next = self.q_target.predict(next_states)

            # Step 2: Compute all possible q-values for Q(s', a; Phi(t)) using evaluation network
            q_eval = self.q_eval.predict(next_states)

            # Step 3: Select actions that maximize q-values for all transitions in batch
            max_actions = np.argmax(q_eval, axis=1)

            # Step 4: Initialize index of batch
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            # Step 5: Compute the action-values for the current states
            q_pred = self.q_eval.predict(states)

            # Step 6: Set q_target = q_pred to set the error term 0 later on.
            # When fitting the network, we want the differences for the off-actions (i.e., actions not used in this
            # transition) to be 0. Only update weights for the actions that were actually used.
            q_target = q_pred

            # Step 7: Compute the TD-Target for all transitions in batch
            # TD-Target: Y_t^DDQN = R_t+1 + gamma * (Q(S_t+1, argmax_a, a; w_t), w_t-)
            q_target[batch_index, action_indices] = rewards \
                                                    + self.gamma * q_next[
                                                        batch_index, max_actions.astype(int)] * terminals

            # Step 8: Update Weights in evaluation network
            # Phi(t+1) = Phi(t) + alpha(TD-Target - Q(S_t, A_t, Phi(t))*GradientPhi(t)Q(S_t, A_t, Phi_t)
            _ = self.q_eval.fit(states, q_target, verbose=0)  # verbose = 0 silent fitting

            # Gradually reduce exploration term epsilon
            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

            # Update weights of target network by setting the target networks equal to weights of evaluation network
            if self.memory.memory_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        """
        Function that sets weights of target network equal to weights of evaluation network.
        """
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        """
        Function that saves the model, i.e., the double deep q-learning agent.
        """
        self.q_eval.save(self.model_file)

    def load_model(self):
        """
        Function that loads a saved model.
        """
        self.q_eval = load_model(self.model_file)

        if self.epsilon <= self.epsilon_min:
            self.update_network_parameters()

#written by us
def set_weights_3D_2D(model_2D, model_3D):
    weights_2D = model_2D.get_weights()
    weights_3D = model_3D.get_weights()

    # index weights
    # 0: input to first dense layer
    # 1: first dense layer to first activation layer
    # 2: activation layer ot second dense layer
    # 3: second dense layer to second activation layer
    # 4: second activation layer ot dense output layer
    weights_3D[0][0] = weights_2D[0][0]  # weights for y-axis
    weights_3D[0][1] = weights_3D[0][1]  # weights for x-axis
    weights_3D[0][2] = np.zeros(256)  # initialize weights for z-axis with zeros

    weights_3D[1] = weights_2D[1]
    weights_3D[2] = weights_2D[2]
    weights_3D[3] = weights_2D[3]
    weights_3D[4][:, :4] = weights_2D[4]   # set the first weights for the first four agents equal
    #### think about the last layer

    model_3D.set_weights(weights_3D)

    return model_3D

