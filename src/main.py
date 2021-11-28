"""
Title: Source code for double deep Q-learning using Pytorch and torchvision.
Author: Christoph Metzner
"""

import sys
import keras
import numpy as np
from src.DDQN import DDQNAgent, set_weights_3D_2D
from typing import List, Tuple

def platform(dimensions, size):
    dim_left = (dimensions/2) - size
    dim_right = (dimensions/2) - (size - 1)
    
    platform = [dim_left, dim_right]
    return (platform)

def smell_range(dimensions, size, spatial_cue):
    
    dim_left = (dimensions/2) - (size * spatial_cue)
    dim_right = (dimensions/2) - ((size - 1) * spatial_cue)
    
    smell_range = [dim_left, dim_right]
    return(smell_range)

def center_location(platform: List[int]) -> int:
    """
    Parameters
    ----------
    platform : List[int]
       Input of the lower and upper limits of the platform

    Returns
    -------
    center_loc : int
        A single point location for the center of the cube/platorm
    """
    center = (platform[1] - platform[0]) / 2
    return center


def manhatten_dis_center(state: List[int], center: int) -> float:
    """
    Parameters
    ----------
    state : List[int]
        The location of the state given by [row, column, z]
    center : int
        The center of the platform or cube

    Returns
    -------
    man_dis : float 
        The manhatten distance from the current state location and the
        center of the cube (3-D) or platform (2-D)
    """
    
    # iterator
    i = 0
    # manhatten distance total
    man_dis = 0
    while i < len(state):
        man_dis = man_dis + abs(state[i] - center)
        
        # iterate
        i = i + 1
    return man_dis


def get_reward(state: List[int], next_state: List[int], platform: List[int], smell_range: List[int], Rewards: List[float]) -> (float, List[float]):
    """
    Parameters
    ----------
    state : List[int]
        The location of the state given by [row, column] or [row, column, z]
    next_state : List[int]
        The location of the next state given by [row, column] or [row, column, z]
    platform : List[int]
        Input of the lower and upper limits of the platform
    smell_range : List[int]
        The lower and upper limits of the smell range
    Rewards : List[float]
        A list of rewards that get updated after every step taken
        Rewards[0] = R_move
            starting value = -0.01
        Rewards[1] = R_close_cheese
            starting value = +2
    Returns
    -------
    Reward : float
        Returns the reward value
    Rewards : List[float]
        Returns an updated list of rewards for every step taken
        Rewards[0] = R_move
            starting value = -0.01
        Rewards[1] = R_close cheese
            starting value = +2
    """
    
    # update R_move == Rewards[0]
    Rewards[0] = Rewards[0] - 0.01
    
    
    # check for 2-D and for 3-D
    if (len(state) == 2):
        #################### 2-Dimensional######################
        # check to see if next_state is in the terminal location
        if (platform[0] <= next_state[0] <= platform[1]): # check to see if we are in the y (rows) of the platform
            if (platform[0] <= next_state[1] <= platform[1]): # check to see if we are in the x (columns) of the platform
                    return 100 # reward for getting to the platform
    
        # check to see if the next_state is in the smell range
        elif (smell_range[0] <= next_state[0] <= smell_range[1]): # check to see if we are in the y (rows) of the smell_range
            if (smell_range[0] <= next_state[1] <= smell_range[1]): # check to see if we are in the x (columns) of the smell_range
                    center = center_location(platform)
                    
                    # calculating manahtten distances
                    state_dis = manhatten_dis_center(state, center)
                    next_state_dis = manhatten_dis_center(next_state, center)
                    
                    # check to see if manhatten distance is smaller
                    if (next_state_dis < state_dis):
                        # update the R_close_cheese value
                        Rewards[1] = Rewards[1] + 2
                        return Rewards[1] + Rewards[0], Rewards
                    # check to see if manhatten distance is larger
                    elif (next_state_dis > state_dis):
                        # update the R_close_cheese_value
                        Rewards[1] = Rewards[1] - 2
                        # Rewards[0] here since we are moving away from the platform
                        return Rewards[0], Rewards
                    else:
                        return Rewards[1] + Rewards[0], Rewards
        else:
            # if not in the cube or in the smell range
            # the reward is just for swimming
            return Rewards[0], Rewards
        
        
    else:
        #################### 3-Dimensional######################
        # check to see if next_state is in the terminal location
        if (platform[0] <= next_state[0] <= platform[1]): # check to see if we are in the y (rows) of the platform/cube
            if (platform[0] <= next_state[1] <= platform[1]): # check to see if we are in the x (columns) of the platform/cube
                if(platform[0] <= next_state[2] <= platform[1]): # check to see if we are in the z-depth of the platform/cube
                    return 100 # reward for getting to the cube or the platform
    
        # check to see if the next_state is in the smell range
        elif (smell_range[0] <= next_state[0] <= smell_range[1]): # check to see if we are in the y (rows) of the smell_range
            if (smell_range[0] <= next_state[1] <= smell_range[1]): # check to see if we are in the x (columns) of the smell_range
                if(smell_range[0] <= next_state[2] <= smell_range[1]): # check to see if we are in the z-depth of the smell_range
                    # reward for getting in the smell range and it increases with subsequent closeness to the platform
                    
                    # calc the center
                    center = center_location(platform)
                    
                    # calculating manahtten distances
                    state_dis = manhatten_dis_center(state, center)
                    next_state_dis = manhatten_dis_center(next_state, center)
                    
                    # check to see if manhatten distance is smaller
                    if (next_state_dis < state_dis):
                        # update the R_close_cheese value
                        Rewards[1] = Rewards[1] + 2
                        return Rewards[1] + Rewards[0], Rewards
                    # check to see if manhatten distance is larger
                    elif (next_state_dis > state_dis):
                        # update the R_close_cheese_value
                        Rewards[1] = Rewards[1] - 2
                        # Rewards[0] here since we are moving away from the cube
                        return Rewards[0], Rewards
                    else:
                        return Rewards[1] + Rewards[0], Rewards
        else:
            # if not in the cube or in the smell range
            # the rewards is just for swimming
            return Rewards[0], Rewards


def get_next_state(state: List[int, ], action: int, dim: int, pos_pf: List[int]) -> Tuple[List[int, ]]:
    """

    Parameters
    ----------
    state: List[int]
        Containing the x, y, and/or z-positions of the agent in environment.
    action: int
        The choice the agent made in the direction to move
        2D has 4 actions (north: 0, south: 1, east: 2, west: 3)
        3D has 6 actions (north: 0, south: 1, east: 2, west: 3, up: 4, down: 5)
    dim: int
        Length of the dimensions for the square grid
    pos_pf: List[int]
        containing the corner positions of the platform

    Returns
    -------

    """
    next_state = state.copy()

    # move North (aka up or up a row on the grid)
    if action == 0:
        if state[0] == 0:
            return state, False
        else:
            next_state[0] = state[0] - 1

    # move South (aka down or down a row on the grid)
    elif action == 1:
        if state[0] == dim - 1:
            return state, False
        else:
            next_state[0] = state[0] + 1

    # move East (aka right or right a column on the grid)
    elif action == 2:
        if state[1] == dim - 1:
            return state, False
        else:
            next_state[1] = state[1] + 1

    # move West (aka left or left a column on the grid)
    elif action == 3:
        if state[1] == 0:
            return state, False
        else:
            next_state[1] = state[1] - 1

    # move up to the surface (aka up the cube)
    elif action == 4:
        if state[2] == 0:
            return state, False
        else:
            next_state[2] = state[2] - 1

    # move down into the pool (aka down in the cube)
    elif action == 5:
        if state[2] == dim - 1:
            return state, False
        else:
            next_state[2] = state[2] + 1

    terminal = get_terminal(next_state=next_state, pos_pf=pos_pf)
    if terminal:
        return state, True
    else:
        return next_state, False


def get_terminal(next_state: List[int], pos_pf: List[int]) -> bool:
    """
    This function checks whether agent is in a terminal state or not.

    Parameters
    ----------
    next_state: List[int]
        List containing x, y, and/or z-positions of agent (rat) for next state.
    pos_pf: List[int]
        List containing x1, x2, y1, y2, and/or z1, z2: Shape of platform in 2D is a quadrant and in 3D a cube.

    Returns
    -------
    bool
        Expression indicating whether episode is in a terminal state or not.
    """
    pos_x_agent = next_state[0]
    pos_y_agent = next_state[1]

    if len(next_state) == 3:
        pos_z_agent = next_state[2]

    # The following if-statement structure checks whether the agent is indeed on the platform and found the goal(cheese)
    # Description: 2D                 3D adding z-axis
    # x1y2  --------- x2y2              ---------
    #       |       |                   |       |
    #       |       |                   |       |
    #       |       |                   |       |
    # x1y1  --------- x2y1        x1y1z1--------- x1y1z2

    # Check whether 2D or 3D platform:
    pos_x1_pf = pos_pf[0]
    pos_x2_pf = pos_pf[1]
    pos_y1_pf = pos_pf[2]
    pos_y2_pf = pos_pf[3]

    if len(pos_pf) == 4:
        if pos_x1_pf <= pos_x_agent <= pos_x2_pf and pos_y1_pf <= pos_y_agent <= pos_y2_pf:
            return True
        else:
            return False

    elif len(pos_pf) == 6:
        pos_z1_pf = pos_pf[4]
        pos_z2_pf = pos_pf[5]
        if pos_x1_pf <= pos_x_agent <= pos_x2_pf \
                and pos_y1_pf <= pos_y_agent <= pos_y2_pf \
                and pos_z1_pf <= pos_z_agent <= pos_z2_pf:
            return True
        else:
            return False


def init_env(input_dims: int, dim2: bool, start_state: List[int] = None) -> Tuple[int, ]:
    # initialize first state of environment, i.e., position of agent in the maze
    # Randomized for one group of agents (rats) or fixed for another group of agents
    # consider dimension of input: 2D or 3D
    if start_state is None:

        # Here select a starting position that is random in maze
        x = np.random.randint(0, input_dims, 1)
        y = np.random.randint(0, input_dims, 1)
        z = np.random.randint(0, input_dims, 1)

        if dim2:
            state = [x, y]
        else:
            state = [x, y, z]

        # Here select a starting position in maze that is fixed
    else:
        state = start_state
    return state


def execute_learning(input_dims, n, dim2=True, start_state = None):
    terminal = False
    max_iterCnt = 1000
    iterCnt = 0

    # init agent:
    if dim2:
        ddqn_agent = DDQNAgent(alpha=0.0005,
                               gamma=0.99,
                               n_actions=8,
                               epsilon=1.0,
                               batch_size=64,
                               input_dims=input_dims,
                               fname='ddqn_model_2D.h5')

    else:  # init ddqn_agent in 3D with the x and y weights of the 2D agent
        ddqn_agent2D = keras.models.load_model('ddqn_model_2D.h5')

        ddqn_agent = DDQNAgent(alpha=0.0005,
                               gamma=0.99,
                               n_actions=26,
                               epsilon=1.0,
                               batch_size=64,
                               input_dims=input_dims,
                               fname='ddqn_model_3D.h5')

        # Setting the weights of agent trained in 2D equal to the untrained weights of agent in 3D
        ddqn_agent.q_val = set_weights_3D_2D(ddqn_agent2D.q_eval, ddqn_agent.q_eval)
        ddqn_agent.q_target = set_weights_3D_2D(ddqn_agent2D.q_target, ddqn_agent.q_target)

    G_history = []
    eps_history = []
    for episode in range(n):
        terminal = False  # set terminal flag to false, to indicate that agent is not a terminal state
        G = 0  # set expected return to zero

        # Initialize first starting state of the agent
        # two groups of agents: 1) fixed starting point and 2) random starting point
        state = init_env(input_dims=input_dims, dim2=dim2, start_state=start_state)

        while not terminal and iterCnt < max_iterCnt:
            action = ddqn_agent.choose_action(state=state)
            next_state, terminal = get_next_state(state=state, action=action)
            reward = get_reward(state=state, action=action, next_state=next_state)

            # Store current transition in memory buffer
            ddqn_agent.remember(state=state, action=action, reward=reward, next_state=next_state, terminal=terminal)

            # Update next state to become state for next iteration of loop
            state = next_state
            ddqn_agent.learn()
            # Add reward to
            G += reward


        G_history.append(G)
        eps_history.append(ddqn_agent.epsilon)

        avg_G = np.mean(G_history[max(0, episode-100):(episode+100)])
        print(f'Episode {episode+1}, G: {G}, Average G: {avg_G}')

        # save model
        if episode % 10 == 0 and episode > 0:
            ddqn_agent.save_model()

    return G_history, eps_history


def main(argv):
    input_dims = argv[1]  # input dimensions - 2 for 2D and 3 for 3D
    n = argv[2]  # number of episodes
    start_state = None  # or list for 2D [y_pos, x_pos] and 3D [y_pos, x_pos, z_pos]
    G_history, eps_history = execute_learning(input_dims=input_dims, n=n, start_state=None)

    # We can plot the two returned variables above.


if __name__ is '__main__':
    main(sys.argv)
