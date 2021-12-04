"""
Title: Source code for double deep Q-learning using Pytorch and torchvision.
Author: Christoph Metzner and Andrew Strick
Date: 12/3/2021

This is the main source code for running the morris water maze experiment. The whole code in this script was developed
by either Andrew Strick or Christoph Metzner.
"""

import sys
import keras
import numpy as np
from DDQN import DDQNAgent, set_weights_3D_2D
from typing import List, Tuple, Union
from typing import TypeVar
import matplotlib.pyplot as plt
import pickle
import time
keras_sequential_model = TypeVar('keras.engine.sequential.Sequential')

def platform(dimensions: int, size: int) -> List[int]:
    """
    Parameters
    ----------
    dimensions : int
        Length of the dimensions for the square or cubed environment
    size : int
        The size of the platform dimensions

    Returns
    -------
    platform : List[int]
        The lower followed by the upper index locations of the platform in a list
        platform[0] == Lower index number of platform
        platform[1] == Upper index number of platform
    """
    dim_left = (dimensions/2) - size
    dim_right = (dimensions/2) - (size - 1)
    
    platform = [dim_left, dim_right]
    return platform


def smell_range(dimensions: int, size: int, spatial_cue: int) -> List[int]:
    """
    Parameters
    ----------
    dimensions : int
        Length of the dimensions for the square or cubed environment
    size : int
        The size of the platform dimensions
    spatial_cue : int
        The coeffient multiply of how big the lower and upper limits of the 
        smell range will be.

    Returns
    -------
    smell_range : List[int]
        The lower followed by the upper index locations of the smell-range in a list
        smell_range[0] == Lower index number of smell_range
        smell_range[1] == Upper index number of smell_range
    """
    dim_left = (dimensions/2) - (size * spatial_cue)
    dim_right = (dimensions/2) - (size * spatial_cue - 1)
    
    smell_range = [dim_left, dim_right]
    return smell_range


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


def get_reward(state: List[int], next_state: List[int], platform: List[int], smell_range: List[int],
               Rewards: List[float]) -> (float, List[float]):
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
            starting value = 0
        Rewards[1] = R_close_cheese
            starting value = +2
    Returns
    -------
    Reward : float
        Returns the reward value
    Rewards : List[float]
        Returns an updated list of rewards for every step taken
        Rewards[0] = R_move
            starting value = 0
        Rewards[1] = R_close cheese
            starting value = +2
    """

    # update R_move == Rewards[0]
    Rewards[0] = Rewards[0] - 0.01

    # check for 2-D and for 3-D

    #################### 2-Dimensional######################
    if len(state) == 2:
        # check to see if next_state is in the terminal location
        #  checks if agent is in y range                    checks if agent is in x range
        if platform[0] <= next_state[0] <= platform[1] and platform[0] <= next_state[1] <= platform[1]:
            return 500 + Rewards[0], Rewards  # reward for getting to the platform

        # check to see if the next_state is in the smell range
        # check if agent is in y range                              check if agent is in x range
        elif smell_range[0] <= next_state[0] <= smell_range[1] and smell_range[0] <= next_state[1] <= smell_range[1]:
                center = center_location(platform)

                # calculating manhatten distances
                state_dis = manhatten_dis_center(state, center)
                next_state_dis = manhatten_dis_center(next_state, center)

                # check to see if manhatten distance is smaller
                if next_state_dis < state_dis:
                    # update the R_close_cheese value
                    # including a cap since, sometimes the agent moved around in area and gradually increasing
                    # the reward on purpose
                    
                    # Rewards [1] will increase only if the agent is getting closer to the target platform or cube
                    # Rewards [1] will linearly increase by a factor of two for each step closer the agent gets
                    # to the target platform or cube.
                    Rewards[1] = ((smell_range[1] - center) - next_state_dis) * 2
                    return Rewards[0] + Rewards[1], Rewards
                # check to see if manhatten distance is larger
                elif next_state_dis > state_dis:
                    # update the R_close_cheese_value
                    # some issue with computing the reward properly cause the agent to received large negative rewards
                    #Rewards[1] = max(Rewards[1] - 2, 0)

                    # Rewards[0] here since we are moving away from the platform
                    return Rewards[0] + Rewards[1], Rewards
                elif next_state_dis == state_dis:
                    return Rewards[0] + Rewards[1], Rewards
        else:
            # if not in the cube or in the smell range
            # the reward is just for swimming
            return Rewards[0], Rewards

    #################### 3-Dimensional######################
    elif len(state) == 3:
        # check to see if next_state is in the terminal location
        if platform[0] <= next_state[0] <= platform[1] \
            and platform[0] <= next_state[1] <= platform[1] \
            and platform[0] <= next_state[2] <= platform[1]:
                return 500, Rewards  # reward for getting to the cube or the platform

        # check to see if the next_state is in the smell range
        elif smell_range[0] <= next_state[0] <= smell_range[1] \
                and smell_range[0] <= next_state[1] <= smell_range[1] \
                and smell_range[0] <= next_state[2] <= smell_range[1]:
                    # calc the center
                    center = center_location(platform)

                    # calculating manahtten distances
                    state_dis = manhatten_dis_center(state, center)
                    next_state_dis = manhatten_dis_center(next_state, center)

                    # check to see if manhatten distance is smaller
                    if next_state_dis < state_dis:
                        # update the R_close_cheese value
                        # including a cap since, sometimes the agent moved around in area and gradually increasing
                        # the reward on purpose
                        
                        # Rewards [1] will increase only if the agent is getting closer to the target platform or cube
                        # Rewards [1] will linearly increase by a factor of two for each step closer the agent gets
                        # to the target platform or cube.
                        Rewards[1] = ((smell_range[1] - center) - next_state_dis) * 2
                        return Rewards[0] + Rewards[1], Rewards
                    # check to see if manhatten distance is larger
                    elif next_state_dis > state_dis:
                        # update the R_close_cheese_value
                        Rewards[1] = max(Rewards[1] - 2, 0)
                        # Rewards[0] here since we are moving away from the cube
                        return Rewards[0], Rewards
                    else:
                        return Rewards[1] + Rewards[0], Rewards
        else:
            # if not in the cube or in the smell range
            # the rewards is just for swimming
            return Rewards[0], Rewards


def get_next_state(
        state: List[int],
        action: int,
        maze_dim: int,
        pos_pf: List[int]) -> Tuple[List[int], bool]:
    """
    Function that computes the next state (i.e., position of the agent) given current state and action of agent.
    Parameters
    ----------
    state: List[int]
        Containing the x, y, and/or z-positions of the agent in environment.
    action: int
        The choice the agent made in the direction to move
        2D has 4 actions (north: 0, south: 1, east: 2, west: 3)
        3D has 6 actions (north: 0, south: 1, east: 2, west: 3, up: 4, down: 5)
    maze_dim: int
        Length of the dimensions for the square or cubed environment
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
        if state[0] == maze_dim - 1:
            return state, False
        else:
            next_state[0] = state[0] + 1

    # move East (aka right or right a column on the grid)
    elif action == 2:
        if state[1] == maze_dim - 1:
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
        if state[2] == maze_dim - 1:
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

    # The following if-statement structure checks whether the agent is indeed on the platform and found the goal(cheese)
    # Description: 2D                 3D adding z-axis
    # x1y2  --------- x2y2              ---------
    #       |       |                   |       |
    #       |       |                   |       |
    #       |       |                   |       |
    # x1y1  --------- x2y1        x1y1z1--------- x1y1z2

    if len(next_state) == 2:
        if pos_pf[0] <= pos_x_agent <= pos_pf[1] and pos_pf[0] <= pos_y_agent <= pos_pf[1]:
            return True
        else:
            return False

    elif len(next_state) == 3:
        pos_z_agent = next_state[2]
        if pos_pf[0] <= pos_x_agent <= pos_pf[1] \
                and pos_pf[0] <= pos_y_agent <= pos_pf[1] \
                and pos_pf[0] <= pos_z_agent <= pos_pf[1]:
            return True
        else:
            return False


def init_starting_state(input_dim: int, maze_dim: int, start_state: List[int]) -> List[int]:
    # initialize first state of environment, i.e., position of agent in the maze
    # Randomized for one group of agents (rats) or fixed for another group of agents
    # consider dimension of input: 2D or 3D
    if start_state is None:

        # Here select a starting position that is random in maze
        x = np.random.randint(0, maze_dim)
        y = np.random.randint(0, maze_dim)
        z = np.random.randint(0, maze_dim)

        if input_dim == 2:
            state = [x, y]
        elif input_dim == 3:
            state = [x, y, z]

        # Here select a starting position in maze that is fixed
    else:
        state = start_state
    return state


def init_ddqn_agent(input_dims: int, start_state: Tuple[List[int], bool], maze_dim: int) -> keras_sequential_model:
    """
    Function to initialize double deep Q-learning agent.
    Parameters
    ----------
    input_dims: int
    start_state: Tuple[List[int], bool]

    Returns
    -------

    """
    if start_state is not None:
        start_s = 'fixed'
    else:
        start_s = 'random'
    if input_dims == 2:
        ddqn_agent = DDQNAgent(alpha=0.0005,
                               gamma=0.99,
                               n_actions=4,
                               epsilon=1.0,
                               batch_size=64,
                               input_dims=input_dims,
                               fname=f'ddqn_model_2D_{maze_dim}_{start_s}.h5')

    elif input_dims == 3:  # init ddqn_agent in 3D with the x and y weights of the 2D agent
        ddqn_agent2D = keras.models.load_model(f'ddqn_model_2D_{maze_dim}_{start_s}.h5')

        ddqn_agent = DDQNAgent(alpha=0.0005,
                               gamma=0.99,
                               n_actions=6,
                               epsilon=1.0,
                               batch_size=64,
                               input_dims=input_dims,
                               fname=f'ddqn_model_3D_{maze_dim}_{start_s}.h5')

        # Setting the weights of agent trained in 2D equal to the untrained weights of agent in 3D
        ddqn_agent.q_val = set_weights_3D_2D(ddqn_agent2D, ddqn_agent.q_eval)
        ddqn_agent.q_target = set_weights_3D_2D(ddqn_agent2D, ddqn_agent.q_target)

    return ddqn_agent


def execute_learning(
        input_dim: int,
        maze_dim: int,
        n: int,
        size_pf: int,
        spatial_cues: int,
        start_state: Union[List[int], None]) -> Tuple[List[float], List[float]]:

    """
    Function that executes learning of agent in environment.

    Parameters
    ----------
    input_dim: int
        Input dimension of environment - 2D squared or 3D cubed.
    maze_dim: int
        Maze dimensions with equal lengths in x, y, and z-direction.
    n: int
        Number of episodes for training.
    size_pf: int
        Size of platform.
    spatial_cues: int
        Factor that increases size of smell range surrounding platform
    start_state: Union[List[int], None]
        Starting state can either be given directly by the user or is random.

    Returns
    -------
    G_history: List[float]
        List containing all expected returns per episodes.
    eps_history: List[float]
        List containing the final epsilons of the agent.
    """

    # init DDQN-Agent
    ddqn_agent = init_ddqn_agent(input_dims=input_dim, start_state=start_state, maze_dim=maze_dim)

    # init empty lists to store results - expected returns and history of epsilon
    G_history = []
    eps_history = []
    states_visited = []

    pos_pf = platform(dimensions=maze_dim, size=size_pf)
    pos_smell = smell_range(dimensions=maze_dim, size=size_pf, spatial_cue=spatial_cues)

    for episode in range(n):
        max_iterCnt = 500
        terminal = False  # set terminal flag to false, to indicate that agent is not a terminal state
        iterCnt = 0
        G = 0  # set expected return to zero

        # Initialize first starting state of the agent
        # two groups of agents: 1) fixed starting point and 2) random starting point
        state = init_starting_state(input_dim=input_dim, maze_dim=maze_dim, start_state=start_state)
        Rewards = [0, 0]
        states_visited_episode = []
        while not terminal and iterCnt < max_iterCnt:
            #print(f'Current Move: {iterCnt}')
            action = ddqn_agent.choose_action(state=state)

            next_state, terminal = get_next_state(
                state=state,
                action=action,
                maze_dim=maze_dim,
                pos_pf=pos_pf)

            states_visited_episode.append(next_state)
            if iterCnt == max_iterCnt-1:
                reward = -1000
            else:
                reward, Rewards = get_reward(
                    state=state,
                    next_state=next_state,
                    platform=pos_pf,
                    smell_range=pos_smell,
                    Rewards=Rewards)

            # Store current transition in memory buffer
            ddqn_agent.remember(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminal=terminal)

            # Update next state to become state for next iteration of loop
            state = next_state
            ddqn_agent.learn()
            # Add reward to
            G += reward
            iterCnt += 1

        G_history.append(G)
        eps_history.append(ddqn_agent.epsilon)

        avg_G = np.mean(G_history[max(0, episode-100):(episode+100)])
        print(f'Episode {episode+1}, G: {G}, Average G: {avg_G}')

        # save model
        if episode % 10 == 0 and episode > 0:
            ddqn_agent.save_model()

    states_visited.append(states_visited_episode)

    return G_history, eps_history, states_visited


def plot_total_returns(total_returns: List[int], exp_num: int):
    x = np.linspace(1, len(total_returns), len(total_returns))

    plt.figure(figsize=(10, 8))

    plt.plot(x, total_returns)
    plt.title(f'Total Return vs Epoch')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Total Return per Epoch', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)

    plt.savefig(f'Total Return vs Epoch {exp_num}.png')


def main(argv):
    n = int(argv[1])  # number of episodes
    input_dims = int(argv[2])  # input dimensions - 2 for 2D and 3 for 3D
    maze_dims = int(argv[3])  # input dimensions of squared or cubed environment - default 128
    size_pf = int(argv[4])  # size of platform where goal is
    spatial_cues = int(argv[5])  # smell range surrounding platform - factor of it
    fix_starting_state = int(argv[6])  # if 0: fixed elif 1: random

    if fix_starting_state == 0:
        if input_dims == 2:
            start_state = [0, 0]
        elif input_dims == 3:
            start_state = [0, 0, 0]

    elif fix_starting_state == 1:
        start_state = None
    begin = time.time()
    G_history, eps_history, states_visited = execute_learning(
        input_dim=input_dims,
        maze_dim=maze_dims,
        n=n,
        size_pf=size_pf,
        spatial_cues=spatial_cues,
        start_state=start_state)  # or list for 2D [y_pos, x_pos] and 3D [y_pos, x_pos, z_pos]
    end = time.time()

    print(f'Training time: {end-begin} seconds.')
    # save results for later use with pickle
    with open(f'results_expected_return_{argv[1]}_{argv[2]}_{argv[6]}.pkl', 'wb') as pickle_g:
        pickle.dump(G_history, pickle_g)

    with open(f'results_eps_{argv[1]}_{argv[2]}_{argv[6]}.pkl', 'wb') as pickle_eps:
        pickle.dump(eps_history, pickle_eps)

    with open(f'results_states_visited_{argv[1]}_{argv[2]}_{argv[6]}.pkl', 'wb') as pickle_states_visited:
        pickle.dump(states_visited, pickle_states_visited)

if __name__ == '__main__':
    simulations = [[0, 500, 2, 12, 2, 8, 0],
                   [0, 500, 2, 12, 2, 8, 1]]
    for sim in simulations:
        main(argv=sim)
