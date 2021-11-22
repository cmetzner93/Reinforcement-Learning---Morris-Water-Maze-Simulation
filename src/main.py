"""
Title: Source code for double deep Q-learning using Pytorch and torchvision.
Author: Christoph Metzner
"""

import sys
import keras
import numpy as np
from src.DDQN import DDQNAgent
from typing import Tuple

def get_reward(state: Tuple[int, ], action: int, next_state: Tuple[int, ]) -> float:
    pass


def get_next_state(state: Tuple[int, ], action: int) -> Tuple[Tuple[int, ], bool]:
    """
    Returns
    -------

    """
    # Placeholder variables
    next_state = 0
    terminal = False
    return next_state, terminal


def init_env(start_pos_fixed: bool, input_dims) -> Tuple[int, ]:
    # initialize first state of environment, i.e., position of agent in the maze
    # Randomized for one group of agents (rats) or fixed for another group of agents
    # consider dimension of input: 2D or 3D
    if start_pos_fixed:
        # Here select a starting position in maze that is fixed
        pass
    else:
        # Here select a starting position that is random in maze
        pass

    # Placeholder variable
    state = 0
    return state


def execute_learning(input_dims, n):
    terminal = False
    max_iterCnt = 1000
    iterCnt = 0

    # init agent:
    if len(input_dims) == 2:
        ddqn_agent = DDQNAgent(alpha=0.0005,
                               gamma=0.99,
                               n_actions=8,
                               epsilon=1.0,
                               batch_size=64,
                               input_dims=input_dims,
                               fname='ddqn_model_2D.h5')

    elif len(input_dims) == 3:  # init ddqn_agent in 3D with the x and y weights of the 2D agent
        ddqn_agent2D = keras.models.load_model('ddqn_model_2D.h5')

        ddqn_agent = DDQNAgent(alpha=0.0005,
                               gamma=0.99,
                               n_actions=26,
                               epsilon=1.0,
                               batch_size=64,
                               input_dims=input_dims,
                               fname='ddqn_model_3D.h5')

        # Setting the weights of agent trained in 2D equal to the untrained weights of agent in 3D
        ddqn_agent.q_eval.set_weights(ddqn_agent2D.q_eval.get_weights())
        ddqn_agent.q_target.set_weights(ddqn_agent2D.q_target.get_weights())

    G_history = []
    eps_history = []
    for episode in range(n):
        terminal = False  # set terminal flag to false, to indicate that agent is not a terminal state
        G = 0  # set expected return to zero

        # Initialize first starting state of the agent
        # two groups of agents: 1) fixed starting point and 2) random starting point
        state = init_env(start_pos_fixed=True, input_dims=input_dims)
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

    G_history, eps_history = execute_learning(input_dims=input_dims, n=n)

    # We can plot the two returned variables above.


if __name__ is '__main__':
    main(sys.argv)
