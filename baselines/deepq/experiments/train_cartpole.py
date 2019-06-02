import os, gym
from time import gmtime, strftime

import numpy
import tensorflow

from baselines import deepq


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    # Get xp_id
    xp_id = strftime("%Y-%m-%d.%H:%M:%S", gmtime())
    print('Experiment: '+xp_id)
    
    dir_to_save = os.path.join('.','save',xp_id)
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    path_to_save = os.path.join(dir_to_save,'cartpole_model.pkl')

    # Set randomness
    seed = 1
    numpy.random.seed(seed)
    tensorflow.set_random_seed(seed)

    env = gym.make("CartPole-v0")
    act = deepq.learn(
        env,
        network='mlp',
        seed = seed,
        lr=1e-3,
        total_timesteps=20000, #100000
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=1,
        callback=callback,
        checkpoint_freq=50, #10000
        checkpoint_path=dir_to_save,
        
    )
    
    print("Saving .pkl model to: ",path_to_save)
    act.save(path_to_save)


if __name__ == '__main__':
    main()
