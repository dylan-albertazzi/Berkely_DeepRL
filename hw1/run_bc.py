#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20
    python run_bc.py expert_data/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 10
    python run_behavioral_cloning.py expert_data/Ant-v2.pkl Ant-v2 --num_rollouts 10
    python run_behavioral_cloning.py expert_data/Hopper-v2.pkl Hopper-v2 --num_rollouts 10
    python run_behavioral_cloning.py expert_data/Reacher-v2.pkl Reacher-v2 --num_rollouts 10
    python run_behavioral_cloning.py expert_data/Walker2d-v2.pkl Walker2d-v2 --num_rollouts 10
Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


def init_tf_sess():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return sess


def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.loads(f.read())


def build_model(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):

    with tf.variable_scope(scope):

        layer = input_placeholder

        for i in range(0, n_layers):
            layer = tf.layers.dense(layer, size, activation=activation)

        output_placeholder = tf.layers.dense(layer, output_size, activation=output_activation)

    return output_placeholder


def placeholders(input_size, output_size):

    input_ph = tf.placeholder(name='input', shape=[None, input_size], dtype=tf.float32)
    output_ph = tf.placeholder(name='output', shape=[None, output_size], dtype=tf.float32)

    return [input_ph, output_ph]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert dataset')
    model = torch.load('myFirstNet.pt')

    
    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        if i % 10 == 0:
            print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.forward(obs.reshape(1, obs.shape[0]))
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


if __name__ == '__main__':
    main()