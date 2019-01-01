import pandas as pd
import numpy as np
import argparse

from config import MEMORY_CAPACITY, TEST_EPISODE_NUM, BATCH_SIZE, E_GREEDY, MAX_STEP

# Add import path and import the lf2gym
import os, sys
sys.path.append(os.path.abspath('lf2gym'))
# Import lf2gym
import lf2gym

def transObser(observation, feature, mode):
    if mode == 'picture':
        observation = np.transpose(observation, (2, 1, 0))
        observation = np.transpose(observation, (0, 2, 1))
    elif mode == 'feature':
        observation = feature
    elif mode == 'mix':
        observation_ = np.transpose(observation, (2, 1, 0))
        observation_ = np.transpose(observation_, (0, 2, 1))
        observation = [observation_, feature]
    return observation

def test(mothod, mode, model_path):
    #load model
    RL.load_model(model_path)

    steps, rewards = [], []
    for episode in range(TEST_EPISODE_NUM):
        # initial
        observation, characters_info = env.reset()
        observation = transObser(observation, characters_info, mode)

        iter_cnt, total_reward = 0, 0
        while True:
            iter_cnt += 1

            # fresh env
            # env.render()

            if algorithm == 'DQN':
                # RL choose action based on observation
                action = RL.choose_action(observation)
                # RL take action and get next observation and reward
                observation_, reward, done, _, characters_info = env.step(action)
                observation_ = transObser(observation_, characters_info, mode)

            # accumulate reward
            total_reward += reward
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                total_reward = round(total_reward, 2)
                steps.append(iter_cnt)
                rewards.append(total_reward)
                print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, iter_cnt, total_reward))
                break

    # end of game
    print('--------------------------------')
    print('After {} episode,\nthe average step is {},\nthe average reward is {}.'.format(TEST_EPISODE_NUM, sum(steps)/len(steps), sum(rewards)/len(rewards)))
    env.close()

if __name__ == "__main__":

    # argument
    parse = argparse.ArgumentParser()
    parse.add_argument('-m', '--mode',
                        default='mix',
                        help='Choose input for network (picture, feature, mix)')
    parse.add_argument('-alog', '--algorithm',
                        default='DQN',
                        help='Choose which rl algorithm used (DQN)')
    parse.add_argument('-t', '--test',
                        default='model/DQN/Dennis_CNN80190LSTMA3C_100episode_0dmp/mix_eval_0.01_{}_{}.pkl'.format(E_GREEDY, BATCH_SIZE),
                        help='The test model path')
    args = parse.parse_args()
    
    # import network model
    if args.mode == 'picture':
        from brain.pbrain import DQN
    elif args.mode == 'feature':
        from brain.fbrain import DQN
    elif args.mode == 'mix':
        from brain.mbrain_wLSTM_A3C import DQN
    else:
        print("Error mode! Use picture instead.")
        from brain.pbrain import DQN
    
    # game setup
    AGENT = 'Dennis'
    OPPOENENT = 'Firen'

    # env setup
    env = lf2gym.make(startServer=True, wrap='skip4', driverType=lf2gym.WebDriver.Chrome, debug=True,
        characters=[lf2gym.Character[AGENT], lf2gym.Character[OPPOENENT]], 
        difficulty=lf2gym.Difficulty.Crusher,
        action_options=['Basic', 'Advanced', 'AJD', 'Full Combos'],
        # rewardList=['hp', 'mp'])
        rewardList=['hp'])
    
    options = env.get_reset_options()
    print('Original reset options: %s' % options)
    options['hp_full'] = 100
    options['mp_start'] = 250
    print('Custom reset options: %s' % options)

    # initial obervation number
    mode = args.mode
    if mode == 'picture':
        state_n = env.observation_space.n
    elif mode == 'feature':
        # state_n = 28
        state_n = 32
    elif mode == 'mix':
        # state_n = [env.observation_space.n, 28]
        state_n = [env.observation_space.n, 32]
    else:
        state_n = env.observation_space.n
        mode = 'picture'

    # algorithm setup
    algorithm = args.algorithm
    if algorithm == 'DQN':
        print("Use DQN...")
        print('--------------------------------')
        env_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # to confirm the shape
        RL = DQN(action_n=env.action_space.n, state_n=state_n, env_shape=env_shape)
    else:
        print("Error algorithm! Use DQN instead.")
        print('--------------------------------')
        env_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # to confirm the shape
        RL = DQN(action_n=env.action_space.n, state_n=state_n, env_shape=env_shape)
        algorithm = 'DQN'

    test(algorithm, mode, args.test)