from unityagents import UnityEnvironment
import numpy as np 
from collections import deque
import matplotlib.pyplot as plt
import torch

from agent import Agent
from model import QNetwork
env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

agent = Agent(state_size, action_size, seed = 100)


def plot_result(score):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(score)), score)
    plt.ylabel('SCORE')
    plt.xlabel('EPISODE')
    plt.show()


def training(n_episodes=100,max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores=[]
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        done = False
        for _ in range(max_t):    
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score+=reward
            if done:
                break
        scores.append(score)
        scores_window.append(score)
        eps = max(eps_end, eps_decay*eps)
        n_mean = np.mean(scores_window)
        n_max = np.max(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, n_mean))
        if i_episode%100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tMax Score: {:.2f}'.format(i_episode, n_mean, n_max ))
        if(np.mean(scores_window)>=13):
            print('\nEnvironment Solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,n_mean))
            torch.save(agent.q_local.state_dict(), 'checkpoint.pth')
            break
    return scores


score = training()
plot_result(score)

while True:
    action = agent.act(action_size, 0.01)
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    score += reward
    state = next_state
    if done:
        break
print(score)
