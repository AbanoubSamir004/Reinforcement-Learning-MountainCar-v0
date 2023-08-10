import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import warnings
import pygame
import pyglet
import imageio
# Ignore all warnings
warnings.filterwarnings("ignore")

# Mountain Car enviroment  v0

def q_learning(env,max_episode_steps=200,n_games=2000,alpha= 0.1,gamma = 0.99):
    buckets=20
    env._max_episode_steps = max_episode_steps
    pos_space = np.linspace(-1.2, 0.6, buckets)
    vel_space = np.linspace(-0.07, 0.07, buckets)

    def get_state(observation):
        if isinstance(observation, tuple):
            pos, vel = observation[0]
        else:
            pos, vel = observation
        if isinstance(pos, np.ndarray):
            pos = pos[0]

        pos_bin = int(np.digitize(pos, pos_space))
        vel_bin = int(np.digitize(vel, vel_space))
        return (pos_bin, vel_bin)

    def max_action(Q, state, actions=[0, 1, 2]):
        values = np.array([Q[state,a] for a in actions])
        action = np.argmax(values)
        return action

    action_space = [0, 1, 2]

    states = []
    for pos in range(buckets+1):
        for vel in range(buckets+1):
            states.append((pos, vel))

    Q = {}
    for state in states:
        for action in action_space:
            Q[state, action] = 0

    score = 0
    total_rewards = np.zeros(n_games)
    total_time=np.zeros(n_games)
    reached=0
    for i in range(n_games):
        start_time= time.time()
        done = False
        obs = env.reset()
        env.state[0] = random.uniform(-0.6, -0.4)
        env.state[1] = 0
        state = get_state(obs)

        steps=1
        score = 0
        while True:
            action = max_action(Q, state)
            obs_, reward, done, _,info = env.step(action)
            state_ = get_state(obs_)
            score += reward
            action_ = max_action(Q, state_)

            Q[state, action] = Q[state, action] + \
                    alpha*(reward + gamma*Q[state_, action_] - Q[state, action])
            state = state_

            if (done):
                print("Goal Reached at Episode: ",i,'score ', score,"Observation:",obs_)
                reached+=1
                break
            if(steps==max_episode_steps):
                break
            steps+=1

        end_time = time.time()
        time_taken = end_time - start_time
        total_time[i]=time_taken
        total_rewards[i] = score

    print("The Agent Reached the Goals: ",reached,' Times')

    frames=[]
    env = gym.make('MountainCar-v0',render_mode='rgb_array')
    for episode in range(50):
        done = False
        observation = env.reset()
        state = get_state(observation)
        flag=True
        steps=1

        while not done:
            frames.append(env.render())
            action = max_action(Q, state)
            observation, reward, done, _,info = env.step(action)
            state = get_state(observation)
            if done:
                flag=False
            if(steps==max_episode_steps):
                break
            steps+=1
        if flag==False:
            env.close()
            break

    frame_duration = 1000 / max_episode_steps
    imageio.mimsave('Q-Learning_Game.gif', frames, duration=frame_duration)

    print("Done Q-Learning Algorithm",'\n')
    plt.plot(total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Q-Learning Rewards')
    plt.savefig('QLearning_mountaincar_Rewards.png')
    plt.close()
    return total_rewards,total_time

#############################################################################################################

def sarsa_algorithm(env,max_episode_steps=200,n_games=2000,alpha= 0.1,gamma = 0.99,epsilon = 1.0):
    env._max_episode_steps = max_episode_steps
    pos_space = np.linspace(-1.2, 0.6, 20)
    vel_space = np.linspace(-0.07, 0.07, 20)

    def get_state(observation):
        if isinstance(observation, tuple):
            pos, vel = observation[0]
        else:
            pos, vel = observation
        if isinstance(pos, np.ndarray):
            pos = pos[0]
        
        pos_bin = int(np.digitize(pos, pos_space))
        vel_bin = int(np.digitize(vel, vel_space))
        return (pos_bin, vel_bin)

    def epsilon_greedy_policy(Q, state,epsilon):
        if np.random.random() < epsilon:
            action = np.random.choice([0, 1, 2])
        else:
            action = np.argmax(Q[state])
        return action

    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos, vel))

    Q = np.zeros((21, 21, 3))

    score = 0
    reached=0
    total_rewards = np.zeros(n_games)
    total_time=np.zeros(n_games)
    for i in range(n_games):
        start_time= time.time()
        done = False
        obs = env.reset()
        env.state[0] = random.uniform(-0.6, -0.4)
        env.state[1] = 0
        state = get_state(obs)

        score = 0
        steps=1

        action = epsilon_greedy_policy(Q, state,epsilon)
        while True:
            obs_, reward, done, _, _ = env.step(action)
            state_ = get_state(obs_)
            score += reward
            action_ = epsilon_greedy_policy(Q, state_,epsilon)
            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * Q[state_][action_] - Q[state][action])
            state = state_
            action = action_
            if (done):
                print("Goal Reached at Episode: ",i,'score ', score,"Observation:",obs_ ,'epsilon %.3f' % epsilon )
                reached+=1
                break
            if(steps==max_episode_steps):
                break
            steps+=1

        end_time = time.time()
        time_taken = end_time - start_time
        total_time[i]=time_taken
        total_rewards[i] = score
        epsilon = max(epsilon - 2/n_games, 0.01)
    
    print("The Agent Reached the Goals: ",reached,' Times')
    frames=[]
    env = gym.make('MountainCar-v0',render_mode='rgb_array')
    for episode in range(50):
        done = False
        observation = env.reset()
        state = get_state(observation)
        flag=True
        steps=1
        while not done:
            frames.append(env.render())
            action = epsilon_greedy_policy(Q, state,epsilon)
            observation, reward, done, _,info = env.step(action)
            state = get_state(observation)
            if done:
                flag=False
            if(steps==max_episode_steps):
                break
            steps+=1
        if flag==False:
            env.close()
            break
    frame_duration = 1000 / max_episode_steps
    imageio.mimsave('Sarsa_Game.gif', frames, duration=frame_duration)

    print("Done Sarsa Algorithm",'\n')
    plt.plot(total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Q-Learning Rewards')
    plt.savefig('Sarsa_mountaincar_Rewards.png')
    plt.close()
    return total_rewards,total_time


##################################################################################################
def monte_carlo_algorithm(env, max_episode_steps=200,n_games=1000, gamma=0.99, epsilon=1.0):
    env._max_episode_steps = max_episode_steps
    pos_space = np.linspace(-1.2, 0.6, 20)
    vel_space = np.linspace(-0.07, 0.07, 20)

    def get_state(observation):
        if isinstance(observation, tuple):
            pos, vel = observation[0]
        else:
            pos, vel = observation
        if isinstance(pos, np.ndarray):
            pos = pos[0]

        pos_bin = int(np.digitize(pos, pos_space))
        vel_bin = int(np.digitize(vel, vel_space))
        return (pos_bin, vel_bin)

    def epsilon_greedy_policy(Q, state, epsilon):
        if np.random.random() < epsilon:
            action = np.random.choice([0, 1, 2])
        else:
            action = np.argmax(Q[state])
        return action

    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos, vel))

    Q = np.zeros((21, 21, 3))
    returns = np.zeros((21, 21, 3))
    N = np.zeros((21, 21, 3))

    score = 0
    reached=0
    total_rewards = np.zeros(n_games)
    total_time=np.zeros(n_games)
    for i in range(n_games):
        
        start_time= time.time()
        done = False
        obs = env.reset()
        state = get_state(obs)
        score = 0
        episode = []
        steps=1
        if i % 10 == 0 and i > 0:
            print('################################# Current Episode ', i,"#################################")
        while True:
            action = epsilon_greedy_policy(Q, state, epsilon)
            obs_, reward, done, _, _ = env.step(action)
            state_ = get_state(obs_)
            score += reward
            episode.append((state, action, reward))
            state = state_
            if (done):
                print("Goal Reached at Episode: ",i,'score ', score,"Observation:",obs_ ,'epsilon %.3f' % epsilon,"steps: ",steps)
                reached+=1
                break
            if(steps==max_episode_steps):
                break
            steps+=1

        total_rewards[i] = score
        epsilon = max(epsilon - 2 / n_games, 0.01)

        G = 0
        #N(St)=N(St)+1
        #V(St)=V(St)+((1/N)*(Gt-V(St))
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state, action) not in episode[:t]:
                N[state][action] += 1
                returns[state][action] += (1 / N[state][action]) * (G - returns[state][action])
                Q[state][action] = returns[state][action]

        end_time = time.time()
        time_taken = end_time - start_time
        total_time[i]=time_taken


    print("The Agent Reached the Goals: ",reached,' Times')
    frames=[]
    env = gym.make('MountainCar-v0',render_mode='rgb_array')
    for episode in range(50):
        done = False
        observation = env.reset()
        state = get_state(observation)
        flag=True
        steps=1

        while not done:
            frames.append(env.render())
            action = epsilon_greedy_policy(Q, state, epsilon)
            observation, reward, done, _,info = env.step(action)
            state = get_state(observation)
            if done:
                flag=False
            if(steps==max_episode_steps):
                break
            steps+=1
        if flag==False:
            env.close()
            break
    frame_duration = 1000 / max_episode_steps
    imageio.mimsave('MonteCarlo_Game.gif', frames, duration=frame_duration)

    print("Done MonteCarlo Algorithm",'\n')
    plt.plot(total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('MonteCarlo Algorithm Rewards')
    plt.savefig('MonteCarlo_mountaincar_Rewards.png')
    plt.close()
    return total_rewards,total_time


###############################################
env = gym.make('MountainCar-v0')
q_learning_start_time = time.time()
print("Running Q-Learning Algorithm")
q_learning_rewards,q_learning_time_episods=q_learning(env,max_episode_steps=4000,n_games=1000,alpha= 0.1,gamma = 0.99)
q_learning_end_time = time.time()
q_learning_time_taken = q_learning_end_time - q_learning_start_time
print("########################################################################################################################")
################################################
env = gym.make('MountainCar-v0')
sarsa_algorithm_start_time = time.time()
print("Running Sarsa Algorithm")
sarsa_algorithm_rewards,sarsa_algorithm_time_episods=sarsa_algorithm(env,max_episode_steps=4000,n_games=1000,alpha= 0.1,gamma = 0.99,epsilon = 1.0)
sarsa_algorithm_end_time = time.time()
sarsa_algorithm_time_taken = sarsa_algorithm_end_time - sarsa_algorithm_start_time
print("########################################################################################################################")

##############################################
env = gym.make('MountainCar-v0')
monte_carlo_start_time = time.time()
print("Running Monte Carlo Algorithm")
monte_carlo_algorithm_rewards,monte_carlo_algorithm_time_episods = monte_carlo_algorithm(env, max_episode_steps=4000,n_games=1000,gamma=0.99, epsilon=1.0)
monte_carlo_end_time = time.time()
monte_carlo_time_taken = monte_carlo_end_time - monte_carlo_start_time
print("########################################################################################################################")

###############################################
#Compare between algorithms 
print('\n')
q_learning_average_mean_reward = np.mean(q_learning_rewards)
print("Q-Learning Average  Reward:", q_learning_average_mean_reward)

sarsa_algorithm_average_mean_reward = np.mean(sarsa_algorithm_rewards)
print("SARSA Algorithm Average  Reward:", sarsa_algorithm_average_mean_reward)

monte_carlo_average_mean_reward = np.mean(monte_carlo_algorithm_rewards)
print("Monte Carlo Average Reward:", monte_carlo_average_mean_reward)
print('\n')

print("Q-Learning Vs SARSA Vs Monte Carlo Time Taken in seconds: ",q_learning_time_taken," Vs ",sarsa_algorithm_time_taken," Vs ",monte_carlo_time_taken)

###############################################

# Plot rewards for both algorithms
plt.plot(q_learning_rewards, label='Q-Learning')
plt.plot(sarsa_algorithm_rewards, label='SARSA')
plt.plot(monte_carlo_algorithm_rewards, label='Monte Carlo')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()
plt.title(' Q-Learning Vs SARSA Vs Monte Carlo Vs - Rewards')
plt.savefig('Rewards__Algorithms_comparison_mountaincar.png')
plt.close()
# plt.show()

# Plot rewards for both algorithms
plt.plot(q_learning_time_episods, label='Q-Learning')
plt.plot(sarsa_algorithm_time_episods, label='SARSA')
plt.plot(monte_carlo_algorithm_time_episods, label='Monte Carlo')
plt.xlabel('Episodes')
plt.ylabel('Tiem')
plt.legend()
plt.title(' Q-Learning Vs SARSA Vs Monte Carlo Vs - Time')
plt.savefig('Time_Algorithms_comparison_mountaincar.png')
plt.close()
# plt.show()
