#==================================================================================================
#SARSA Reinforcement Learning Code by Mochanics based upon code snippets from the following people:
#Inspired by https://www.datacamp.com/tutorial/sarsa-reinforcement-learning-algorithm-in-python
#==================================================================================================

#Imports for training and evaluating the agent
import numpy as np
import random
import gymnasium as gym
import os
import sys
import warnings

#Imports for profiling memory allocation and timing the training algorithm
from threading import Event, Thread
from time import sleep
import tracemalloc
import cProfile
import pstats
from time import perf_counter
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import HumanRendering
from gymnasium.envs.toy_text.frozen_lake import generate_random_map #Only used when needing to generate a random map (MAP_NAME) for "FrozenLake-v1"

#Imports for graphing performance
import matplotlib.pyplot as plt

#=================================
#------------Constants------------
#=================================
#Checking whether in training or testing mode
if len(sys.argv) > 1:
    if sys.argv[1].lower() == "train": 
        TRAINING_MODE = True
    elif sys.argv[1].lower() == "test":
        TRAINING_MODE = False
    else:
        print("Invalid Parameter")
        quit()
        
    #Checking if map size data is passed and if so, what size to use.
    if len(sys.argv) == 3 and sys.argv[2].isdigit():
        MAP_SIZE = int(sys.argv[2])
    else:
        print("Invalid Parameter")
        quit()
else:
    print("Invalid Parameter")
    quit()

#Environmental constants
GAME = "FrozenLake-v1"
REWARD_SCHEDULE = (1, -5, -0.1)

EPISODES = 5000 #Total number of learning episodes
TEST_EPISODES = 100 #Total number of test episodes

#Defining all hyperparameters
ALPHA = 0.1
GAMMA = 0.99

MAX_EPSILON = 1.0
MIN_EPSILON = 0.05
DECAY_RATE = 0.0005

#Debug Constants
PLOTTING = True #Enable plotting of graphs
MEMORY_PROFILING = False #Enable the profiling of memory. FOR ACCURATE RESULTS, DO NOT USE AT THE SAME TIME AS SPEED_PROFILING
SPEED_PROFILING = False #Enable training time measurement. FOR ACCURATE RESULTS, DO NOT USE AT THE SAME TIME AS MEMORY_PROFILING
REPRODUCIBLE = False #Makes the training and evaluating repeatable

#Seed everything for reproducible results
if REPRODUCIBLE:
    SEED = 2024
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

#For "FrozenLake-v1" only: Use predefined map if size 4 or 8 or generate one if above 8.
if MAP_SIZE > 8:
    MAP_NAME = generate_random_map(size=MAP_SIZE, p=0.9, seed=2000) #The seed is to make sure that the generated map is constant.
else:
    MAP_NAME = f"{MAP_SIZE}x{MAP_SIZE}"

#=================================
#-----Rolling Average Function----
#=================================
def rolling_average(data, window): #The purpose is to "smooth out" the a very noisy dataset such as the rewards per episode.
    #"window" is how many episodes to average over.
    kernel = np.ones(window) / window
    
    #Slides the kernel across the data to compute the rolling average. mode='same' ensures the output is the same length as the input.
    averaged = np.convolve(data, kernel, mode='same')
    
    #Mask the edges where padding distorts the result
    half = window // 2 #Finds the middle of the data quickly find the edges and remove them
    averaged[:half] = np.nan
    averaged[-half:] = np.nan
    return averaged

#=================================
#--------Environment Class--------
#=================================
class Environment:
    def __init__(self, environment, mode, name, recording_mode):
        #Create training environment
        if environment == "FrozenLake-v1": 
            if MAP_SIZE > 8:
                self.env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=mode, desc=MAP_NAME, reward_schedule=REWARD_SCHEDULE)
            else:
                self.env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=mode, map_name=MAP_NAME, reward_schedule=REWARD_SCHEDULE)
        elif environment == "CliffWalking-v1":
            self.env = gym.make("CliffWalking-v1", is_slippery=False, render_mode=mode)
            
        if recording_mode == True:
            with warnings.catch_warnings(action="ignore"):
                self.env = RecordVideo(
                    self.env,
                    video_folder="./sarsa/recordings", # Folder to save videos
                    name_prefix="vid",               # Prefix for video filenames
                    episode_trigger=lambda x: True    # Record every episode
                )

            self.env = HumanRendering(self.env)

    def info(self):
        return self.env
        
    def sample(self):
        return self.env.action_space.sample() 
    
    def reset(self):
        # Reset environment to start a new run/episode
        observation, info = self.env.reset()
        return observation

    def step(self, action):
        # Do the action and get results
        observation, reward, terminated, truncated, info = self.env.step(action)

        return (observation, reward, terminated, truncated, info)

    def end(self):
        #Close training envrionment
        self.env.close()

def create_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space)),
    return Qtable[0]

#=================================
#-----Random Policy Function------
#=================================
def random_policy(env):
    action = env.sample()
    return action

#=================================
#-Epsilon Greedy Policy Function--
#=================================
def epsilon_greedy_policy(env, Qtable, state, epsilon):
    random_val = random.uniform(0,1)
    if random_val > epsilon:
        action = np.argmax(Qtable[state])
    else:
        action = env.sample()
    return action

#=================================
#-----Greedy Policy Function------
#=================================
def greedy_policy(Qtable, state):
    return np.argmax(Qtable[state])

#=================================
#--------Training Function--------
#=================================
def train_agent():  
    env = Environment(GAME, None, None, False)
    
    env_info = env.info()
    Qtable = create_q_table(env_info.observation_space.n, env_info.action_space.n)
    
    if PLOTTING == True:
        episodes_num = []
        rewards = []
        steps = []
    
    start_time = perf_counter()  
    
    for episode in range(EPISODES):
        #Setup Episode
        terminated = False
        truncated = False
        episode_rewards = 0
        episode_steps = 0
        is_only_greedy = False
        
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON)*np.exp(-DECAY_RATE*episode) #Decaying epsilon
        
        state = env.reset()
        action = greedy_policy(Qtable, state)
        
        while (not terminated and not truncated):
            #Get the new state, reward and wether the episode has terminated or has been truncated 
            next_state, reward, terminated, truncated, info = env.step(action)
            
            if (is_only_greedy is True):
                next_action = greedy_policy(Qtable, next_state)
            else:
                next_action = epsilon_greedy_policy(env, Qtable, next_state, epsilon)
            
            #Perform SARSA update on Q-table
            Qtable[state, action] += ALPHA * (reward + GAMMA * Qtable[next_state, next_action] - Qtable[state, action])
            state = next_state
            action = next_action
            
            episode_rewards += reward
            episode_steps += 1
        
        if PLOTTING == True:
            episodes_num.append(episode)
            rewards.append(episode_rewards)
            steps.append(episode_steps)

        print(f"Episode: {episode + 1}, " f"Episode Steps: {episode_steps}, " f"Episode Rewards: {episode_rewards:.2f}, " f"Epsilon: {epsilon:.4f}")
    
    end_time = perf_counter() 
    time = end_time - start_time
    print("Time: ", time)
    with open( './sarsa/timeProfiling.log', 'w' ) as f:
        f.write(str(time))
    
    if PLOTTING == True:
        #Plotting rewards against episodes on a graph.
        window = 100 #Rolling average window.
        plt.plot(episodes_num, rewards, color="steelblue", label="Rewards per Episode")
        plt.plot(episodes_num, rolling_average(rewards, window), color="red", label=f"Rolling Average (window={window})")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.grid(linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

        #Saving plot data to a csv file
        np.savetxt('./sarsa/training_graph_data.csv', np.column_stack([episodes_num, rewards]), delimiter=",", header="Episodes, Rewards", fmt='%s')
    
    env.end()
    return Qtable

#=================================
#-------Evaluating Function-------
#=================================
def evaluate_agent(Qtable):
    env = Environment(GAME, "rgb_array", None, True)
    
    episodes_num = []
    rewards = []
    steps = []
    success = 0
    
    for episode in range(TEST_EPISODES):
        terminated = False
        truncated = False
        episode_rewards = 0
        episode_steps = 0
        state = env.reset()
    
        while (not terminated and not truncated):
            #Take the action (index) that have the maximum reward
            action = greedy_policy(Qtable, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            state = new_state
            episode_rewards += reward
            episode_steps += 1
            
            if terminated and ((GAME == "FrozenLake-v1" and reward == REWARD_SCHEDULE[0]) or (GAME == "CliffWalking-v1" and reward != -100)):
                success += 1
        
        print(f"Episode: {episode + 1}, " f"Steps: {episode_steps:}, " f"Reward: {episode_rewards:.2f}")
        
        episodes_num.append(episode)
        rewards.append(episode_rewards)
        steps.append(episode_steps)
    
    print("Percentage success rate: ", (success/TEST_EPISODES)*100, "%")
    
    env.end()
    
    #Plotting the results on two graphs.
    window = 100 #Rolling average window.
    plt.subplot(1, 2, 1) #First graph is rewards per episode.
    plt.plot(episodes_num, rewards, color="steelblue", label="Rewards per Episode")
    plt.plot(episodes_num, rolling_average(rewards, window), color="red", label=f"Rolling Average (window={window})")
    plt.title("Rewards per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend()
    plt.subplot(1, 2, 2) #Second graph is steps per episode.
    plt.plot(episodes_num, steps, color="steelblue", label="Steps per Episode")
    plt.plot(episodes_num, rolling_average(steps, window), color="red", label=f"Rolling Average (window={window})")
    plt.title("Steps per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

    #Saving plot data to a csv file
    np.savetxt('./sarsa/evaluating_graph_data.csv', np.column_stack([episodes_num, rewards, steps]), delimiter=",", header="Episodes, Rewards, Steps", fmt='%s')
    
#=================================
#------Memory Profiler Class------
#=================================
class Profiler(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = None
    
    def run(self):
        values = []
        tracemalloc.start()
        while not end_profiler.is_set():
            value, peak = tracemalloc.get_traced_memory()
            values.append(value)
            #print(f"Current RAM Usage: {value} | Peak Overall RAM Usage: {peak}")
            sleep(0.01)
        tracemalloc.stop()
        average = sum(values)/len(values)
        
        with open('./sarsa/memoryProfiling.log', 'w') as f:
            for value in values:
                f.write(str(value) + "\n")
            f.write("Peak: " + str(peak) + "\n")
            f.write("Average: " + str(average) + "\n")
        
        self.results = (average, peak)

#=================================
#----------Main Function----------
#=================================
if __name__ == "__main__":
    #Creating and starting a thread to profile the memory usage
    if MEMORY_PROFILING == True:
        end_profiler = Event()
        thread = Profiler()
        thread.start()

    if TRAINING_MODE: #If in training mode        
        #Creating a profile to time how long the train_agent function takes to run (how long it takes to train the agent)
        if SPEED_PROFILING == True:
            pr = cProfile.Profile()
            pr.enable()

        Qtable = train_agent()
        
        if SPEED_PROFILING == True:
            pr.disable()
            with open( './sarsa/profilingResults.log', 'w' ) as f:
                pstats.Stats( pr, stream=f ).strip_dirs().sort_stats("cumulative").print_stats()

        np.savetxt("./sarsa/q-table.txt", Qtable)
        print(Qtable)
    
    else: #If not in training mode
        Qtable = np.loadtxt("./sarsa/q-table.txt")
        evaluate_agent(Qtable)
    
    if MEMORY_PROFILING == True:
        end_profiler.set()
        thread.join()
