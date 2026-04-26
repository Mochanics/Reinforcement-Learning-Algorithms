#==============================================================================================================
#Deep Q-Learning code by Mochanics based upon code snippets from the following people:
#Mehdi Shahbazi: https://github.com/MehdiShahbazi/DQN-Frozenlake-Gymnasium (MIT License)
#Johnny Code: https://github.com/johnnycode8/dqn_pytorch (MIT License)
#--------------------------------------------------------------------------------------------------------------
#The flat float tile encoding that makes this PPO perceptive appears to be unique, at least for Gymnasium's "Frozen Lake".
#It was inspired by the grid encoding found in Minigrid (https://minigrid.farama.org/api/wrapper/). It is a training 
# environment made by the Farama Foundation and is based upon OpenAI's gymnasium library. The Minigrid grid encoding
# however has the player position as part of the map grid encoding whereas this implementation has it separated and
# with a separate relative distance to goal added on top [player_position, relative_distance_goal, map_encoding].
#==============================================================================================================

#=================================
#---------Import Libraries--------
#=================================
#Imports for training and evaluating the agent
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import os
import sys
import warnings
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

#Imports for profiling memory allocation and timing the training algorithm
from threading import Event, Thread
from time import sleep
import tracemalloc
import cProfile
import pstats
from time import perf_counter
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import HumanRendering

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
        MAP_SIZE = int(sys.argv[2]) #Map size for frozen lake. Default static maps are 4x4 and 8x8. Any bigger map will be randomly generated using a seed (so it's also kept constant).
    else:
        print("Invalid Parameter")
        quit()
else:
    print("Invalid Parameter")
    quit()

#Defining all hyperparameters
CLIP_GRAD_NORM = 3 #How much to adjust the weight/biases of the policy network to match the target policy network during training. A too big value might lead to overshoots while a too small of a value will increase training time
LEARNING_RATE = 6e-4 #How much to adjust the weight/biases of the policy network to match the target policy network during training. A too big value might lead to overshoots while a too small of a value will increase training time. A too small of a value however can cause the training to be stuck in a local suboptimal minimum.
DISCOUNT_FACTOR = 0.93 #Tells how much to take into account future rewards instead of immediate rewards
BATCH_SIZE = 32 #How big the batch batch of state transitions taken from the replay memory to train the policy network is
UPDATE_FREQUENCY = 10 #How often to update the target network

EPSILON_MAX = 0.999 if TRAINING_MODE else -1 #Maximum epsilon value. If in training mode, the -1 causes the "select_action" function to use a greedy policy only
EPSILON_MIN = 0.01 #Minimum epsilon value
EPSILON_DECAY = 0.999 #The decay rate for the epsilon value

MAX_EPISODES = 3000 #Maximum number of training episode
MAX_STEPS = 200 #The maximum steps allowed per training episode
MEMORY_CAPACITY = 4000 if TRAINING_MODE else 0 #How large the memory replay buffer is. It should be large enough to retain enough state transitions to train the policy network

#Environmental constants
GAME = "frozen_lake" #Alternatively "cliff_walking"
REWARD_SCHEDULE = (1, -1, -0.05) #When to give a reward to the agent and how high that reward is. The values are: Reach Goal, Reach Hole, Reach Frozen (includes Start), respectively
NUM_STATES = MAP_SIZE ** 2 if GAME == "frozen_lake" else 48 #The total number of states the agent could potentially have for the given environment

#Additional non-DQL related constants
MAX_EVALUATING_EPISODES = 100 #Maximum number of testing/evaluating episodes
RENDER = not TRAINING_MODE #What mode to render to environment in. Either None for no rendering at all or "human" for visible rendering.
RECORDING_MODE = not TRAINING_MODE #Records video of the agent in evaluation mode
LOAD_PATH = "./deep-q-learning/models/final_weights_" + str(MAX_EPISODES) + ".pth" #Loading path for the weights and biases used by the policy network in the testing/evaluating mode
SAVE_PATH = "./deep-q-learning/models/final_weights" #Save path for the weights and biases of the policy network after training
SAVE_INTERVAL = 100 #After how many training episodes should the weights and biases of the policy network be saved
DEVICE = torch.device( #Checking which devices are available and selecting it based on that
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

#Evaluation Parameters
PLOTTING = False #Enable plotting of graphs
MEMORY_PROFILING = False #Enable the profiling of memory. FOR ACCURATE RESULTS, DO NOT USE AT THE SAME TIME AS SPEED_PROFILING
SPEED_PROFILING = False #Enable training time measurement. FOR ACCURATE RESULTS, DO NOT USE AT THE SAME TIME AS MEMORY_PROFILING
REPRODUCIBLE = False #Makes the training and evaluating repeatable

#Seed everything for reproducible results
if REPRODUCIBLE:
    SEED = 2024
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
#------Replay Memory Buffer-------
#=================================
class ReplayMemory: #Storing all state transitions from specific actions and their associated rewards and and allows for random sampling of a batch of them for training the neural network.
    #Initiates the replay memory for a given capacity
    def __init__(self, capacity):
        #Data is stored in a few double-ended queues. This is a datatype that allows appending and popping elements from both ends of it.
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.terminateds = deque(maxlen=capacity)
    
    #Stores a specific transition
    def store(self, state, action, next_state, reward, terminated):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.terminateds.append(terminated)
    
    #Takes a random batch of transition samples and returns them as tensors
    def sample(self, batch_size):
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=DEVICE) for i in indices]).to(DEVICE)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=DEVICE)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=DEVICE) for i in indices]).to(DEVICE)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=DEVICE)
        terminateds = torch.as_tensor([self.terminateds[i] for i in indices], dtype=torch.bool, device=DEVICE)

        return states, actions, next_states, rewards, terminateds

    #Returns the current used length of the replay memory
    def __len__(self):
        return len(self.terminateds)

#=================================
#----DQN Neural Network Class-----
#=================================
class DQN_Network(nn.Module): #The architecture of the policy neural network used for DQN
    def __init__(self, num_actions, num_observations):
        super(DQN_Network, self).__init__() #Calling the __init__ method from the parent class this class is derived from
                                                          
        self.FC = nn.Sequential( #Sequential tells PyTorch that the neural network is to be build in the order that its given in the code/constructor.
            nn.Linear(num_observations, 12), #Input layer to 1st hidden layer (observation space)
            nn.ReLU(inplace=True), #ReLU (Rectified Linear Unit) activation function between input and 1st layers. This function only lets a neuron pass a value forward if its positive.
            nn.Linear(12, 8), #1st hidden layer to 2nd hidden layer
            nn.ReLU(inplace=True), #Same ReLU activation function between 1st and 2nd layers
            nn.Linear(8, num_actions), #2nd hidden layer to output layer (action space)
            )
        
        #Initializing the FC layer weights using He initialization
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
    #Take a state and pass it "forward" through the neural network defined in "init" to get a q-value as ouput
    def forward(self, x):
        Q = self.FC(x)
        return Q

#=================================
#-----------Agent Class-----------
#=================================        
class Agent:
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay, clip_grad_norm, learning_rate, discount, memory_capacity):
        #Reinforcement learning hyperparameters (epsilon and discount factor)
        self.epsilon_max   = epsilon_max
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount      = discount

        self.action_space  = env.action_space #Getting the environment action space
        self.observation_space = env.observation_space #Getting the environment observation space
        self.replay_memory = ReplayMemory(memory_capacity) #Creating a replay memory for training
        
        # Initiate the network models
        self.main_network = DQN_Network(num_actions=self.action_space.n, num_observations=self.observation_space.n).to(DEVICE) #Neural network for the policy
        self.target_network = DQN_Network(num_actions=self.action_space.n, num_observations=self.observation_space.n).to(DEVICE).eval() #Target neural network used to train the network policy
        self.target_network.load_state_dict(self.main_network.state_dict()) #Copying the weights and biases from the policy network to the target policy network

        self.clip_grad_norm = clip_grad_norm #For clipping exploding gradients caused by high reward value
        self.criterion = nn.MSELoss() #Mean square error is used to find the loss
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate) #Adam optimization algorithm used. It's the standard algorithm used in reinforcement learning by PyTorch.

    #Selects an action to take based on the current state by either feeding it to the policy neural network or taking it a random (depends on whether the policy is epsilon greedy or just greedy)
    def select_action(self, state):
        #Exploration: epsilon-greedy
        if np.random.random() < self.epsilon_max: #Will also fail if epsilon is manually set to -1 (this occurs when evaluating and not training)
            return self.action_space.sample()
        
        #Exploitation: greedy (the action is selected based on the Q-values, the maximum value is taken)
        with torch.no_grad(): #Turns off gradient evaluation for the following lines to save on processing (PyTorch automatically does otherwise)
            Q_values = self.main_network(state)
            action = torch.argmax(Q_values).item() #item() converts the data back to a standard python value (non-tensor) depending on what the tensor type was (int, float etc.)
            return action

    #Function for optimizing the policy network using the target policy network
    def learn(self, batch_size, terminated):  
        #Sample a batch of experiences from the replay memory
        states, actions, next_states, rewards, terminateds = self.replay_memory.sample(batch_size)
         
        #Preprocessing the data for training (unsqueeze adds a dimension in the direction defined by the argument (1 in this instance) 
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        terminateds = terminateds.unsqueeze(1)

        predicted_q = self.main_network(states) #Forward pass through the main network to find the current policy's predicition of the Q-values for the given states
        predicted_q = predicted_q.gather(dim=1, index=actions) #Selecting only the Q-values for the actions that were actually taken

        #Computing the maximum Q-value for each of the next states using the target network
        with torch.no_grad(): #Turns off gradient evaluation for the following lines to save on processing (PyTorch automatically does otherwise)
            target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0] #Argmax not used because we want the maxmimum q-value, not the action that maximize it

        target_q_value[terminateds] = 0 #Sets the target Q-value for terminal states to zero. This is done so the new_target_q_value is equal to the rewards (see next line) as definsed in the DQN target policy update equation.
        #The reason the target_q_value is made zero when the agent reaches a terminal state is because there's no more future reward to be taken into account since it is the last state.
        
        new_target_q_value = rewards + (self.discount * target_q_value) #Computes the target Q-values using the DQN target formula. This formula takes into account current rewards as well as future rewards with a discount factor.
        loss = self.criterion(predicted_q, new_target_q_value) #Computes the loss (aka difference between predicted and actual) using the MSE algorithm (see above) for all Q-values
            
        self.optimizer.zero_grad() #Zero the gradients. The gradients are representative of which direction the optimizer should shift the weights and biases to in order to minimize the loss
        loss.backward() #Perform backward pass and update the gradients using the loss calculated above

        #Cliping the gradients to prevent "exploding" gradients (too steep)
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)

        #Update the parameters of the main network using the Adam optimizer (seen above). Optimization tries to find weights and biases in order to minimize the loss
        self.optimizer.step() #One step or a single round of optmimization is done using the back propagated values obtained from the loss.
        
    #Function to update the weights and biases from the policy network to the target policy network
    def hard_update(self):
        self.target_network.load_state_dict(self.main_network.state_dict()) 
    
    def update_epsilon(self):     
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def save(self, path):
        torch.save(self.main_network.state_dict(), path)

#=================================
#-Model Training/Evaluating Class-
#=================================              
class Model_TrainTest: #This class is there to store the hyperparameters and methods required to both train and evaluate the agent.
    def __init__(self):
        #Defining RL Hyperparameters for the class
        self.RL_load_path = LOAD_PATH
        self.save_path = SAVE_PATH
        self.save_interval = SAVE_INTERVAL

        self.clip_grad_norm = CLIP_GRAD_NORM
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.batch_size = BATCH_SIZE
        self.update_frequency = UPDATE_FREQUENCY

        self.max_episodes = MAX_EPISODES
        self.max_evaluating_episodes = MAX_EVALUATING_EPISODES
        self.max_steps = MAX_STEPS
        self.render = RENDER
        self.recording_mode = RECORDING_MODE

        self.epsilon_max = EPSILON_MAX
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.memory_capacity = MEMORY_CAPACITY

        self.game = GAME
        self.map_name = MAP_NAME

        self.map_size = MAP_SIZE
        self.num_states = NUM_STATES
                                
        #Defining the environment and its parameters
        if (self.game == "frozen_lake"):
                if self.map_size > 8:
                    self.env = gym.make('FrozenLake-v1', desc=self.map_name, is_slippery=False, render_mode="rgb_array" if self.render else None, reward_schedule=REWARD_SCHEDULE) # max_episode_steps=self.max_steps
                else:
                    self.env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=False, render_mode="rgb_array" if self.render else None, reward_schedule=REWARD_SCHEDULE) # max_episode_steps=self.max_steps    
        elif (self.game == "cliff_walking"):
                self.env = gym.make('CliffWalking-v1', render_mode="rgb_array" if self.render else None)
        else:
                print("ERROR: Not a valid game")
                quit()
                
        if self.recording_mode == True:
            with warnings.catch_warnings(action="ignore"):
                self.env = RecordVideo(
                    self.env,
                    video_folder="./deep-q-learning/recordings", # Folder to save videos
                    name_prefix="vid",               # Prefix for video filenames
                    episode_trigger=lambda x: True    # Record every episode
                )

            self.env = HumanRendering(self.env)
        
        #Defining the agent (instance the Agent class) with its associated hyperparameters
        self.agent = Agent(env                      = self.env, 
                                epsilon_max         = self.epsilon_max, 
                                epsilon_min         = self.epsilon_min, 
                                epsilon_decay       = self.epsilon_decay,
                                clip_grad_norm      = self.clip_grad_norm,
                                learning_rate       = self.learning_rate,
                                discount            = self.discount_factor,
                                memory_capacity     = self.memory_capacity)          
    
    #=================================
    #State Tensor Preprocessing Method
    #=================================
    def state_preprocess(self, state:int, num_states:int): #Creating a zero tensor and putting a 1 at the location in the tensor representing the current state. This tensor describes what the current state of the system is
        onehot_vector = torch.zeros(num_states, dtype=torch.float32, device=DEVICE) #Creating a tensor full of zeroes whose size is equal to the number of possible states the agent can have.
        onehot_vector[state] = 1 #Set the tensor equal to one at the current state.
        return onehot_vector
    
    #=================================
    #---------Training Method---------
    #=================================
    def train(self):
        total_steps = 0
        
        if PLOTTING == True:
            episodes_num = []
            rewards = []
        
        start_time = perf_counter()
        
        #Training loop running for n number of episodes        
        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            state = self.state_preprocess(state, num_states=self.num_states)
            terminated = False
            truncated = False
            steps_taken = 0
            episode_rewards = 0
                                                
            while not terminated and not truncated:
                #Select an action from the current state (since its training, that action will be chosen using epsilon-greedy)
                action = self.agent.select_action(state)
                
                #Do the action and get the new state obtained from taking it
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                #Creating a state tensor for the next_state (a tensor the size of all possible states the agent is in that is filled with zeroes except for the current state being 1)
                next_state = self.state_preprocess(next_state, num_states=self.num_states)
                
                #Store the new state obtained from the previous state and action as well as the obtained reward
                self.agent.replay_memory.store(state, action, next_state, reward, terminated)
        
                #Train the agent by calling the learn function once enough transitions have been stored in the replay memory (when the replay memory is larget than the batch size) and there is more than 0 total rewards
                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (terminated or truncated))

                if reward == -100 and self.game == "cliff_walking":
                    truncated = True
                
                state = next_state
                episode_rewards += reward
                steps_taken += 1
                                                                       
            #Updating (decaying) epsilon at the end of each episode
            self.agent.update_epsilon()
            
            #Update target policy network weights and biases every n number of total steps (as defined i the update frequency hyperparameter)
            if total_steps % self.update_frequency == 0:
                self.agent.hard_update() #Calling the target update function
            
            #Printing episode stats 
            print(f"Episode: {episode + 1}, " f"Episode Steps: {steps_taken}, " f"Episode Rewards: {episode_rewards:.2f}, " f"Epsilon: {self.agent.epsilon_max:.4f}")  
            
            #Saving the trained model every "save interval" (number of episodes)
            if (episode + 1) % self.save_interval == 0:
                print("\n==== Saving model parameters ====\n")
                self.agent.save(self.save_path + '_' + f'{episode + 1}' + '.pth')
                
            if PLOTTING == True:
                episodes_num.append(episode)
                rewards.append(episode_rewards)
         
        end_time = perf_counter() 
        time = end_time - start_time
        print("Time: ", time)
        with open('./deep-q-learning/timeProfiling.log', 'w') as f:
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
            np.savetxt('./deep-q-learning/training_graph_data.csv', np.column_stack([episodes_num, rewards]), delimiter=",", header="Episodes, Rewards", fmt='%s')                                           
    
    #=================================
    #--------Evaluating Method--------
    #=================================
    def test(self):      
        #Loading the weights of the trained policy network being tested
        print(self.RL_load_path)
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path)) #Copy the weights and biases from a save file to the policy network 
        self.agent.main_network.eval() #Sets the policy network to evaluation mode.
        
        episodes_num = []
        rewards = []
        steps = []
        success = 0
        
        #Testing loop over episodes
        for episode in range(self.max_evaluating_episodes):         
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            episode_steps = 0
            episode_rewards = 0
                                                           
            while not terminated and not truncated:
                state = self.state_preprocess(state, num_states=self.num_states)
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                                
                state = next_state
                episode_rewards += reward
                episode_steps += 1
                        
                if terminated and ((GAME == "frozen_lake" and reward == REWARD_SCHEDULE[0]) or (GAME == "cliff_walking" and reward != -100)):
                    success += 1
                                                                                                                       
            #Printing episode stats            
            print(f"Episode: {episode + 1}, " f"Steps: {episode_steps:}, " f"Rewards: {episode_rewards:.2f}")
            
            episodes_num.append(episode)
            rewards.append(episode_rewards)
            steps.append(episode_steps)
          
        print("Percentage success rate: ", (success/self.max_evaluating_episodes)*100, "%")
        
        self.env.close() #Closing the rendering window after the episode is over
        
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
        np.savetxt('./deep-q-learning/evaluating_graph_data.csv', np.column_stack([episodes_num, rewards, steps]), delimiter=",", header="Episodes, Rewards, Steps", fmt='%s')  

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
        
        with open('./deep-q-learning/memoryProfiling.log', 'w') as f:
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
    
    #Creating a profile to time how long the train_agent function takes to run (how long it takes to train the agent)
    if SPEED_PROFILING == True:
        pr = cProfile.Profile()
        pr.enable()
    
    DRL = Model_TrainTest() #Defining the instance

    if TRAINING_MODE:
        # Training the policy network
        DRL.train()
    else:
        #Testing and evaluating the policy network
        DRL.test()
    
    if SPEED_PROFILING == True:
        pr.disable()
        with open('./deep-q-learning/timeDetailedProfiling.log', 'w') as f:
            pstats.Stats( pr, stream=f ).strip_dirs().sort_stats("cumulative").print_stats()
        
    if MEMORY_PROFILING == True:
        end_profiler.set()
        thread.join()
