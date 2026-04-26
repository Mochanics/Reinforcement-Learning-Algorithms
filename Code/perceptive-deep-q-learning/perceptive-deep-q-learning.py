#==============================================================================================================
#Perceptive deep Q-Learning code by Mochanics based upon code snippets from the following people:
#Mehdi Shahbazi: https://github.com/MehdiShahbazi/DQN-Frozenlake-Gymnasium (MIT License)
#Johnny Code: https://github.com/johnnycode8/dqn_pytorch (MIT License)
#--------------------------------------------------------------------------------------------------------------
#The flat float tile encoding that makes this DQL perceptive appears to be unique, at least for Gymnasium's "Frozen Lake".
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
        MAP_SIZE = int(sys.argv[2]) #Map size for frozen lake. Default static maps are 4x4 and 8x8. Any bigger map will be randomly generated (unless "REPRODUCIBLE = True").
    else:
        print("Invalid Parameter")
        quit()
else:
    print("Invalid Parameter")
    quit()

#Defining all hyperparameters for DQL
CLIP_GRAD_NORM = 3 #How much to adjust the weight/biases of the policy network to match the target policy network during training. A too big value might lead to overshoots while a too small of a value will increase training time.
LEARNING_RATE = 1e-4 #How much to adjust the weight/biases of the policy network to match the target policy network during training. A too big value might lead to overshoots while a too small of a value will increase training time. A too small of a value however can cause the training to be stuck in a local suboptimal minimum.
DISCOUNT_FACTOR = 0.99 #Tells how much to take into account future rewards instead of immediate rewards.
BATCH_SIZE = 128 #How big the batch batch of state transitions taken from the replay memory to train the policy network is.
TARGET_UPDATE_STEPS = 500 #Instead of training every episode, the agent trains every TARGET_UPDATE_STEPS number of steps. Episode-based updates made the training rate unpredictable. For example a run of 1-step episodes would update the target network every steps, destroying stability.
TRAIN_FREQ = 4 #Instead of training every step, the agent trains every TRAIN_FREQ steps. Makes training more stable.
LEARNING_STARTS = 1000 #The number of transitions that need to be stored first before the agent starts learning. Learning from a near-empty buffer produces extremely correlated batches.
N_STEP = 3 #N-step returns. Instead of storing single-step transitions, the agent accumulates n steps and then stores the discounted n-step return. n=3 is a safe choice.

EPSILON_MAX   = 0.999 if TRAINING_MODE else -1 #Maximum epsilon value. If in testing mode, -1 forces the "select_action" function to use a greedy policy only.
EPSILON_MIN   = 0.01 #Minimum epsilon value.
EPSILON_DECAY = 0.999988 #The decay rate for the epsilon value. Now per step instead of per episode.

MAX_EPISODES    = 6000 #10000 #Maximum number of training episode.
MEMORY_CAPACITY = 100000 if TRAINING_MODE else 0 #How large the memory replay buffer is. It should be large enough to retain enough state transitions to train the policy network.

#Environmental constants
ENV_NAME         = "FrozenLake-v1" #Only this environement is used for this version of DQL.
P_FROZEN         = 0.9 #Probability of a tile being frozen. Used by the random map generation function.
REWARD_SCHEDULE  = (1, -1, -0.04) #(1, -5, -0.01) #Reach Goal, Reach Hole, Reach Frozen (includes Start)
TILE_ENCODING    = {b'S': 0.0, b'F': 0.25, b'H': 1.0, b'G': 0.5}#{b'S': 0.0, b'F': 0.00, b'H': -1.0, b'G': 1.0} #Tile encoding for rich state: S (start) = 0.00, F (frozen) = 0.25, H (hole) = 1.00, G (goal) = 0.50. Those values were picked arbitrarily.
NUM_OBSERVATIONS = 4 + MAP_SIZE ** 2 #Rich state observation dimensions. State vector: [norm_row, norm_col] + flattened encoded tile map.
MAX_STEPS_TRAIN = 200 #Max number of training episode steps. training episodes need more steps than the FrozenLake-v1 default of 100 so that random exploration has a realistic chance of stumbling onto the goal.
MAX_STEPS_TEST  = 100 #Max number of testing episode steps.
GENERATE_MAP_PER_EPISODE = True #Parameter specific to perceptive models. Generates a new map every training episode to learn to generalize hole avoiding behaviour.

#Additional non-DQL related constants
MAX_EVALUATING_EPISODES = 100 #Maximum number of testing/evaluating episodes.
RENDER          = not TRAINING_MODE #What mode to render to environment in. Either None for no rendering at all or "human" for visible rendering.
RECORDING_MODE  = not TRAINING_MODE #Records video of the agent in evaluation mode.
LOAD_PATH       = "./perceptive-deep-q-learning/models/final_weights_" + str(MAX_EPISODES) + ".pth" #Loading path for the weights and biases used by the policy network in the testing/evaluating mode.
SAVE_PATH       = "./perceptive-deep-q-learning/models/final_weights" #Save path for the weights and biases of the policy network after training.
SAVE_INTERVAL   = 500 #After how many training episodes should the weights and biases of the policy network be saved.
DEVICE = torch.device( #Checking which devices are available and selecting it based on that.
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

#Evaluation Parameters
PLOTTING = True #Enable plotting of graphs.
MEMORY_PROFILING = False #Enable the profiling of memory. FOR ACCURATE RESULTS, DO NOT USE AT THE SAME TIME AS SPEED_PROFILING.
SPEED_PROFILING = False #Enable training time measurement. FOR ACCURATE RESULTS, DO NOT USE AT THE SAME TIME AS MEMORY_PROFILING.
REPRODUCIBLE = False #Makes the training and evaluating repeatable.

#Seed everything for reproducible results.
if REPRODUCIBLE:
    SEED = 2024
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

#Use predefined map if size 4 or 8 or generate one if above 8.
if MAP_SIZE > 8:
    MAP_NAME = generate_random_map(size=MAP_SIZE, p=P_FROZEN, seed=2000)
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

    #Stores a specific transition.
    def store(self, state, action, next_state, reward, terminated):
        #Data is stored in a few double-ended queues. This is a datatype that allows appending and popping elements from both ends of it.
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.terminateds.append(terminated)

    #Takes a random batch of transition samples and returns them as tensors.
    def sample(self, batch_size):
        indices     = np.random.choice(len(self), size=batch_size, replace=False)
        states      = torch.stack([torch.as_tensor(self.states[i],      dtype=torch.float32, device=DEVICE) for i in indices])
        actions     = torch.as_tensor([self.actions[i]     for i in indices], dtype=torch.long,    device=DEVICE)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=DEVICE) for i in indices])
        rewards     = torch.as_tensor([self.rewards[i]     for i in indices], dtype=torch.float32, device=DEVICE)
        terminateds = torch.as_tensor([self.terminateds[i] for i in indices], dtype=torch.bool,    device=DEVICE)
        return states, actions, next_states, rewards, terminateds

    #Returns the current used length of the replay memory.
    def __len__(self):
        return len(self.terminateds)

#=================================
#----DQN Neural Network Class-----
#=================================
class DQN_Network(nn.Module): #The architecture of the policy neural network used for DQN   
    def __init__(self, num_actions, num_observations): #num_observations now accepts the rich state dimension (2 + MAP_SIZE^2)
        super(DQN_Network, self).__init__() #Calling the __init__ method from the parent class this class is derived from
                                                          
        self.FC = nn.Sequential( #Sequential tells PyTorch that the neural network is to be build in the order that its given in the code/constructor.
            nn.Linear(num_observations, 128), #Input layer to 1st hidden layer — wider to handle the richer input
            nn.LayerNorm(128),
            nn.ReLU(inplace=True), #ReLU (Rectified Linear Unit) activation function between input and 1st layers. This function only lets a neuron pass a value forward if its positive.
            nn.Linear(128, 128), #1st hidden layer to 2nd hidden layer
            nn.LayerNorm(128),
            nn.ReLU(inplace=True), #Same ReLU activation function between 1st and 2nd layers
            nn.Linear(128, num_actions), #2nd hidden layer to output layer (action space)
        )
        
        #Initializing the FC layer weights using He initialization
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    #Take a state and pass it "forward" through the neural network defined in "init" to get a q-value as ouput
    def forward(self, x):
        return self.FC(x)

#=================================
#-----------Agent Class-----------
#=================================
class Agent:
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay, clip_grad_norm, learning_rate, discount, memory_capacity, num_observations):
        #Reinforcement learning hyperparameters (epsilon and discount factor)
        self.epsilon_max   = epsilon_max
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount      = discount

        self.action_space      = env.action_space #Getting the environment action space
        
        if REPRODUCIBLE:
            self.action_space.seed(SEED) #Set the seed to get reproducible results when sampling the action space
        
        self.observation_space = env.observation_space #Getting the environment observation space
        self.replay_memory     = ReplayMemory(memory_capacity) #Creating a replay memory for training

        #num_observations is now passed in explicitly (rich state dim) rather than being inferred from observation_space.n
        self.main_network   = DQN_Network(num_actions=self.action_space.n, num_observations=num_observations).to(DEVICE)
        self.target_network = DQN_Network(num_actions=self.action_space.n, num_observations=num_observations).to(DEVICE).eval()
        self.target_network.load_state_dict(self.main_network.state_dict()) #Copying the weights and biases from the policy network to the target policy network

        self.clip_grad_norm = clip_grad_norm #For clipping exploding gradients caused by high reward value
        self.criterion = nn.SmoothL1Loss() #SmoothL1Loss algorithm used in this version instead of the mean square error (nn.MSELoss()) used in other versions of DQL. It is used to find the loss
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate) #Adam optimization algorithm used. It's the standard algorithm used in reinforcement learning by PyTorch.

    #Selects an action to take based on the current state by either feeding it to the policy neural network or taking it a random (depends on whether the policy is epsilon greedy or just greedy)
    def select_action(self, state):
        #Exploration: epsilon-greedy
        if np.random.random() < self.epsilon_max: #Will also fail if epsilon is manually set to -1 (this occurs when evaluating and not training)
            return self.action_space.sample()
        
        #Exploitation: greedy (the action is selected based on the Q-values, the maximum value is taken)
        with torch.no_grad(): #Turns off gradient evaluation for the following lines to save on processing (PyTorch automatically does otherwise)
            return torch.argmax(self.main_network(state)).item() #item() converts the data back to a standard python value (non-tensor) depending on what the tensor type was (int, float etc.)

    #Method for optimizing the policy network using the target policy network.
    def learn(self, batch_size):
        #Sample a batch of experiences from the replay memory
        states, actions, next_states, rewards, terminated = self.replay_memory.sample(batch_size)
         
        #Preprocessing the data for training (unsqueeze adds a dimension in the direction defined by the argument (1 in this instance) 
        actions    = actions.unsqueeze(1)
        rewards    = rewards.unsqueeze(1)
        terminated = terminated.unsqueeze(1)

        #Forward pass through the main network to find the current policy's predicition of the Q-values for the given states and then selecting only the Q-values for the actions that were actually taken.
        predicted_q = self.main_network(states).gather(dim=1, index=actions)  

        #Computing the maximum Q-value for each of the next states using the target network.
        with torch.no_grad():
            best_actions   = self.main_network(next_states).argmax(dim=1, keepdim=True) #Adding a double DQN target
            target_q_value = self.target_network(next_states).gather(1, best_actions)
            #Old single DQN would systematically overestimates Q-values because the max operator selects the noisiest-high estimate so was replaced with double DQN.
            #The main network picks the best action, the target network evaluates it. The two networks are unlikely to overestimate the same action simultaneously, so the bias cancels.

        target_q_value[terminated] = 0 #Sets the target Q-value for terminal states to zero. This is done so the new_target_q_value is equal to the rewards (see next line) as definsed in the DQN target policy update equation.
        #The reason the target_q_value is made zero when the agent reaches a terminal state is because there's no more future reward to be taken into account since it is the last state.

        new_target_q_value = rewards + (self.discount ** N_STEP) * target_q_value #Computes the target Q-values using a modified DQN target formula which has n-step discount. Transitions arriving here carry an n-step return

        loss = self.criterion(predicted_q, new_target_q_value) #Computes the loss (aka difference between predicted and actual) using the MSE algorithm (see above) for all Q-values

        self.optimizer.zero_grad() #Zero the gradients. The gradients are representative of which direction the optimizer should shift the weights and biases to in order to minimize the loss
        loss.backward() #Perform backward pass and update the gradients using the loss calculated above

        #Cliping the gradients to prevent "exploding" gradients (too steep)
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)

        #Update the parameters of the main network using the Adma optimizer (seen above). Optimization tries to find weights and biases in order to minimize the loss
        self.optimizer.step() #One step or a single round of optmimization is done using the back propagated values obtained from the loss.

    #Function to update the weights and biases from the policy network to the target policy network. 
    def hard_update(self):
        self.target_network.load_state_dict(self.main_network.state_dict()) 

    #Function to update the epsilon value by decaying it.
    def update_epsilon(self):     
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    #Function to save the Deep Q Network weight and biases to a file.
    def save(self, path):
        torch.save(self.main_network.state_dict(), path)

#=================================
#---------N-Step Buffer-----------
#=================================
#Code taken from:
#Tim Miller: https://gibberblot.github.io/rl-notes/single-agent/n-step.html
#Jinwoo Park: https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/07_n_step_learning.py
#Alex: https://www.kaggle.com/code/auxeno/n-step-dqn-on-acrobot-rl#N-Step-DQN
#Original Paper: http://www.incompleteideas.net/papers/fernando-sutton-2019.pdf
class NStepBuffer: #Sliding window that accumulates N transitions and computes an n-step return.
    #An n-step return compresses n consecutive rewards into one target: step_reward_{t} + gamma*step_reward_{t+1} + gamma^2*step_reward_{t+2} + ... + gamma^{n-1}*r_{t+n-1}
    #The it bootstraps from s_{t+n} using the target network. This means the goal reward propagates back n steps in a single update.
    def __init__(self, n_steps, discount_factor):
        self.n_steps         = n_steps          #How many steps to accumulate before storing a transition.
        self.discount_factor = discount_factor  #Also known as gamma. It is used to weight rewards further in the future less heavily.
        self.window          = deque(maxlen=n_steps) #Circular buffer holding the last n_steps transitions.

    def add(self, state, action, step_reward, next_state, is_terminated, is_truncated):
        self.window.append((state, action, step_reward, next_state, is_terminated, is_truncated)) #Pushing one raw transition into the sliding window.

    def is_ready(self):
        return len(self.window) == self.n_steps #Returns "True" once the window holds exactly n_steps transitions. A full n-step return can then be computed using get().

    def get(self): #Computing the n-step discounted return for the oldest transition in the window.
        #Iterates forward through the window, accumulating: n_step_return = step_reward_{t} + gamma*step_reward_{t+1} + gamma^2*step_reward_{t+2} + ...
        #Stops early if a true terminal state is encountered inside the window, because no future rewards exist beyond it as the episode is over.
        n_step_return    = 0.0
        window_terminated = False

        for step_offset, (_, _, step_reward, _, is_terminated, _) in enumerate(self.window):
            #Add the reward at this offset, discounted by "gamma^step_offset"
            n_step_return += (self.discount_factor ** step_offset) * step_reward

            if is_terminated:
                #Occurs when true terminal (hole or goal) was reached inside the window.
                #No future rewards exist beyond this point, so stop accumulating.
                window_terminated = True
                break

        #The state and action we are computing the return for are always the oldest ones in the window.
        oldest_state,  oldest_action, _, _, _, _ = self.window[0]
        
        #The bootstrap next state is always the newest state in the window (s_{t+n})
        _, _, _, final_next_state, _, _          = self.window[-1]

        return oldest_state, oldest_action, n_step_return, final_next_state, window_terminated
        #oldest_state: the state the agent was in at the start of the window
        #oldest_action: the action taken from oldest_state
        #n_step_return: the accumulated discounted return R_n
        #final_next_state: the next state reached after the full n steps (used for bootstrapping)
        #window_terminated: True if any step inside the window was a true terminal

    def flush_to(self, replay_memory): #Deleting any remaining partial windows at episode end.
        #During a full training episode, get() is called every time the window is full (n_steps transitions). This means the last (n_steps - 1) transitions of an episode never triggered. Hence a full get() call and would be silently discarded.
        #flush_to() therefore iterates over those remaining transitions and stores each one with its own shorter-horizon n-step return (using however many steps are left in the window rather than the full n_steps).
        buffer_snapshot   = list(self.window)
        
        #If the buffer is exactly full (common mid-episode), the last full window was already stored in get(); start from position 1 to avoid duplication.
        flush_start_index = 1 if len(buffer_snapshot) == self.n_steps else 0

        for window_start in range(flush_start_index, len(buffer_snapshot)):
            n_step_return    = 0.0
            window_terminated = False

            #Accumulate the return from this position to the end of the snapshot
            for step_offset, (_, _, step_reward, _, is_terminated, _) in enumerate(buffer_snapshot[window_start:]):
                #Add the reward at this offset, discounted by "gamma^step_offset"
                n_step_return += (self.discount_factor ** step_offset) * step_reward
                
                if is_terminated:
                    #Occurs when true terminal (hole or goal) was reached inside the window.
                    #No future rewards exist beyond this point, so stop accumulating.
                    window_terminated = True
                    break

            #Oldest state/action in this sub-window
            oldest_state, oldest_action, _, _, _, _ = buffer_snapshot[window_start]
            
            #Bootstrap from the last state in the full snapshot
            _, _, _, final_next_state, _, _          = buffer_snapshot[-1]

            #Storing the remaining 1-(n-1) transitions that didn't form a full window during the episode.
            replay_memory.store(oldest_state, oldest_action, final_next_state, n_step_return, window_terminated)

        #Reset the window so the next episode starts with an empty buffer
        self.window.clear()

#=================================
#-Model Training/Evaluating Class-
#=================================
class Model_TrainTest: #This class is there to store the hyperparameters and methods required to both train and evaluate the agent.
    def __init__(self):
        #Defining RL Hyperparameters for the class
        self.rl_load_path    = LOAD_PATH
        self.save_path       = SAVE_PATH
        self.save_interval   = SAVE_INTERVAL

        self.clip_grad_norm  = CLIP_GRAD_NORM
        self.learning_rate   = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.batch_size      = BATCH_SIZE

        self.max_episodes            = MAX_EPISODES
        self.max_evaluating_episodes = MAX_EVALUATING_EPISODES
        self.render                  = RENDER
        self.recording_mode          = RECORDING_MODE

        self.epsilon_max   = EPSILON_MAX
        self.epsilon_min   = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.memory_capacity  = MEMORY_CAPACITY
        self.n_step_buffer = NStepBuffer(n_steps=N_STEP, discount_factor=DISCOUNT_FACTOR) #N-step buffer shared across all training episodes.
        
        self.map_size         = MAP_SIZE
        self.num_observations = NUM_OBSERVATIONS #Rich state dimension. Different from the non-perceptive deep Q learning.
        self.env = self.generate_new_map("rgb_array" if self.render else None, GENERATE_MAP_PER_EPISODE, MAX_STEPS_TRAIN)

        if self.recording_mode: #If enabled, record video (only used in testing)
            with warnings.catch_warnings(action="ignore"):
                self.env = RecordVideo(
                    self.env,
                    video_folder="./perceptive-deep-q-learning/recordings", #Folder to save videos to.
                    name_prefix="vid",               #Prefix for video filenames.
                    episode_trigger=lambda x: True    #Record every episode.
                )
            
            self.env = HumanRendering(self.env)

        #Defining the agent (instance the Agent class) with its associated hyperparameters.
        self.agent = Agent(
            env              = self.env,
            epsilon_max      = self.epsilon_max,
            epsilon_min      = self.epsilon_min,
            epsilon_decay    = self.epsilon_decay,
            clip_grad_norm   = self.clip_grad_norm,
            learning_rate    = self.learning_rate,
            discount         = self.discount_factor,
            memory_capacity  = self.memory_capacity,
            num_observations = self.num_observations, #Passing new rich state dimension
        )

    #=================================
    #--------Generate New Map---------
    #=================================
    def generate_new_map(self, mode, random, max_steps): #Creating a new map.
        if random: #If map is set to be randomly generated
            env = gym.make(ENV_NAME, is_slippery=False, reward_schedule=REWARD_SCHEDULE, desc=generate_random_map(size=MAP_SIZE, p=P_FROZEN), render_mode=mode, max_episode_steps=max_steps)
        else: #If map is not set to be randomly generated
            if MAP_SIZE > 8: #If map is above 8 in size, generate a map.
                env = gym.make(ENV_NAME, is_slippery=False, reward_schedule=REWARD_SCHEDULE, desc=MAP_NAME, render_mode=mode, max_episode_steps=max_steps)
            else: #If not, use one of the pre-made Gymnasium maps.
                env = gym.make(ENV_NAME, is_slippery=False, reward_schedule=REWARD_SCHEDULE, map_name=MAP_NAME, render_mode=mode, max_episode_steps=max_steps)
        return env

    #=================================
    #State Tensor Preprocessing Method
    #=================================
    def state_preprocess(self, state, cached_map=None): #Building a rich state tensor from a flat tile index (FrozenLake observation). Returns tensor layout (length = 4 + MAP_SIZE^2): [norm_row, norm_col, rel_goal_row, rel_goal_col, tile_0, ..., tile_n]
        grid_size = self.env.unwrapped.nrow
        
        #Decoding the flat index into rows and columns (row, col)
        row = state // grid_size
        col = state % grid_size

        #Normalised absolute position of the player [0, 1].
        player_pos = torch.tensor(
            [row / (grid_size - 1), col / (grid_size - 1)],
            dtype=torch.float32, device=DEVICE
        )

        raw_map  = self.env.unwrapped.desc #Map information (grid_size, grid_size), dtype bytes.
        
        goal_pos = np.argwhere(raw_map == b'G')[0] #Getting the goal positive.
        
        #Defining the (relative) distance between the player and the goal.
        rel_goal = torch.tensor(
            [
                (goal_pos[0] - row) / (grid_size - 1),
                (goal_pos[1] - col) / (grid_size - 1),
            ],
            dtype=torch.float32, device=DEVICE
        )

        #Encoding the full tile map as a flat float vector
        if cached_map is None: #Accepting a pre-built cached_map tensor. The tile encoding is computationally expensive, so it is computed at episode start and reused.
            cached_map = torch.tensor(
                [TILE_ENCODING[cell] for row_tiles in raw_map for cell in row_tiles],
                dtype=torch.float32, device=DEVICE)

        #Concatenating: [norm_row, norm_col, tile_0, ..., tile_(n-1)]
        return torch.cat([player_pos, rel_goal, cached_map])

    #=================================
    #----Building Map Cache Method----
    #=================================
    @staticmethod
    def build_map_cache(env): #Encode the map tile grid once at the start of each episode.
        raw_map = env.unwrapped.desc
        return torch.tensor(
            [TILE_ENCODING[cell] for row_tiles in raw_map for cell in row_tiles],
            dtype=torch.float32, device=DEVICE)

    #=================================
    #---------Training Method---------
    #=================================
    def train(self):
        total_steps = 0

        if PLOTTING:
            episodes_num = []
            rewards  = []

        start_time = perf_counter()

        #Training loop running for n number of training episodes.
        for episode in range(self.max_episodes):

            if GENERATE_MAP_PER_EPISODE: #If enabled, generate a new map per episode.
                self.env.close()
                self.env = self.generate_new_map(None, True, MAX_STEPS_TRAIN) #Now passing MAX_STEPS_TRAIN so training episodes are 200 steps.
                self.agent.action_space = self.env.action_space

            cached_map = self.build_map_cache(self.env) #Caching map encoding once per training episode. Saves on processing.

            state, _   = self.env.reset()
            state      = self.state_preprocess(state, cached_map)

            terminated      = False
            truncated       = False
            steps_taken     = 0
            episode_rewards = 0

            while not terminated and not truncated:
                #Select an action from the current state (since its training, that action will be chosen using epsilon-greedy)
                action = self.agent.select_action(state)

                #Do the action and get the new state obtained from taking it
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                #Creating a state tensor for the next_state
                next_state = self.state_preprocess(next_state, cached_map)

                #Store the new state obtained from the previous state and action as well as the obtained reward 
                self.n_step_buffer.add(state, action, reward, next_state, terminated, truncated) #Pushing the raw single-step transition into the n-step buffer. 
                #The buffer accumulates N steps, then stores a single transition with the discounted n-step return into the main replay buffer. Only when n transitions have been buffered does a transition enter the replay memory.

                if self.n_step_buffer.is_ready(): #When n-step return buffer is filled (len(N_STEP))
                    oldest_state, oldest_action, n_step_return, final_next_state, window_terminated = self.n_step_buffer.get() #Get n-step return from accumulated steps.
                    self.agent.replay_memory.store(oldest_state, oldest_action, final_next_state, n_step_return, window_terminated) #Store that n-step return as a single transition for training.

                total_steps     += 1
                steps_taken     += 1
                episode_rewards += reward
                
                self.agent.update_epsilon() #Epsilon now decays every step, not once per episode.

                #Agent learns every TRAIN_FREQ steps after LEARNING_STARTS
                if (total_steps > LEARNING_STARTS and len(self.agent.replay_memory) > self.batch_size and total_steps % TRAIN_FREQ == 0):
                    self.agent.learn(self.batch_size)

                #Target network updated every TARGET_UPDATE_STEPS instead of N episodes.
                if total_steps % TARGET_UPDATE_STEPS == 0: 
                    self.agent.hard_update()

                state = next_state

            self.n_step_buffer.flush_to(self.agent.replay_memory) #Flushing the remaining partial n-step windows at the end of the episode so the last (n-1) transitions are not discarded.

            #Printing episode stats.
            print(f"Episode: {episode + 1}, " f"Episode Steps: {steps_taken}, " f"Episode Rewards: {episode_rewards:.2f}, " f"Epsilon: {self.agent.epsilon_max:.4f}")

            #Saving the trained model every "save interval" (number of episodes).
            if (episode + 1) % self.save_interval == 0:
                print("\n==== Saving model parameters ====\n")
                self.agent.save(self.save_path + '_' + str(episode + 1) + '.pth')

            if PLOTTING:
                episodes_num.append(episode)
                rewards.append(episode_rewards)
        
        #Calculating elapsed training time.
        end_time = perf_counter()
        elapsed  = end_time - start_time
        print("Time: ", elapsed)
        with open('./perceptive-deep-q-learning/timeProfiling.log', 'w') as f:
            f.write(str(elapsed)) #Saving elapsed training time.

        if PLOTTING:
            #Plotting rewards against episodes on a graph.
            window = 10 #Rolling average window.
            plt.plot(episodes_num, rewards, color="steelblue", label="Rewards per Episode")
            plt.plot(episodes_num, rolling_average(rewards, window), color="red", label=f"Rolling Average (window={window})")
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.grid(linestyle='--', linewidth=0.5)
            plt.legend()
            plt.show()

            #Saving plot data to a csv file
            np.savetxt('./perceptive-deep-q-learning/training_graph_data.csv', np.column_stack([episodes_num, rewards]), delimiter=",", header="Episodes, Rewards", fmt='%s')

        self.env.close()

    #=================================
    #--------Evaluating Method--------
    #=================================
    def test(self):
        #Loading the weights of the trained policy network being tested.
        self.agent.main_network.load_state_dict(torch.load(self.rl_load_path)) #Copy the weights and biases from a save file to the policy network.
        self.agent.main_network.eval() #Sets the policy network to evaluation mode.

        episodes_num = []
        rewards      = []
        steps        = []
        success      = 0

        self.env = self.generate_new_map("rgb_array", GENERATE_MAP_PER_EPISODE, MAX_STEPS_TEST)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.env = RecordVideo(
                self.env,
                video_folder="./perceptive-deep-q-learning/recordings",
                name_prefix="vid_" + str(0) + "_",
                episode_trigger=lambda x: True
            )
        self.env = HumanRendering(self.env)

        #Testing loop over all evaluating/testing episodes.
        for episode in range(self.max_evaluating_episodes):
            state, _ = self.env.reset()
            cached_map = self.build_map_cache(self.env) #Caching map encoding once per test episode. Saves on processing.

            terminated      = False
            truncated       = False
            episode_steps   = 0
            episode_rewards = 0

            while not terminated and not truncated:
                state  = self.state_preprocess(state, cached_map)
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                state            = next_state
                episode_rewards += reward
                episode_steps   += 1

                if terminated and ((ENV_NAME == "FrozenLake-v1"   and reward == REWARD_SCHEDULE[0]) or (ENV_NAME == "CliffWalking-v1" and reward != -100)):
                    success += 1

            #Printing episode stats
            print(f"Episode: {episode + 1}, Steps: {episode_steps}, Rewards: {episode_rewards:.2f}")

            episodes_num.append(episode)
            rewards.append(episode_rewards)
            steps.append(episode_steps)

            #Regenerate map for the next evaluation episode if configured to do so.
            if GENERATE_MAP_PER_EPISODE:
                self.env.close()
                self.env = self.generate_new_map("rgb_array", True, MAX_STEPS_TEST)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.env = RecordVideo(
                        self.env,
                        video_folder="./perceptive-deep-q-learning/recordings",
                        name_prefix="vid_" + str(episode + 1) + "_",
                        episode_trigger=lambda x: True
                    )
                self.env = HumanRendering(self.env)

        print("Percentage success rate: ", (success / self.max_evaluating_episodes) * 100, "%")
        
        self.env.close() #Closing the rendering window after the episode is over.

        #Plotting the results on two graphs.
        window = 10 #Rolling average window.
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
        np.savetxt('./perceptive-deep-q-learning/evaluating_graph_data.csv', np.column_stack([episodes_num, rewards, steps]), delimiter=",", header="Episodes, Rewards, Steps", fmt='%s')

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
        average = sum(values) / len(values)
        with open('./perceptive-deep-q-learning/memoryProfiling.log', 'w') as f:
            for value in values:
                f.write(str(value) + "\n")
            f.write("Peak: " + str(peak) + "\n")
            f.write("Average: " + str(average) + "\n")
        self.results = (average, peak)


#=================================
#----------Main Function----------
#=================================
if __name__ == "__main__":
    #Creating and starting a thread to profile the memory usage.
    if MEMORY_PROFILING:
        end_profiler = Event()
        thread = Profiler()
        thread.start()
    
    #Creating a profile to time how long the train_agent function takes to run (how long it takes to train the agent).
    if SPEED_PROFILING:
        pr = cProfile.Profile()
        pr.enable()
    
    DRL = Model_TrainTest() #Defining the instance.

    if TRAINING_MODE:
        #Training the policy network
        DRL.train()
    else:
        #Testing and evaluating the policy network
        DRL.test()
    
    if SPEED_PROFILING:
        pr.disable()
        with open('./perceptive-deep-q-learning/timeDetailedProfiling.log', 'w') as f:
            pstats.Stats(pr, stream=f).strip_dirs().sort_stats("cumulative").print_stats()
        
    if MEMORY_PROFILING:
        end_profiler.set()
        thread.join()
