#==============================================================================================================
#CNN-Based Perceptive Proximal Policy Optimization code by Mochanics based upon code snippets from the following people:
#Arun Nanda: https://www.datacamp.com/tutorial/proximal-policy-optimization (License not specified)
#Aleksandar Nikoloski: https://github.com/epsill0n/PPO-Implementation/blob/main/ppo.py (License not specified)
#Eric Yu: https://github.com/ericyangyu/PPO-for-Beginners/tree/master (MIT License)
#Bogdan Penkovsky: https://penkovsky.com/neural-networks/beyond/#introduction (License not specified)
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
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
import sys
import warnings
from gymnasium.envs.toy_text.frozen_lake import generate_random_map #Only used when needing to generate a random map (MAP_NAME) for "FrozenLake-v1"

#Imports for profiling memory allocation and timing the training algorithm
from threading import Event, Thread
from time import sleep, perf_counter
import tracemalloc
import cProfile
import pstats
from gymnasium.wrappers import RecordVideo, HumanRendering

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

#Defining all hyperparameters for PPO
GAMMA = 0.99 #Discount.
LAMBDA = 0.95 #GAE parameter.
CLIP_EPS = 0.2 #Actor/policy clipping.
LR_ACTOR = 3e-4 #Learning rate of the actor network.
LR_CRITIC = 3e-4 #Learning rate of the critic network.
EPOCHS = 10 #Number of learning epochs per episode.
BATCH_SIZE = 256 #Size of the batch of steps to be taken at random from the rollout buffer
ROLLOUT_STEPS = 1024 #256 #Current number of time steps to be recorded in a rollout buffer
ENTROPY_COEF_MAX = 0.15 #Maximum entropy or randomness/uncertainty of the policy. The higher the entropy, the more random the agent actions are (similar probabilities for all actions). The lower the entropy, the more deterministic the agent is.
CURRICULUM_ENTROPY = 0.1 #Entropy boost/rest. It is the value the entropy variable gets when the curriculum episode count is reached (CURRICULUM_START).
ENTROPY_COEF_MIN = 0.05 #Minimum entropy at the end of training.
ENTROPY_DECAY = 0.9995 #How fast the entropy decreases.
DETERMINISTIC = True #Whether the agent selects actions at random based on probabilities of which one is the best (non-deterministic) or always takes the best action (determinsitic) during testing.
MAX_EPISODES = 8000 #The number of training episodes..

#Environmental constants
ENV_NAME = "FrozenLake-v1" #Only this environement is used for this version of PPO.
P_FROZEN = 0.9 #Probability of a tile being frozen. Used by the random map generation function.
REWARD_SCHEDULE = (1, -1, -0.01) #When to give a reward to the agent and how high that reward is. The values are: Reach Goal, Reach Hole, Reach Frozen (includes Start), respectively
TILE_ENCODING = {b'S': 0.0, b'F': 0.0, b'H': -1.0, b'G': 1.0} #Tile encoding for rich state: S (start) = 0.00, F (frozen) = 0.00, H (hole) = -1.00, G (goal) = 1.00. Those values were picked arbitrarily
GENERATE_MAP_PER_EPISODE = True #Parameter specific to perceptive models. Generates a new map every training episode to learn to generalize hole avoiding behaviour.
CURRICULUM_START = 2000  #Start of when maps are being randomly generated. The probability of them being randomly generated increases until CURRICULUM_END.
CURRICULUM_END = 3000    #When ALL maps are randomly generated.
MAX_STEPS_TRAIN = 100 #Maximum number of steps per training episode.

#Additional non-PPO related constants
MAX_EVALUATING_EPISODES = 100 #Maximum number of testing/evaluating episodes.
SAVE_INTERVAL = 100 #After how many training episodes should the weights and biases of the critic and actor neural networks be saved. 
DEVICE = torch.device( #Checking which devices are available and selecting it based on that.
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

#Debug Constants
PLOTTING = True #Enable plotting of graphs
MEMORY_PROFILING = False #Enable the profiling of memory. FOR ACCURATE RESULTS, DO NOT USE AT THE SAME TIME AS SPEED_PROFILING
SPEED_PROFILING = False #Enable training time measurement. FOR ACCURATE RESULTS, DO NOT USE AT THE SAME TIME AS MEMORY_PROFILING
REPRODUCIBLE = False #Makes the training and evaluating repeatable

#Seed everything for reproducible results
if REPRODUCIBLE:
    SEED = 2024 #The actual seed used (can be changed as needed)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
#-Get Probability of a Random Map-
#=================================
def get_map_randomness_prob(episode): #Gets the probability of the episode's map being randomly generated or not (used during the transition phase between fixed map and randomly generated map) when GENERATE_MAP_PER_EPISODE = True
    if episode < CURRICULUM_START:
        return 0.0
    elif episode >= CURRICULUM_END:
        return 1.0
    else:
        # Linear interpolation between 0 and 1
        return (episode - CURRICULUM_START) / (CURRICULUM_END - CURRICULUM_START)

#=================================
#--------Generate New Map---------
#=================================
def generate_new_map(mode, random): #Setting up the environment
    if random:
        env = gym.make(ENV_NAME, is_slippery=False, reward_schedule=REWARD_SCHEDULE, desc=generate_random_map(size=MAP_SIZE, p=0.9), render_mode=mode, max_episode_steps = MAX_STEPS_TRAIN)
    else:
        if MAP_SIZE > 8:
            env = gym.make(ENV_NAME, is_slippery=False, reward_schedule=REWARD_SCHEDULE, desc=MAP_NAME, render_mode=mode, max_episode_steps = MAX_STEPS_TRAIN)
        else:
            env = gym.make(ENV_NAME, is_slippery=False, reward_schedule=REWARD_SCHEDULE, map_name=MAP_NAME, render_mode=mode, max_episode_steps = MAX_STEPS_TRAIN)
    return env

#=================================
#------State Preprocessing--------
#=================================
def state_preprocess(obs, env, cached_map=None): #Building a rich state tensor from a flat tile index (FrozenLake observation). Returns tensor layout (length = 4 + MAP_SIZE^2): [norm_row, norm_col, rel_goal_row, rel_goal_col, tile_0, ..., tile_n]
    grid_size = env.unwrapped.nrow #Get grid_size information from env
    row = obs // grid_size #Number of map rows
    col = obs % grid_size #Number of map columns

    #Finding goal position
    raw_map = env.unwrapped.desc #Get raw map tile information from env
    goal_pos = np.argwhere(raw_map == b'G')[0]  #Get where the goal is (goal_row, goal_col)

    #Getting normalised absolute position (where the agent is in the map)
    player_pos = np.array([row / (grid_size - 1), col / (grid_size - 1)], dtype=np.float32)

    #Getting normalised relative displacement to goal (distance to goal)
    rel_goal = np.array([(goal_pos[0] - row) / (grid_size - 1), (goal_pos[1] - col) / (grid_size - 1)], dtype=np.float32)

    #Encoding the full tile map as a flat float vector          
    if cached_map is None: #Accepting a pre-built cached_map tensor. The tile encoding is computationally expensive, so it is computed at episode start and reused.
        cached_map = np.array([TILE_ENCODING[cell] for row_tiles in raw_map for cell in row_tiles], dtype=np.float32)

    #Concatenating: [norm_row, norm_col, tile_0, ..., tile_(n-1)]
    return np.concatenate([player_pos, rel_goal, cached_map])

#=================================
#---Building Map Cache Function---
#=================================
def build_map_cache(env): #Encoding the map tile grid once at the start of each episode.
    raw_map = env.unwrapped.desc
    return np.array([TILE_ENCODING[cell] for row_tiles in raw_map for cell in row_tiles], dtype=np.float32)

#=============================
#--CNN Actor (Policy) Network-
#=============================
class Actor(nn.Module):
    def __init__(self, action_dim, grid_size):
        super().__init__()
        
        # CNN processes the map spatially
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, grid_size, grid_size) -> (16, grid_size, grid_size)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (16, grid_size, grid_size) -> (32, grid_size, grid_size)
            nn.ReLU(inplace=True),
        )
        
        self.grid_size = grid_size
        cnn_out_dim = 32 * grid_size * grid_size

        # MLP head combines CNN features with player position and rel_goal
        self.net = nn.Sequential(
            nn.Linear(cnn_out_dim + 4, 128),  # +4 for player_pos(2) and rel_goal(2)
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

        # Orthogonal initialization for linear layers
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)  # add batch dimension

        pos_and_goal = x[:, :4]                          # [batch, 4]
        map_flat     = x[:, 4:]                          # [batch, grid_size^2]
        map_2d       = map_flat.view(-1, 1, self.grid_size, self.grid_size)  # [batch, 1, grid_size, grid_size]

        cnn_features = self.cnn(map_2d).view(x.size(0), -1)  # [batch, cnn_out_dim]
        combined     = torch.cat([pos_and_goal, cnn_features], dim=1)
        out          = self.net(combined)

        return out.squeeze(0) if single else out  # restore original shape if single sample


#=============================
#--CNN Critic (Value) Network-
#=============================
class Critic(nn.Module):
    def __init__(self, grid_size):
        super().__init__()

        # CNN processes the map spatially
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.grid_size = grid_size
        cnn_out_dim = 32 * grid_size * grid_size

        # MLP head combines CNN features with player position and rel_goal
        self.net = nn.Sequential(
            nn.Linear(cnn_out_dim + 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        pos_and_goal = x[:, :4]
        map_flat     = x[:, 4:]
        map_2d       = map_flat.view(-1, 1, self.grid_size, self.grid_size)

        cnn_features = self.cnn(map_2d).view(x.size(0), -1)
        combined     = torch.cat([pos_and_goal, cnn_features], dim=1)
        out          = self.net(combined)

        return out.squeeze(0) if single else out

#=============================
#--GAE Advantage Calculation--
#=============================
def compute_gae(rewards, values, dones, next_value): #Function to calculate the Generalized Advantage Estimation (GAE)
    advantages = [] #Array to store the resulting advantages
    gae = 0 #Gae value
    values = values + [next_value] #What

    for step in reversed(range(len(rewards))): #For each step in a batch  
        delta = rewards[step] + GAMMA * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + GAMMA * LAMBDA * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        if dones[step]: #ADDED: reset gae at episode boundaries so that returns
            gae = 0 #ADDED: from one episode never bleed into a previous one.

    return advantages

#GAE estimates the advantage by using an exponentially decreasing weight every step away from the 

#=================================
#--------Training Function--------
#=================================
#For more detailed comments on the PPO training function, see the original one in "proximal-policy-optimization.py".
#They are mostly the same except with the addition of a rich state tensor that is preprocessed using state_preprocess().
def train():
    start_time = perf_counter()

    entropy_coef = ENTROPY_COEF_MAX
    epochs = EPOCHS

    env = generate_new_map(None, False)
   
    action_dim = env.action_space.n #Size of the action space
    grid_size = env.unwrapped.nrow
    
    actor  = Actor(action_dim=action_dim, grid_size=grid_size).to(DEVICE)
    critic = Critic(grid_size=grid_size).to(DEVICE)

    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR, eps=1e-5)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC, eps=1e-5)

    if PLOTTING == True:
        episodes_num = []
        episode_rewards = []

    obs, _ = env.reset()
    cached_map = build_map_cache(env) #Caching map so that the tile encoding isn't done every step (saves on computing)
    obs_state = state_preprocess(obs, env, cached_map) #state_preprocess returns a rich state vector for FrozenLake

    for episode in range(MAX_EPISODES):
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        frac = 1.0 - episode / MAX_EPISODES
        for param_group in optimizer_actor.param_groups:
            param_group['lr'] = LR_ACTOR * frac
        for param_group in optimizer_critic.param_groups:
            param_group['lr'] = LR_CRITIC * frac
        
        if GENERATE_MAP_PER_EPISODE == True:
            if episode == CURRICULUM_START:
                entropy_coef = CURRICULUM_ENTROPY
                epochs = 4
                print("\n==== Switching to random maps ====\n")
        
            randomness_prob = get_map_randomness_prob(episode)
            env.close()
            if np.random.random() < randomness_prob:
                env = generate_new_map(None, True)
            else:
                env = generate_new_map(None, False)
        
        obs, _    = env.reset()
        cached_map = build_map_cache(env) #Caching map so that the tile encoding isn't done every step (saves on computing)
        obs_state = state_preprocess(obs, env, cached_map)

        episode_reward = 0       #Tracks the average rewards for the CURRENT episode only
        episode_steps = 0
        completed_episodes = []  #Stores each finished episode's total average rewards
  
        #Collect rollout
        for step in range(ROLLOUT_STEPS): #Recording interactions of the agent with the environment in the rollout buffer
            terminated = False
            truncated = False
            
            state_tensor = torch.tensor(obs_state, dtype=torch.float32, device=DEVICE)
            
            with torch.no_grad():
                logits = actor(state_tensor)
                value = critic(state_tensor)

            dist = Categorical(logits=logits)
            action = dist.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action.item())

            done = terminated or truncated

            states.append(obs_state)
            actions.append(action.item())
            log_probs.append(dist.log_prob(action).item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            obs = next_obs
            obs_state = state_preprocess(obs, env, cached_map) #Rich state for the next step.

            episode_reward += reward
            episode_steps += 1

            if done:
                completed_episodes.append(episode_reward)
                episode_reward = 0
                episode_steps = 0
                
                obs, _ = env.reset()         
                obs_state = state_preprocess(obs, env, cached_map) #Rich state for the next step.
                
        if episode_steps > 0:
            completed_episodes.append(episode_reward)

        #Computing next value for GAE
        with torch.no_grad():
            next_state_tensor = torch.tensor(obs_state, dtype=torch.float32, device=DEVICE)
            next_value = critic(next_state_tensor).item()

        advantages = compute_gae(rewards, values, dones, next_value)
        returns = [adv + val for adv, val in zip(advantages, values)]

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=DEVICE)
        old_log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(np.array(returns), dtype=torch.float32, device=DEVICE)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32, device=DEVICE)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        dataset_size = states.size(0)

        for _ in range(epochs):
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, BATCH_SIZE):
                batch_idx = indices[start:start+BATCH_SIZE] #Getting a batch of transitions

                # ----- Actor Update -----
                logits = actor(states[batch_idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy = dist.entropy().mean()

                #Clip Objective Function
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])

                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[batch_idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy_coef * entropy 
                actor_loss = policy_loss + entropy_loss

                #Making an optimiztion pass for the actor neural network
                optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5) #Limiting the maximum gradient so there is no overshoots
                optimizer_actor.step()

                # ----- Critic Update -----
                value = critic(states[batch_idx]).squeeze()
                critic_loss = (returns[batch_idx] - value).pow(2).mean() #MSE loss
                
                critic_loss = 0.5 * critic_loss #Done originally in the PPO paper, halving the critic loss

                #Making an optimiztion pass for the critic neural network
                optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5) #Limiting the maximum gradient so there is no overshoots
                optimizer_critic.step()
        
        avg_reward = np.mean(completed_episodes) if completed_episodes else 0.0
        
        entropy_coef = max(ENTROPY_COEF_MIN, entropy_coef * ENTROPY_DECAY)
        
        #Printing episode stats
        print(f"Episode: {episode + 1}, " f"Episode Average Reward: {avg_reward:.4f}, " f"Entropy Coef: {entropy_coef:.4f}")
        
        if (episode + 1) % SAVE_INTERVAL == 0:
            print("\n==== Saving model parameters ====\n")
            torch.save(actor.state_dict(), "./perceptive-proximal-policy-optimization/models/" + f'actor_{episode + 1}' + ".pth")
            torch.save(critic.state_dict(), "./perceptive-proximal-policy-optimization/models/" + f'critic_{episode + 1}' + ".pth")
        
        if PLOTTING == True:
            episodes_num.append(episode)
            episode_rewards.append(avg_reward)
    
    end_time = perf_counter()
    time = end_time - start_time
    print("Time: ", time)
    with open('./perceptive-proximal-policy-optimization/timeProfiling.log', 'w') as f:
        f.write(str(time))
    
    if PLOTTING == True:
        #Plotting rewards against episodes on a graph.
        window = 10 #Rolling average window.
        plt.plot(episodes_num, episode_rewards, color="steelblue", label="Rewards per Episode")
        plt.plot(episodes_num, rolling_average(episode_rewards, window), color="red", label=f"Rolling Average (window={window})")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.grid(linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

        #Saving plot data to a csv file
        np.savetxt('./perceptive-proximal-policy-optimization/training_graph_data.csv', np.column_stack([episodes_num, episode_rewards]), delimiter=",", header="Episodes, Rewards", fmt='%s') 

    env.close()
    return actor, critic

#=================================
#-------Evaluating Function-------
#=================================
def evaluate(model, episodes):
    #Setting up the environment
    env = generate_new_map("rgb_array", GENERATE_MAP_PER_EPISODE)

    with warnings.catch_warnings(action="ignore"):
        env = RecordVideo(
            env,
            video_folder="./perceptive-proximal-policy-optimization/recordings",
            name_prefix="vid_" + str(0) + "_",
            episode_trigger=lambda x: True
        )

    #env = HumanRendering(env)

    episodes_num = []
    rewards = []
    steps = []
    success = 0

    for episode in range(episodes):
        obs, _ = env.reset()
        cached_map = build_map_cache(env) 
        obs_state = state_preprocess(obs, env, cached_map) #Using the same rich state preprocessing as training

        terminated = False
        truncated = False
        episode_steps = 0
        episode_rewards = 0

        while not terminated and not truncated:
            state_tensor = torch.tensor(obs_state, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits = model(state_tensor)

            #Select action from model
            if DETERMINISTIC:
                action = torch.argmax(logits).item() #Better for fixed maps
            else:
                #Non-deterministic output gives a better result than deterministic for randomly generated maps deter which is deterministic greedy action.
                dist = Categorical(logits=logits)
                action = dist.sample().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)

            episode_rewards += reward
            episode_steps += 1

            obs = next_obs
            obs_state = state_preprocess(obs, env, cached_map) #Rich state for the next step.
            
            if terminated and ((ENV_NAME == "FrozenLake-v1" and reward == REWARD_SCHEDULE[0]) or (ENV_NAME == "CliffWalking-v1" and reward != -100)):
                success += 1

        print(f"Episode: {episode + 1}, " f"Steps: {episode_steps:}, " f"Reward: {episode_rewards:.2f}")

        episodes_num.append(episode)
        rewards.append(episode_rewards)
        steps.append(episode_steps)
        
        if GENERATE_MAP_PER_EPISODE == True: #Generates a new map per evaluating episode
            env.close()
            env = generate_new_map("rgb_array", GENERATE_MAP_PER_EPISODE)
            
            with warnings.catch_warnings(action="ignore"):
                env = RecordVideo(
                    env,
                    video_folder="./perceptive-proximal-policy-optimization/recordings",
                    name_prefix="vid_" + str(episode + 1) + "_",
                    episode_trigger=lambda x: True
                )

            #env = HumanRendering(env)
    
    print("Percentage success rate: ", (success/episodes)*100, "%")
    
    env.close() #Closing the rendering window after the episode is over   
    
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
    np.savetxt('./perceptive-proximal-policy-optimization/evaluating_graph_data.csv', np.column_stack([episodes_num, rewards, steps]), delimiter=",", header="Episodes, Rewards, Steps", fmt='%s')

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
        
        with open('./perceptive-proximal-policy-optimization/memoryProfiling.log', 'w') as f:
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
    
        #Train model
        actor, critic = train()
        
        if SPEED_PROFILING == True:
            pr.disable()
            with open("./perceptive-proximal-policy-optimization/timeDetailedProfiling.log", 'w') as f:
                pstats.Stats(pr, stream=f).strip_dirs().sort_stats("cumulative").print_stats()
    else: #If not in training mode
        #--------------------------------------------------------------------------------------------------
        #Load environment to get observation space and action space sizes for building the network
        env = generate_new_map(None, GENERATE_MAP_PER_EPISODE)
        
        # obs_dim must match training — use get_obs_dim() so the loaded network
        # has the same input size as the one that was saved
        grid_size = env.unwrapped.nrow
        loaded_model = Actor(action_dim=env.action_space.n, grid_size=grid_size).to(DEVICE)
        env.close()
        
        env = generate_new_map(None, GENERATE_MAP_PER_EPISODE)
        #--------------------------------------------------------------------------------------------------
        
        #Load and evaluate trained model
        loaded_model.load_state_dict(torch.load("./perceptive-proximal-policy-optimization/models/actor_" + str(MAX_EPISODES) + ".pth"))
        loaded_model.eval()
        evaluate(loaded_model, MAX_EVALUATING_EPISODES)
   
    #Ending the memory profiling thread by setting a flag and then ending the thread.
    if MEMORY_PROFILING == True:
        end_profiler.set()
        thread.join()
