#==============================================================================================================
#Proximal Policy Optimization code by Mochanics based upon code snippets from the following people:
#Arun Nanda: https://www.datacamp.com/tutorial/proximal-policy-optimization (License not specified)
#Aleksandar Nikoloski: https://github.com/epsill0n/PPO-Implementation/blob/main/ppo.py (License not specified)
#Eric Yu: https://github.com/ericyangyu/PPO-for-Beginners/tree/master (MIT License)
#Bogdan Penkovsky: https://penkovsky.com/neural-networks/beyond/#introduction (License not specified)
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
        MAP_SIZE = int(sys.argv[2]) #Map size for frozen lake. Default static maps are 4x4 and 8x8. Any bigger map will be randomly generated using a seed (so it's also kept constant).
    else:
        print("Invalid Parameter")
        quit()
else:
    print("Invalid Parameter")
    quit()

#Defining all hyperparameters for PPO
GAMMA = 0.99 #Discount
LAMBDA = 0.95 #GAE parameter
CLIP_EPS = 0.2 #Actor/policy clipping
LR_ACTOR = 3e-4 #Learning rate of the actor network
LR_CRITIC = 3e-4 #Learning rate of the critic network
EPOCHS = 10 #Number of learning epochs per episode
BATCH_SIZE = 64 #Size of the batch of steps to be taken at random from the rollout buffer
ROLLOUT_STEPS = 256 #Current number of time steps to be recorded in a rollout buffer
ENTROPY_COEF = 0.05 #Randomness/uncertainty of the policy. The higher the entropy, the more random the agent actions are (similar probabilities for all actions). The lower the entropy, the more deterministic the agent is.
ENTROPY_DECAY = 0.9995 #How fast the entropy decreases.
MAX_EPISODES = 300 #The number of training episodes

#Environmental constants
ENV_NAME = "FrozenLake-v1" #Or alternatively "CliffWalking-v1"
REWARD_SCHEDULE = (2, -1, -0.01) #When to give a reward to the agent and how high that reward is. Only for "FrozenLake-v1" The values are: Reach Goal, Reach Hole, Reach Frozen (includes Start), respectively

#Additional non-PPO related constants
MAX_EVALUATING_EPISODES = 100 #Maximum number of testing/evaluating episodes
SAVE_INTERVAL = 100 #After how many training episodes should the weights and biases of the critic and actor neural networks be saved. 
DEVICE = torch.device( #Checking which devices are available and selecting it based on that
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

#Debug Constants
PLOTTING = False #Enable plotting of graphs
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

#=============================
#----Actor (Policy) Network---
#=============================
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(         
            nn.Linear(obs_dim, 64), #Input layer to 1st hidden layer (observation space)
            nn.ReLU(inplace=True), #ReLU (Rectified Linear Unit) activation function between input and 1st layers. This function only lets a neuron pass a value forward if its positive.
            nn.Linear(64, 64), #1st hidden layer to 2nd hidden layer
            nn.ReLU(inplace=True), #Same ReLU activation function between 1st and 2nd layers
            nn.Linear(64, action_dim), #2nd hidden layer to output layer (action space)
        )

        #Othogonal initialization
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        #Adding a gain of 0.01 for the last layer        
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        return self.net(x)

#=============================
#----Critic (Value) Network---
#=============================
class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), #Input layer to 1st hidden layer (observation space)
            nn.ReLU(inplace=True), #ReLU (Rectified Linear Unit) activation function between input and 1st layers. This function only lets a neuron pass a value forward if its positive.
            nn.Linear(64, 64), #1st hidden layer to 2nd hidden layer
            nn.ReLU(inplace=True), #Same ReLU activation function between 1st and 2nd layers
            nn.Linear(64, 1), #2nd hidden layer to output layer (action space)
        )
        
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        return self.net(x)

#=============================
#--GAE Advantage Calculation--
#=============================
def compute_gae(rewards, values, dones, next_value): #Function to calculate the Generalized Advantage Estimation (GAE): gae_t = delta_t + GAMMA * LAMBDA * gae_{t+1}
    advantages = [] #Array to store the resulting advantages
    gae = 0 #Gae value
    values = values + [next_value] #Appending next_value to values so that values[step+1] is always accessible for later (during bootstrapping).

    #Iterating backwards through the rollout (from last to first step)
    for step in reversed(range(len(rewards))): #For each step in a batch
        #Note: 1 - dones[step] is used to zero out delta and later the gae when an episode is termintated as there are no more rewards (dones = True = 1)
        delta = rewards[step] + GAMMA * values[step + 1] * (1 - dones[step]) - values[step] #1-step TD error: delta = r + gamma * V(s') - V(s). Describes how different/surprising that transition was compared to current estimates.
        gae = delta + GAMMA * LAMBDA * (1 - dones[step]) * gae #Calculating GAE from previous one. This is where the values[step+1] is used for the bootstrapping. This equation is the exponetially weighted sum of future TD errors.
        advantages.insert(0, gae) #Prepending to list. Since the iteration is backwards, the appending is backwards too.

    return advantages

#GAE estimates the advantage by using an exponentially decreasing weight every step away from the 

#=================================
#--------Training Function--------
#=================================
def train():
    start_time = perf_counter()

    entropy_coef = ENTROPY_COEF

    #Setting up the environment
    if ENV_NAME == "CliffWalking-v1":
        env = gym.make(ENV_NAME)
    elif ENV_NAME == "FrozenLake-v1":
        if MAP_SIZE > 8:
            env = gym.make(ENV_NAME, is_slippery = False, reward_schedule = REWARD_SCHEDULE, desc = MAP_NAME)
        else:
            env = gym.make(ENV_NAME, is_slippery = False, reward_schedule = REWARD_SCHEDULE, map_name = MAP_NAME)
    
    obs_dim = env.observation_space.n #Size of the observation space
    action_dim = env.action_space.n #Size of the action space

    actor = Actor(obs_dim, action_dim).to(DEVICE) #Creating the actor (value) network
    critic = Critic(obs_dim).to(DEVICE) #Creating the critic (value) network

    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR, eps=1e-5) #Using the adam optimizer. It is the standard one used for PPO in PyTorch.
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC, eps=1e-5) #Using the adam optimizer. It is the standard one used for PPO in PyTorch.

    if PLOTTING == True:
        episodes_num = []
        episode_rewards = []

    for episode in range(MAX_EPISODES):
        obs, _ = env.reset()
        obs_onehot = np.eye(obs_dim)[obs]

        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        episode_reward = 0       #Tracks the average rewards for the CURRENT episode only
        completed_episodes = []  #Stores each finished episode's total average rewards

        #Collect rollout
        for step in range(ROLLOUT_STEPS): #Recording interactions of the agent with the environement (state for a set number of steps) in rollout buffer
            terminated = False
            truncated = False
            
            state_tensor = torch.tensor(np.array(obs_onehot), dtype=torch.float32, device=DEVICE) #One-hot encoding the discrete state to feed to the neural network.
            
            with torch.no_grad(): #Disabling gradient tracking as only data is being collected. No training (gradient update) will be done here. Save on computation.
                #Note: logits are not probabilities as they don't sume to 1 and can be negative but they could be through the use of the softmax function.
                logits = actor(state_tensor) #Getting the Actor network raw logits outputs (unnormalized action scores). Logits are the raw numbers from a neural network.
                value = critic(state_tensor)  #Getting the Critic V(s_t) outputs. This is the estimated state value.

            dist = Categorical(logits=logits) #Using Categorial distribution to convert logits to a probability distribution (this is where softmax function resides)
            action = dist.sample() #Taking a random action (sample) based upon that probability distribution.

            next_obs, reward, terminated, truncated, _ = env.step(action.item()) #Taking a step in the environment by using the action sampled above.
            
            if ENV_NAME == "CliffWalking-v1" and reward <= -1: #Checking is truncated or not.
                truncated = True

            done = terminated or truncated #If truncated or terminated. Set done as true.

            #Storing transtitions in the rollout buffer.
            states.append(obs_onehot)
            actions.append(action.item())
            log_probs.append(dist.log_prob(action).item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            #Advancing to the next one hot state (s_{t+1})
            obs = next_obs
            obs_onehot = np.eye(obs_dim)[obs]

            episode_reward += reward

            if done: #If the episode ended mid-rollout, this records the episode rewards and reset the environment.
                completed_episodes.append(episode_reward)
                episode_reward = 0
                obs, _ = env.reset()
                obs_onehot = np.eye(obs_dim)[obs] #Getting one hot state.

        #Recording partial episode rewards if the rollout ended mid-episode (no terminal done).
        if episode_reward != 0:
            completed_episodes.append(episode_reward)

        #Computing next state value for GAE bootstapping.
        with torch.no_grad():
            next_state_tensor = torch.tensor(np.array(obs_onehot), dtype=torch.float32, device=DEVICE)
            next_value = critic(next_state_tensor).item()

        #Computing GAE advantages.
        advantages = compute_gae(rewards, values, dones, next_value)
        
        #Computing returns. Returns are the target for the critic network. R_t = A_t + V(s_t)
        returns = [adv + val for adv, val in zip(advantages, values)]

        #Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=DEVICE)
        old_log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(np.array(returns), dtype=torch.float32, device=DEVICE)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32, device=DEVICE)

        #Normalize advantages. This keeps the advantage values at the same scale across different rollouts.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        #PPO Update
        dataset_size = states.size(0)

        for _ in range(EPOCHS): #Number of training epochs (small clipped changes in policy).
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, BATCH_SIZE): #Running through the entire batch.
                batch_idx = indices[start:start+BATCH_SIZE] #Getting the current element id inside the batch.

                #Actor Update
                logits = actor(states[batch_idx]) #Gettting logits for the current batch element id.
                dist = Categorical(logits=logits) #Getting probability distribution from the logits.
                new_log_probs = dist.log_prob(actions[batch_idx]) #Getting log of probalities.
                entropy = dist.entropy().mean() #Getting the entropy from the probability distribution. This measures how spread out the distribution is. High entropy means high policy uncertainty.

                #--------------------------------------------------------------------------------------------------
                #Clip Objective Function
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx]) #Calculating te entropy ratio r_t. Ratio tells PPO how off from the old policy this new policy is.

                #Clip function sub-parts
                surr1 = ratio * advantages[batch_idx]  #The unclipped objective (direction policy should update towards)
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[batch_idx] #The clipped objectives. Prevents policy from changing too much.

                #Note: The reason for the negative sign is that PyTorch optimizers minimize but in this instance, maximization of the objective is needed.
                policy_loss = -torch.min(surr1, surr2).mean() #Finding the minimum between the clipped objective and the unclipped objective.
                entropy_loss = -entropy_coef * entropy #Calculating entropy loss. This tells how encouraged exploration is and will decay as entropy decays over time making the policy more deterministic over training steps. The negative sign implies a maximization of entropy.
                actor_loss = policy_loss + entropy_loss #Total actor loss = policy loss + entropy regularisation
                #--------------------------------------------------------------------------------------------------

                optimizer_actor.zero_grad() #Zero the gradient from previous epoch.
                actor_loss.backward() #Calculate new gradients form the loss.
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5) #Limiting the maximum gradient so there is no overshoots.
                optimizer_actor.step() #Run one step of gradient update through the actor network (backpropagation using the Adam algorithm).

                #Critic Update
                value = critic(states[batch_idx]).squeeze() #Getting one value from the batch.
                critic_loss = (returns[batch_idx] - value).pow(2).mean() #Computing the MSE loss.
                
                critic_loss = 0.5 * critic_loss #Done originally in the PPO paper, halfing the critic loss.

                optimizer_critic.zero_grad() #Zero the gradient from previous epoch.
                critic_loss.backward() #Calculate new gradients form the loss.
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5) #Limiting the maximum gradient so there is no overshoots.
                optimizer_critic.step() #Run one step of gradient update through the critic network (backpropagation using the Adam algorithm).

        entropy_coef = max(0.01, entropy_coef * ENTROPY_DECAY)
        
        avg_reward = np.mean(completed_episodes) if completed_episodes else 0.0
        
        #Printing episode stats
        print(f"Episode: {episode + 1}, " f"Episode Reward: {avg_reward:.4f}, " f"Entropy Coef: {entropy_coef:.4f}")
        
        if (episode + 1) % SAVE_INTERVAL == 0:
            print("\n==== Saving model parameters ====\n")
            torch.save(actor.state_dict(), "./proximal-policy-optimization/models/" + f'actor_{episode + 1}' + ".pth")
            torch.save(critic.state_dict(), "./proximal-policy-optimization/models/" + f'critic_{episode + 1}' + ".pth")
        
        if PLOTTING == True:
            episodes_num.append(episode)
            episode_rewards.append(avg_reward)
    
    end_time = perf_counter()
    time = end_time - start_time
    print("Time: ", time)
    with open('./proximal-policy-optimization/timeProfiling.log', 'w') as f:
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
        np.savetxt('./proximal-policy-optimization/training_graph_data.csv', np.column_stack([episodes_num, episode_rewards]), delimiter=",", header="Episodes, Rewards", fmt='%s')
        
    env.close()
    return actor, critic

#=================================
#-------Evaluating Function-------
#=================================
def evaluate(model, episodes):
    #Setting up the environment
    if ENV_NAME == "CliffWalking-v1":
        env = gym.make(ENV_NAME, render_mode="rgb_array")
    elif ENV_NAME == "FrozenLake-v1":
        if MAP_SIZE > 8:
            env = gym.make(ENV_NAME, is_slippery = False, reward_schedule = REWARD_SCHEDULE, desc = MAP_NAME, render_mode="rgb_array")
        else:
            env = gym.make(ENV_NAME, is_slippery = False, reward_schedule = REWARD_SCHEDULE, map_name = MAP_NAME, render_mode="rgb_array")

    with warnings.catch_warnings(action="ignore"):
        env = RecordVideo(
            env,
            video_folder="./proximal-policy-optimization/recordings", # Folder to save videos
            name_prefix="vid",               #Prefix for video filenames
            episode_trigger=lambda x: True    #Record every episode
        )

    env = HumanRendering(env)
    
    obs_dim = env.observation_space.n

    episodes_num = []
    rewards = []
    steps = []
    success = 0

    for episode in range(episodes):
        obs, _ = env.reset()
        obs_onehot = np.eye(obs_dim)[obs]

        terminated = False
        truncated = False
        episode_rewards = 0
        episode_steps = 0

        while not terminated and not truncated:
            state_tensor = torch.tensor(np.array(obs_onehot), dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                logits = model(state_tensor)

            #Deterministic action (greedy)
            action = torch.argmax(logits).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)

            episode_rewards += reward
            episode_steps += 1

            obs = next_obs
            obs_onehot = np.eye(obs_dim)[obs]
            
            if terminated and ((ENV_NAME == "FrozenLake-v1" and reward == REWARD_SCHEDULE[0]) or (ENV_NAME == "CliffWalking-v1" and reward != -100)):
                success += 1

        print(f"Episode: {episode + 1}, " f"Steps: {episode_steps:}, " f"Reward: {episode_rewards:.2f}")

        episodes_num.append(episode)
        rewards.append(episode_rewards)
        steps.append(episode_steps)     
    
    print("Percentage success rate: ", (success/episodes)*100, "%")
    
    env.close()
    
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
    np.savetxt('./proximal-policy-optimization/evaluating_graph_data.csv', np.column_stack([episodes_num, rewards, steps]), delimiter=",", header="Episodes, Rewards, Steps", fmt='%s')

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
        
        with open( './proximal-policy-optimization/memoryProfiling.log', 'w' ) as f:
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
            with open("./proximal-policy-optimization/timeDetailedProfiling.log", 'w') as f:
                pstats.Stats( pr, stream=f ).strip_dirs().sort_stats("cumulative").print_stats()
    else: #If not in training mode
        #--------------------------------------------------------------------------------------------------
        #Load get environment observation space and action space sizes.
        if ENV_NAME == "CliffWalking-v1":
            env = gym.make(ENV_NAME)
        elif ENV_NAME == "FrozenLake-v1":
            if MAP_SIZE > 8:
                env = gym.make(ENV_NAME, is_slippery = False, reward_schedule = REWARD_SCHEDULE, desc = MAP_NAME)
            else:
                env = gym.make(ENV_NAME, is_slippery = False, reward_schedule = REWARD_SCHEDULE, map_name = MAP_NAME)
        
        loaded_model = Actor(obs_dim = env.observation_space.n, action_dim = env.action_space.n).to(DEVICE)
        env.close()
        #--------------------------------------------------------------------------------------------------
        
        #Load and evaluate trained model
        loaded_model.load_state_dict(torch.load("./proximal-policy-optimization/models/actor_" + str(MAX_EPISODES) + ".pth")) #Load the latest saved actor neural network weights and biases
        evaluate(loaded_model, MAX_EVALUATING_EPISODES) #Evaluate the model 10 times (number of episodes)
   
    #Ending the memory profiling thread by setting a flag and then ending the thread.
    if MEMORY_PROFILING == True:
        end_profiler.set()
        thread.join() 
