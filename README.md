# Cart-Pole Reinforcement Learning (This is a work in progress)

This project implements **the classic cart-pole balancing problem** without using Gym or pre-built physics for learning purposes.  
Both the **environment** and **reinforcement learning agent** were created **from scratch** to learn the underlying concepts.

The goal:  
A cart moves left/right to balance a pendulum upright. The agent must learn a policy that picks up the pole and prevents the pole from falling.

---

##  Features

- **Custom physics environment** (`cartpole_env.py`)
- **Neural Network policy** (`nn.py`) built with **PyTorch**
- **Actor-Critic training** (`train.py`)
- **Manual + Agent play mode** (`agent_play.py`)
- Adjustable reward functions and training hyperparameters
- Optional interactive human input while agent plays
- Optional curricular learning
- Training can be stopped at any point by pressing **CTRL+C**, the policy will be saved
- Learning curve is plotted after training 

---

##  Controls During Agent Play

| Key | Action |
|---|---|
| **A** / **←** | Push cart left |
| **D** / **→** | Push cart right |
| **SPACE** | Swap between manual and agent |

This allows you to assist or test the policy in real time.

---

## Setup Instructions

This project uses **Conda** to manage dependencies.
Make sure Conda (or Miniconda / Anaconda) is installed before running the setup script.

### 1. Create the Environment Automatically

Linux/MacOS  
`./setup.sh`  
Windows(Powershell)  
Open PowerShell in the project folder and run:  
`.\setup.ps1`  

This will:

- Create a conda environment named `cartpole`
- Install PyTorch, NumPy, Pygame, Matplotlib
- Activate the environment automatically

### 2. Activate the Environment Later

`conda activate cartpole`

### 3. Run Training

`python train.py`

To start training from scratch just remove or delete cartpole_policy.pth from the project folder.


### 4. Run the Trained Agent

`python agent_play.py`


If the model file `cartpole_policy.pth` exists, the agent will attempt to balance the pole to the best of its capabilities.  
Space toggles **manual override**, and arrow keys let you nudge the cart and play for yourself.

---

## Project Structure

- │── cartpole_env.py # Environment & physics simulation
- │── nn.py # Policy + value neural network
- │── train.py # Training script (Actor-Critic)
- │── agent_play.py # Visual play mode with optional human control
- │── cartpole_policy.pth # Saved model (after training)
- |── setup.sh # Sets up the conda environment on Linux
- |── setup.ps1 # Sets up the conda environment on Windows
- |── environment.yml # Conda environment


## How the Learning Works

This project uses **actor-critic reinforcement learning** and optionaly **curriculum learning**:

- **Policy network** outputs action probabilities
- **Value network** estimates expected return (baseline)
- **Advantage** = return − baseline  
  → Encourages actions better than expected  
  → Reduces variance, improves learning stability

The agent receives rewards for:
- Keeping the pole upright  

The agent is slightly penalized for:
- Fast movements of the pendulum and itself

The learning is done through a actor-critic network. Which means a value network tries to predict the future reward from the current state (Critic). The agent (Actor) tries to learn which is the best action to take in a state. After taking the action it gets a reward. If the reward is higher than the expected value the action was good. The difference between reward and prediction is called advantage. This makes learning more stable by letting the actor improve even in bad scenarious. It is like giving feedback to the network.

The training loop contains a randomization variable. This variable specifys how difficult the starting position at each episode can be. Randomization of 1 means that the pole starts basically all the time. The highest difficulty is 32, which means the pole can be at any position at the start of an episode. After reaching a mean reward of a certain amount over 20 episodes the difficulty increases. This is curricular learning. The agent learns to solve the easy environment first and then difficulty increases. During the developement I tried both curricular learning and normal learning. Non-curricular learning will solve the environment faster, but seemingly curricular learning will provide more robust agents.

I invite anyone who is interested in this project to try for themselves different reward functions to see the outcome of training an agent with curriculum learning or just normally. For this purpose at the end of the training a learning curve is created and plotted.
