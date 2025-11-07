# Cart-Pole Reinforcement Learning (From Scratch)

This project implements **the classic cart-pole balancing problem** without using Gym or pre-built physics.  
Both the **environment** and **reinforcement learning agent** were created **from scratch** to learn the underlying concepts.

The goal:  
A cart moves left/right to balance a pendulum upright. The agent must learn a policy that prevents the pole from falling.

---

##  Features

- **Custom physics environment** (`cartpole_env.py`)
- **Neural Network policy** (`nn.py`) built with **PyTorch**
- **REINFORCE / Actor-Critic training** (`train.py`)
- **Manual + Agent play mode** (`agent_play.py`)
- Adjustable reward functions and training hyperparameters
- Optional interactive human input while agent plays

---

##  Controls During Agent Play

| Key | Action |
|---|---|
| **A** / **←** | Push cart left |
| **D** / **→** | Push cart right |
| *(no key pressed)* | Agent controls the cart |
| **SPACE** | Toggle *manual override* on/off |

This allows you to assist or test the policy in real time.

---

## Project Structure

│── cartpole_env.py # Environment & physics simulation
│── nn.py # Policy + value neural network
│── train.py # Training script (Actor-Critic)
│── agent_play.py # Visual play mode with optional human control
│── cartpole_policy.pth # Saved model (after training)


## How the Learning Works

This project uses **actor-critic reinforcement learning**:

- **Policy network** outputs action probabilities
- **Value network** estimates expected return (baseline)
- **Advantage** = return − baseline  
  → Encourages actions better than expected  
  → Reduces variance, improves learning stability

The agent receives rewards for:
- Keeping the pole upright  
- Staying near the center  
- Avoiding falling or leaving bounds
