import torch
from torch.distributions import Categorical
from cartpole_env import cartpole_env
from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
episodes = 2000

# Initialize
env = cartpole_env()
policy = NeuralNetwork()
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

episode_rewards = []
episode_log_probs = []
episode_values = []   


def select_action(state):
    state = torch.tensor(state, dtype=torch.float32)
    probs, value = policy(state)

    dist = Categorical(probs)
    action = dist.sample()

    episode_log_probs.append(dist.log_prob(action))
    episode_values.append(value) 

    return action.item()


def finish_episode():
    returns = []
    G = 0

    for r in reversed(episode_rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    log_probs = torch.stack(episode_log_probs)
    values = torch.stack(episode_values).squeeze()

    advantages = returns - values.detach()

    policy_loss = -(log_probs * advantages).mean()
    value_loss = (returns - values).pow(2).mean()

    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_log_probs.clear()
    episode_rewards.clear()
    episode_values.clear()

    # ---- SAFE SAVE ON INTERRUPT ----
def save_model():
    torch.save(policy.state_dict(), "cartpole_policy.pth")
    print("\nModel saved to cartpole_policy.pth")

def handle_interrupt(sig, frame):
    print("\nTraining interrupted by user.")
    save_model()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

def main():
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = select_action(state)
            next_state, reward, done = env.step(action)   # <-- FIXED
            step += 1

            episode_rewards.append(reward)                # <-- KEEP
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        finish_episode()                                  # <-- uses stored log_probs + values

        if episode % 50 == 0 and episode != 0:
            print(f"Episode {episode} | Mean Reward: {np.mean(rewards[-50:])}")

        
    torch.save(policy.state_dict(), 'cartpole_policy.pth')
    print("Training complete. Model saved as cartpole_policy.pth")

    median = np.median(rewards)
    mean = np.mean(rewards)
    std = np.std(rewards)
    max = np.max(rewards)
    print(f"Max Reward: {max}, Mean Reward: {mean}, Std Dev: {std}, Median Reward: {median}")

    # plot rewards
    plt.plot(rewards)
    plt.axhline(y=mean, color='r', linestyle='--', label='Mean Reward')
    plt.axhline(y=median, color='g', linestyle='--', label='Median Reward')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

if __name__ == "__main__":
    main()