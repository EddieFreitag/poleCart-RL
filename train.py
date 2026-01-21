import torch
from torch.distributions import Categorical
from cartpole_env import cartpole_env
from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys
from tqdm import tqdm


episode_rewards = []
episode_log_probs = []
episode_values = []   
difficulty_updates = []
rewards = []

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
episodes = 3000

# Initialize
env = cartpole_env()
policy = NeuralNetwork()
try:
    policy.load_state_dict(torch.load('cartpole_policy.pth'))
    print("Loaded trained model.")
except FileNotFoundError:
    print("No trained model found, starting fresh.")
policy.train()  # set network to training mode
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

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

def plot_learning_curve(rewards, difficulty_updates):
    median = np.median(rewards)
    mean = np.mean(rewards)
    std = np.std(rewards)
    max = np.max(rewards)
    print(f"Max Reward: {max:.2f}, Mean Reward: {mean:.2f}, Std Dev: {std:.2f}, Median Reward: {median:.2f}")

    # compute rolling average for plotting
    window_size = 50
    rolling_averages = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    # plot rewards
    plt.scatter(range(len(rewards)), rewards, alpha=0.1, label='Episode Reward')
    plt.axhline(y=mean, color='orange', linestyle='--', label='Mean Reward')
    plt.axhline(y=median, color='g', linestyle='--', label='Median Reward')
    plt.plot(range(window_size - 1, len(rewards)), rolling_averages, color='r', label='Rolling Average (50 eps)')
    for i, ep in enumerate(difficulty_updates):
        plt.axvline(
            x=ep,
            color='blue',
            linestyle=':',
            alpha=0.7,
            linewidth=1.5,
            label='Difficulty Increase' if i == 0 else None
        )
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

# ---- SAFE SAVE ON INTERRUPT ----
def save_model():
    torch.save(policy.state_dict(), "cartpole_policy.pth")
    print("\nModel saved to cartpole_policy.pth")

def handle_interrupt(sig, frame):
    print("\nTraining interrupted by user.")
    save_model()
    plot_learning_curve(rewards, difficulty_updates)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

def main():
    r = 32  # reset randomization factor
    max_diff = 32  # maximum difficulty level
    print(f"Starting training with randomization factor: {r}")
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset(r=r)
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = select_action(state)
            next_state, reward, done = env.step(action, step=step)   # <-- FIXED
            step += 1

            episode_rewards.append(reward)                # <-- KEEP
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        finish_episode()                                  # <-- uses stored log_probs + values
        mod = 20
        if episode % mod == 0 and episode != 0:
            mean_reward = np.mean(rewards[-mod:])
            tqdm.write(f"Episode {episode} | Mean Reward: {mean_reward:.2f}")
            if mean_reward >= 350 and r < max_diff:
                r += 1  # increase randomization factor
                tqdm.write(f"Increasing difficulty to lvl: {r}/{max_diff} !")
                difficulty_updates.append(episode)
            elif r == max_diff:
                tqdm.write("Max difficulty reached.")
                

        
    torch.save(policy.state_dict(), 'cartpole_policy.pth')
    print("Training complete. Model saved as cartpole_policy.pth")
    print(f"Latest difficulty lvl: {r}/{max_diff}")
    plot_learning_curve(rewards, difficulty_updates)

if __name__ == "__main__":
    main()