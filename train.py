import torch
from torch.distributions import Categorical
from cartpole_env import cartpole_env
from nn import NeuralNetwork
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


def select_action(state):
    state = torch.tensor(state, dtype=torch.float32)
    probs = policy(state)
    dist = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def finish_episode():
    G = 0
    returns = []

    # compute discounted returns backwards
    for r in reversed(episode_rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    loss = 0
    for log_prob, Gt in zip(episode_log_probs, returns):
        loss += -log_prob * Gt

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_rewards.clear()
    episode_log_probs.clear()


for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, log_prob = select_action(state)
        next_state, reward, done = env.step(action)

        episode_log_probs.append(log_prob)
        episode_rewards.append(reward)

        state = next_state
        total_reward += reward

    finish_episode()

    if episode % 50 == 0:
        print(f"Episode {episode} | Total Reward: {total_reward}")