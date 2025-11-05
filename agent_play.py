import torch
from cartpole_env import cartpole_env
from nn import NeuralNetwork
import time

nn = NeuralNetwork()
nn.load_state_dict(torch.load('cartpole_policy.pth'))
nn.eval() # run network in inference mode

env = cartpole_env()

state = env.reset()
done = False

while not done:
    env.render()
    state_tensor = torch.tensor(state, dtype=torch.float32)
    probs = nn(state_tensor)
    action = torch.argmax(probs).item()
    print(f"Action: {action}")
    state, reward, done = env.step(action)
    print(f"Reward: {reward}")

