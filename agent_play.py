import torch
from cartpole_env import cartpole_env
from nn import NeuralNetwork
import pygame

nn = NeuralNetwork()
nn.load_state_dict(torch.load('cartpole_policy.pth'))
nn.eval() # run network in inference mode

env = cartpole_env()

state = env.reset()
done = False
manual_override = False

while not done:
    env.render()
    
        # --- Handle keyboard input ---
    keys = pygame.key.get_pressed()

    manual_action = None

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        manual_action = 0   # push left
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        manual_action = 2   # push right
    elif keys[pygame.K_SPACE]:
        manual_override = not manual_override
        print("Manual Override:", manual_override)

    # --- Choose action ---
    if manual_override or manual_action is not None:
        action = manual_action  # human input
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs, value = nn(state_tensor)
        action = torch.argmax(probs).item()  # agent action

    # Apply action
    state, reward, done = env.step(action)

    print(f"Action: {action}")
    print(f"State: {state}")
    print(f"Reward: {reward}")

