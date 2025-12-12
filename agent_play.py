import torch
from cartpole_env import cartpole_env
from nn import NeuralNetwork
import pygame

nn = NeuralNetwork()
try:
    nn.load_state_dict(torch.load('cartpole_policy.pth'))
    print("Loaded trained model.")
except FileNotFoundError:
    print("No trained model found, using untrained network.")
nn.eval() # run network in inference mode

env = cartpole_env()

state = env.reset()
done = False
space_pressed = False
manual_play = False
while True:
    env.render(manual_play)
    
        # --- Handle keyboard input ---
    keys = pygame.key.get_pressed()

    manual_action = 1

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        manual_action = 0   # push left
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        manual_action = 2   # push right
    elif keys[pygame.K_r]:
        env.reset()  # no force

    if keys[pygame.K_SPACE]:
        if space_pressed == False:
            space_pressed = True
            manual_play = not manual_play
    else:
        space_pressed = False
    

    # --- Choose action ---
    if manual_play:
        print("Manual Override:", manual_play)
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

    if done:
        env.reset()
        

