import pygame
import math
import random
import numpy

pygame.init()
screen = pygame.display.set_mode((1280,720))
clock = pygame.time.Clock()
running = True

# physics constants
G = 9.81
MASS_CART = 2.0
MASS_POLE = 0.05
LENGTH_POLE = 1.0 # half the pole length
FORCE_MAG = 20
MAX_FORCE = 300

# scale of simulation
PIXELS_PER_METER = 200
CART_Y = 500

# state variables

x = 0.0 # cart postition
x_dot = 0.0 # initial velocity(m/s)
theta = random.uniform(-0.5,0.5) # pole angle in radians
theta_dot = 0 # angular velocity(radians/sec)

# Drawing sontants
cart_width = 100
cart_height = 50
pole_length_px = int(LENGTH_POLE * PIXELS_PER_METER)  # full pole length in pixels



while running:

    dt = clock.tick(60) / 1000

    #check for stop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running=False
    
    # get movement input left right
    keys = pygame.key.get_pressed()
    F = 0.0
    if keys[pygame.K_a]:
        F = -FORCE_MAG
    elif keys[pygame.K_d]:
        F = FORCE_MAG
    elif keys[pygame.K_r]:
        x, x_dot, theta, theta_dot = 0, 0, 0.1, 0

    # --- Prevent pushing into walls ---
    max_x = 3.0
    if (x <= -max_x and F < 0) or (x >= max_x and F > 0):
        F = 0

    # Physics equations
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    total_mass = MASS_CART + MASS_POLE
    temp = (F + MASS_POLE * LENGTH_POLE * theta_dot**2 * sin_theta) / total_mass
    theta_ddot = (G * sin_theta - cos_theta * temp) / (
        LENGTH_POLE * (4.0 / 3.0 - (MASS_POLE * cos_theta**2) / total_mass)
    )
    # x acceleration
    x_ddot = temp - MASS_POLE * LENGTH_POLE * theta_ddot * cos_theta / total_mass

    # integrate
    x_dot += x_ddot * dt # x_velocity
    x += x_dot * dt
    theta_dot += theta_ddot * dt # theta velocity
    theta += theta_dot * dt
    

    # --- Damping (friction / air resistance) ---
    x_dot *= 0.99       # cart friction
    theta_dot *= 0.999  # pole air drag


    # --- Clamp cart position and stop at edges ---
    max_x = 3.0  # 3 meters from center (~600 px if 200 px/m)
    if x > max_x:
        x = max_x
        x_dot = 0
        x_ddot = 0
    elif x < -max_x:
        x = -max_x
        x_dot = 0
        x_ddot = 0

    


    # set Background
    screen.fill("purple")

    cart_x_screen = int(screen.get_width() / 2 + x * PIXELS_PER_METER)
    cart_rect = pygame.Rect(cart_x_screen - cart_width//2, CART_Y, cart_width, cart_height)
    pygame.draw.rect(screen, (255, 0, 0), cart_rect)
   

    # Pole pivot point
    pivot = (cart_x_screen, CART_Y)
    # Pole tip position
    pole_x = pivot[0] + pole_length_px * math.sin(theta)
    pole_y = pivot[1] - pole_length_px * math.cos(theta)
    pygame.draw.line(screen, (0, 0, 0), pivot, (pole_x, pole_y), 6)
    pygame.draw.circle(screen, (0, 0, 255), (int(pole_x), int(pole_y)), 10)

    pygame.display.flip()

pygame.quit()