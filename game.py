import pygame
import math

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

# --- Physics constants ---
g = 9.81
M = 1.0    # cart mass
m = 0.1    # pole mass
L = 1.0    # half the pole length (meters)
force_mag = 50.0
max_force = 200.0

# --- Simulation scale ---
PIXELS_PER_METER = 200
cart_y = 500  # y position of cart on screen

# --- State variables ---
x = 0.0          # cart position (m)
x_dot = 0.0      # cart velocity (m/s)
theta = 0.1      # pole angle (radians)
theta_dot = 0.0  # pole angular velocity (rad/s)

# --- Drawing constants ---
cart_width = 100
cart_height = 50
pole_length_px = int(L * 2 * PIXELS_PER_METER)  # full pole length in pixels

while running:
    dt = clock.tick(60) / 1000  # seconds per frame

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Input: cart force ---
    keys = pygame.key.get_pressed()
    F = 0.0
    if keys[pygame.K_a]:
        F = -force_mag
    elif keys[pygame.K_d]:
        F = force_mag

    # --- Physics (realistic equations) ---
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    # theta acceleration
    num = g * sin_theta + cos_theta * (
        -F - m * L * (theta_dot**2) * sin_theta
    ) / (M + m)
    den = L * (4.0 / 3.0 - (m * cos_theta**2) / (M + m))
    theta_ddot = num / den

    # cart acceleration
    x_ddot = (F + m * L * (theta_dot**2 * sin_theta - theta_ddot * cos_theta)) / (M + m)

    # integrate
    x_dot += x_ddot * dt
    x += x_dot * dt
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

    # --- Drawing ---
    screen.fill((160, 100, 200))
    cart_x_screen = int(screen.get_width() / 2 + x * PIXELS_PER_METER)
    cart_rect = pygame.Rect(cart_x_screen - cart_width//2, cart_y, cart_width, cart_height)
    pygame.draw.rect(screen, (255, 0, 0), cart_rect)

    # Pole pivot point
    pivot = (cart_x_screen, cart_y)
    # Pole tip position
    pole_x = pivot[0] + pole_length_px * math.sin(theta)
    pole_y = pivot[1] - pole_length_px * math.cos(theta)
    pygame.draw.line(screen, (0, 0, 0), pivot, (pole_x, pole_y), 6)
    pygame.draw.circle(screen, (0, 0, 255), (int(pole_x), int(pole_y)), 10)

    pygame.display.flip()

pygame.display.quit()
