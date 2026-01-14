import numpy as np
import math
import pygame
import os

class cartpole_env:
    def __init__(self):
        # physics constants
        self.G = 9.81
        self.MASS_CART = 2.0
        self.MASS_POLE = 0.05
        self.LENGTH_POLE = 1.0 # half the pole length
        self.FORCE_MAG = 20
        self.MAX_FORCE = 300

        # state variables
        self.x = 0.0 # cart postition
        self.x_dot = 0.0 # initial velocity(m/s)
        self.theta = 0.5 # pole angle in radians
        self.theta_dot = 0 # angular velocity(radians/sec)

        # state
        self.state = None
        self.done = False
        self.time = 0

        # reward
        self.reward = 1
        self.penalty = -1
        self.fallen = 0
        self.upright_steps = 0

        # Pygame variables
        self.screen = None
        self.clock = None
        self.bg = None
        self.font = None
        self.bg_width = 0

    def reset(self, r=1):
    # Random small initial values so it doesn’t start perfectly balanced
        self.x = np.random.uniform(-((0.1*r)%3), (0.1*r)%3)
        self.x_dot = np.random.uniform(-0.1, 0.1)
        self.theta = np.random.uniform(-0.1*r, 0.1*r)
        self.theta_dot = np.random.uniform(-0.1, 0.1)
        self.time = 0
        self.done = False
        self.fallen = 0
        self.upright_steps = 0
        self.state = np.array([self.x, self.x_dot, np.sin(self.theta), np.cos(self.theta), self.theta_dot], dtype=np.float32)
        return self.state   


    def step(self, action, step=0, dt=0.02):

        # calculate force from direction
        if action == 0:
            F = -self.FORCE_MAG
        elif action == 2:
            F = self.FORCE_MAG
        else:
            F = 0

        # Physics equations
        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)

        total_mass = self.MASS_CART + self.MASS_POLE
        temp = (F + self.MASS_POLE * self.LENGTH_POLE * self.theta_dot**2 * sin_theta) / total_mass
        theta_ddot = (self.G * sin_theta - cos_theta * temp) / (
            self.LENGTH_POLE * (4.0 / 3.0 - (self.MASS_POLE * cos_theta**2) / total_mass)
        )
        # acceleration
        x_ddot = temp - self.MASS_POLE * self.LENGTH_POLE * theta_ddot * cos_theta / total_mass

        # integrate
        self.x_dot += x_ddot * dt # x_velocity
        self.x += self.x_dot * dt
        self.theta_dot += theta_ddot * dt # self.theta velocity
        self.theta += self.theta_dot * dt
        

        # --- Damping (friction / air resistance) ---
        damping_factor = 0.4
        self.x_dot *= (1 - damping_factor * dt)     # cart friction
        self.theta_dot *= (1 - damping_factor * dt)  # pole air drag

        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)

        # update state
        state = np.array([self.x, self.x_dot, sin_theta, cos_theta, self.theta_dot], dtype=np.float32)
        
        # compute reward and done
        reward, done = self.compute_reward(step)
        
        return  state, reward, done
        
    def compute_reward(self, step):
        done = False
        reward = 0
        cos_t = np.cos(self.theta)
    
        # Reward upright pole (cos(theta) is +1 when vertical)
        reward += cos_t   # +1 good, 0 neutral, -1 bad
        
        # Soft penalty for high velocities
        reward -= 0.01 * abs(self.theta_dot)
        reward -= 0.01 * abs(self.x_dot)
        reward -= 0.001 * abs(self.x)  # small penalty for being far from center
        # Episode success if survived long
        if step >= 500:
            done = True
        
        return reward, done

    def render(self, manual_play=False):
        # Lazy-init Pygame only when rendering
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1440, 720))
            self.clock = pygame.time.Clock()
             # Load and scale background once
            self.bg = pygame.image.load(os.path.join('images', 'bg.png')).convert()
            self.bg = pygame.transform.scale(self.bg, (1440, 720))
            self.bg_width = self.bg.get_width()
            self.font = pygame.font.SysFont(None, 36)


        screen = self.screen
        clock = self.clock

        for event in pygame.event.get():   # <-- THIS prevents freezing
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

        #screen.fill("purple")
        
        
        
        # Coordinate conversions
        PIXELS_PER_METER = 200
        CART_Y = 500
        cart_width = 100
        cart_height = 50
        pole_length_px = int(1.0 * PIXELS_PER_METER)  # same LENGTH_POLE as physics

        # Track background scrolling
        bgX = 0 - (self.x * PIXELS_PER_METER) % self.bg.get_width()
        screen.blit(self.bg, (bgX, 0))
        screen.blit(self.bg, (bgX + self.bg_width, 0))

        # cart_x_screen = int(screen.get_width() / 2 + self.x * PIXELS_PER_METER)
        cart_x_screen = int(screen.get_width() // 2)
        cart_rect = pygame.Rect(cart_x_screen - cart_width//2, CART_Y, cart_width, cart_height)
        pygame.draw.rect(screen, (255, 0, 0), cart_rect)

        


        # --- Draw manual/agent indicator ---
        label = "Manual" if manual_play else "Agent"
        text = self.font.render(label, True, (255, 255, 255))
        screen.blit(text, (20, 20))

        color = (0, 255, 0) if manual_play else (255, 0, 0)
        pygame.draw.rect(screen, color, (120, 20, 20, 20))

        pivot = (cart_x_screen, CART_Y)
        pole_x = pivot[0] + pole_length_px * math.sin(self.theta)
        pole_y = pivot[1] - pole_length_px * math.cos(self.theta)

        pygame.draw.line(screen, (0, 0, 0), pivot, (pole_x, pole_y), 6)
        pygame.draw.circle(screen, (0, 0, 255), (int(pole_x), int(pole_y)), 10)

        pygame.display.flip()
        clock.tick(60)

 
    
        