import numpy as np
import math

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

    def reset(self):
    # Random small initial values so it doesn’t start perfectly balanced
        self.x = np.random.uniform(-0.05, 0.05)
        self.x_dot = np.random.uniform(-0.05, 0.05)
        self.theta = np.random.uniform(-1, 1)
        self.theta_dot = np.random.uniform(-0.05, 0.05)
        self.time = 0
        self.done = False
        self.state = np.array([self.x, self.x_dot, self.theta, self.theta_dot], dtype=np.float32)
        return self.state   


    def step(self, action, step=0, dt=0.02):

        # calculate force from direction
        F = action * self.FORCE_MAG

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
        self.x_dot *= 0.99       # cart friction
        self.theta_dot *= 0.999  # pole air drag

        # update state
        state = np.array([self.x, self.x_dot, self.theta, self.theta_dot], dtype=np.float32)
        
        # compute reward and done
        done = False
        reward = 0
        if 3 < abs(self.x):
            done = True
            reward += -100
        else:
            reward += 1
        
        if step >= 500:
            done = True

        reward += np.cos(self.theta) * 2  # reward for keeping pole upright
        
        return  state, reward, done
        

    def render(self):
        # Lazy-init Pygame only when rendering
        if self.screen is None:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((1280, 720))
            self.clock = pygame.time.Clock()

        import pygame
        screen = self.screen
        clock = self.clock

        screen.fill("purple")

        # Coordinate conversions
        PIXELS_PER_METER = 200
        CART_Y = 500
        cart_width = 100
        cart_height = 50
        pole_length_px = int(1.0 * PIXELS_PER_METER)  # same LENGTH_POLE as physics

        cart_x_screen = int(screen.get_width() / 2 + self.x * PIXELS_PER_METER)
        cart_rect = pygame.Rect(cart_x_screen - cart_width//2, CART_Y, cart_width, cart_height)
        pygame.draw.rect(screen, (255, 0, 0), cart_rect)

        pivot = (cart_x_screen, CART_Y)
        pole_x = pivot[0] + pole_length_px * math.sin(self.theta)
        pole_y = pivot[1] - pole_length_px * math.cos(self.theta)

        pygame.draw.line(screen, (0, 0, 0), pivot, (pole_x, pole_y), 6)
        pygame.draw.circle(screen, (0, 0, 255), (int(pole_x), int(pole_y)), 10)

        pygame.display.flip()
        clock.tick(60)

 
    
        