import pygame

pygame.init()
screen = pygame.display.set_mode((1280,720))
clock = pygame.time.Clock()
running = True
dt = 0
player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
player_acc = 0
acceleration = 1000
cube_width = 100
cube_height = 50
cube = pygame.Rect(0,0, cube_width, cube_height)
max_speed = 300

while running:
    #check for stop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running=False
    

    screen.fill("purple")
    cube.x = player_pos.x
    cube.y = player_pos.y
    pygame.draw.rect(screen, "red", cube)
    pygame.draw.circle(screen, "black", (cube.x, cube.y+cube.height), 10)
    pygame.draw.circle(screen, "black", (cube.x+cube.width, cube.y+cube.height), 10)

    # move left right
    keys = pygame.key.get_pressed()

    if keys[pygame.K_a]:
        player_acc -= acceleration * dt
    if keys[pygame.K_d]:
        player_acc += acceleration * dt

    # Clamp speed
    player_acc = max(-max_speed, min(max_speed, player_acc))

    # Update position
    player_pos.x += player_acc * dt


    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()