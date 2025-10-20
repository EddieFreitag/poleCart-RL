import pygame

pygame.init()
screen = pygame.display.set_mode((1280,720))
clock = pygame.time.Clock()
running = True
dt = 0
player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
cube = pygame.Rect(0,0, 100, 50)
while running:
    #check for stop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running=False
    

    screen.fill("purple")
    cube.x = player_pos.x
    cube.y = player_pos.y
    pygame.draw.rect(screen, "red", cube)

    # move left right
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        player_pos.x -= 300 * dt
    if keys[pygame.K_d]:
        player_pos.x += 300 * dt


    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()