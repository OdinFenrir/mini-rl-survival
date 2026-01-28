import pygame
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption('Test Pygame')
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
