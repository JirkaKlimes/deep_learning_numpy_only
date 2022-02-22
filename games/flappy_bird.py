import time
import pygame
import random
pygame.init()


class Game:

    class Bird:
        
        GRAVITY = 2098

        def __init__(self, size, color=(227, 190, 7)):
            self.size = size
            self.color = color
            self.color_outline = (int(max(0, color[0]-50)), int(max(0, color[1]-50)),int(max(0, color[2]-50)))
            self.y = int(size[1]/2)
            self.speed = 0
            self.last_update_time = None

            self.jump_power = -450

            self.is_alive = True
            self.increments = 0
        
        def update(self, jump=False):
            if not self.is_alive: return
            self.increments += 1
            if self.last_update_time is None:
                self.last_update_time = time.time()
            delta = time.time() - self.last_update_time
            if jump: self.speed = self.jump_power
            self.speed += self.GRAVITY * delta
            self.y += self.speed * delta
            if self.y > self.size[1]:
                self.y = self.size[1]
            if self.y < 0:
                self.y = 0
            self.last_update_time = time.time()
            

    COLOR_BACKGROUND = (7, 176, 227)
    COLOR_PIPES = (38, 156, 36)

    SPEED = 300

    def __init__(self, size=(800, 400)):
        self.size = size
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()

        # self.hole_size = 80
        self.hole_size = 40
        # self.pipe_spacing = int(size[0]/2)
        self.pipe_spacing = 700
        # self.pipe_half_widght = int(self.size[0]/20)
        self.pipe_half_widght = 10
        self.pipe_ofset = int(self.size[1]/8)

        self.all_pipes = [[self.size[0], random.randint(0+self.hole_size, self.size[1]-self.hole_size)]]

        self.birds_x = int(self.size[0]/5)
        self.bird_radius = self.size[1]/25

        self.last_update_time = None

        self.speed = self.SPEED
        self.fps = 60

        self.birds = []

        self.background_surface = pygame.Surface(size)

        self.updates = 0

    def restart(self, birds=1):
        self.speed = self.SPEED
        self.all_pipes = [[self.size[0], random.randint(0+self.hole_size, self.size[1]-self.hole_size)]]
        self.last_update_time = None
        self.birds = []
        for _ in range(birds):
            self.birds.append(self.Bird(self.size))
    
    def create_pipes(self, pos):
        x, y = pos
        top_y = y-self.hole_size
        bottom_y = y+self.hole_size

        top_pipe = [(x-self.pipe_half_widght, top_y),
                    (x+self.pipe_half_widght, top_y),
                    (x+self.pipe_half_widght, 0),
                    (x-self.pipe_half_widght, 0)]
        
        bottom_pipe = [(x-self.pipe_half_widght, bottom_y),
                    (x+self.pipe_half_widght, bottom_y),
                    (x+self.pipe_half_widght, self.size[1]),
                    (x-self.pipe_half_widght, self.size[1])]

        return top_pipe, bottom_pipe


    def update_pipes(self):
        
        if self.all_pipes[-1][0] < self.size[0] - self.pipe_spacing:
            new_pipes = [self.size[0], random.randint(0+self.hole_size+self.pipe_ofset, self.size[1]-self.hole_size-self.pipe_ofset)]
            self.all_pipes.append(new_pipes)

    def check_colisions(self):
        for pipe in sorted(self.all_pipes):
            if pipe[0]+self.bird_radius+self.pipe_half_widght > self.birds_x:
                break
        self.closset_pipe = pipe
        px, py = pipe
        for bird in self.birds:
            if px + self.pipe_half_widght > self.birds_x + self.bird_radius+5 > px - self.pipe_half_widght:
                if not py + self.hole_size > bird.y + self.bird_radius+5 > py - self.hole_size:
                    bird.is_alive = False
            if bird.y < self.bird_radius or bird.y > self.size[1] - self.bird_radius:
                bird.is_alive = False


    def update(self):
        self.updates += 1
        if self.updates > 50:
            self.speed += 10
            self.updates = 0
        if self.last_update_time is None:
            self.last_update_time = time.time()
        delta = time.time() - self.last_update_time

        self.update_pipes()

        self.check_colisions()

        self.screen.fill(self.COLOR_BACKGROUND)
        self.background_surface.fill(self.COLOR_BACKGROUND)

        for pipe in self.all_pipes:
            pipe[0] -= self.speed * delta
            top_pipe, bottom_pipe = self.create_pipes(pipe)
            pygame.draw.polygon(self.background_surface, self.COLOR_PIPES, top_pipe)
            pygame.draw.polygon(self.background_surface, self.COLOR_PIPES, bottom_pipe)

        for bird in self.birds:
            if bird.is_alive:
                pygame.draw.circle(self.background_surface, bird.color, (self.birds_x, bird.y), self.bird_radius)
                pygame.draw.circle(self.background_surface, bird.color_outline, (self.birds_x, bird.y), self.bird_radius, 2)

        self.screen.blit(self.background_surface, (0,0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                quit()

        pygame.display.flip()
        self.last_update_time = time.time()
        self.clock.tick(self.fps)

if __name__ == '__main__':
    game = Game()
    b = game.Bird(game.size)
    game.birds.append(b)
    while True:
        game.update()