import pygame
import imutils
import math
import random
import numpy as np


class Snake:

    DOWN = np.array([0, 1], dtype=np.byte)
    LEFT = np.array([-1, 0], dtype=np.byte)
    UP = np.array([0, -1], dtype=np.byte)
    RIGHT = np.array([1, 0], dtype=np.byte)

    dirs = [UP, LEFT, DOWN, RIGHT]

    def __init__(self, size):
        # self.seed = 0
        # random.seed(self.seed)
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype=np.byte)
        self.direction = self.RIGHT

        self.head = np.array([int(self.size/2), int(self.size/2)], dtype=np.byte)
        self.body = np.array([[int(self.size/2)-1, int(self.size/2)],
                              [int(self.size/2)-2, int(self.size/2)],
                              [int(self.size/2)-3, int(self.size/2)]])

        for piece in self.body:
            self.board[piece[1]][piece[0]] = 1
        self.board[self.head[1]][self.head[0]] = 2
        
        self._get_new_apple_pos()

        self.board[self.apple[1]][self.apple[0]] = 3

        self.reward = 0

        self.steps_taken = 0
        self.score = 0
        self.gameover = False


    def _get_new_apple_pos(self):
        while True:
            pos = [random.randint(0, self.size-1), random.randint(0, self.size)-1]
            if self.board[pos[1]][pos[0]] == 0:
                self.apple = pos
                break

    def _re_draw(self):
        self.board = np.zeros((self.size, self.size), dtype=np.byte)
        self.board[self.apple[1]][self.apple[0]] = 3
        for piece in self.body:
            self.board[piece[1]][piece[0]] = 1
        self.board[self.head[1]][self.head[0]] = 2
    
    def update(self):
        # self.seed += 1
        # random.seed(self.seed)
        if self.gameover: return self.score
        self.body = np.vstack([self.head, self.body])
        new_head = self.head + self.direction
        if new_head[0] >= self.size or new_head[0] < 0 or new_head[1] >= self.size or new_head[1] < 0:
            self.gameover = True
            self.body = self.body[:-1]
            self._re_draw()
            return self.score

        elif self.board[new_head[1]][new_head[0]] == 1:
            self.gameover = True
            self.body = self.body[:-1]
            self._re_draw()
            return self.score

        elif self.board[new_head[1]][new_head[0]] == 3:
            self.head = new_head
            self._get_new_apple_pos()
            self.score += 1
            self._re_draw()
            self.steps_taken += 1
            return True
        else:
            self.body = self.body[:-1]
            self.head = new_head
            self._re_draw()
            self.steps_taken += 1
            return False
    
    def set_dir(self, direction):
        if direction is self.UP:
            if self.direction is not self.DOWN:
                self.direction = self.UP
                return
        if direction is self.LEFT:
            if self.direction is not self.RIGHT:
                self.direction = self.LEFT
                return
        if direction is self.DOWN:
            if self.direction is not self.UP:
                self.direction = self.DOWN
                return
        if direction is self.RIGHT:
            if self.direction is not self.LEFT:
                self.direction = self.RIGHT
                return

    def get_steps(self):

        ms_r = self.size - self.head[0] - 1
        ms_l = self.size - ms_r - 1
        
        ms_d = self.size - self.head[1] - 1
        ms_u = self.size - ms_r - 1

        x = self.head[0]
        y = self.head[1]
        s_u = [self.board[y-i-1][x] for i in range(ms_u)]
        s_l = [self.board[y][x-i-1] for i in range(ms_l)]
        s_d = [self.board[y+i+1][x] for i in range(ms_d)]
        s_r = [self.board[y][x+i+1] for i in range(ms_r)]

        s_u += [1] * 100
        s_l += [1] * 100
        s_d += [1] * 100
        s_r += [1] * 100

        # if 3 in s_u and s_u.index(3) < s_u.index(1):
        #     print('apple up')

        # if 3 in s_l and s_l.index(3) < s_l.index(1):
        #     print('apple left')

        # if 3 in s_d and s_d.index(3) < s_d.index(1):
        #     print('apple down')

        # if 3 in s_r and s_r.index(3) < s_r.index(1):
        #     print('apple right')
        
        info = np.array([int(3 in s_u and s_u.index(3) < s_u.index(1)), int(3 in s_l and s_l.index(3) < s_l.index(1)),
                         int(3 in s_d and s_d.index(3) < s_d.index(1)), int(3 in s_r and s_r.index(3) < s_r.index(1)),
                         int(s_u.index(1)==0), int(s_l.index(1)==0), int(s_d.index(1)==0), int(s_r.index(1)==0)])

        # print(int(s_u.index(1)==0))
        # print(int(s_l.index(1)==0))
        # print(int(s_d.index(1)==0))
        # print(int(s_r.index(1)==0))


        return info
        
class Vis:

    def __init__(self, snakes, size=(800, 800)):
        self.size = size
        self.screen = pygame.display.set_mode(size)
        self.running = True
        self.snakes = snakes

        self.surfaces = []

    def create_surf(self, snake):
        size = self.cubeside - 2
        board = np.array(snake.board, copy=True, order='K')
        image = np.zeros((snake.size, snake.size, 3), dtype=np.uint8)
        for y in range(len(board)):
            for x in range(len(board[0])):
                if board[y][x] == 0:
                    image[y][x] = (10, 10, 10)
                if board[y][x] == 1:
                    image[y][x] = (82, 255, 77)
                if board[y][x] == 2:
                    image[y][x] = (0, 161, 0)
                if board[y][x] == 3:
                    image[y][x] = (255, 13, 13)
        
        image = imutils.resize(np.flip(np.rot90(image, k=3), axis=1), size)
        surf = pygame.surfarray.make_surface(image)
        self.surfaces.append(surf)

    def get_snake_distribution(self):
        w = self.size[0]
        h = self.size[1]
        c = max(1, len(self.snakes))
        pw = math.ceil(math.sqrt(c*w/h))
        sw = 0
        sh = 0

        if math.floor(pw*h/w)*pw < c:
            sw = h/math.ceil(pw*h/w)
        else:
            sw = w/pw

        ph = math.ceil(math.sqrt(c*h/w))

        if math.floor(pw*w/h)*ph < c:
            sh = w/math.ceil(w*ph/h)
        else:
            sh = h/ph

        self.cubeside = int(max(sh, sw))

    def update(self):
        self.screen.fill((60, 60, 60))

        self.surfaces = []

        self.get_snake_distribution()

        for snake in self.snakes:
            self.create_surf(snake)

        y = 0
        x = 0
        n = 0
        n_on_row = int(self.size[0] / self.cubeside)
        for surf in self.surfaces:
            self.screen.blit(surf, (x + 1, y ))
            n += 1
            x += self.cubeside
            if n == n_on_row:
                y += self.cubeside
                x = 0
                n = 0

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                quit()
