import pygame as pg
import random
import math
import keyboard


class Projectile:

    SPEED = 30

    def __init__(self, pos, velocity, owner):
        self.pos = pos
        self.velocity = [velocity[0] * self.SPEED, velocity[1] * self.SPEED]
        self.owner = owner

    def update(self):
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
    
class Game:

    class Player:

        def __init__(self, game, spawn, Id, color_1=None, color_2=None):
            if color_1 is None: color_1 = (46, 45, 31)
            if color_2 is None: color_2 = (55, 56, 35)

            self.game = game

            self.color_1 = color_1
            self.color_2 = color_2
            self.pos = list(spawn)
            self.radius = 10
            self.length = 15
            self.width = 4
            self.angle = random.random()*2*math.pi
            self.velocity = [0, 0]
            self.id = Id

            self.max_vel = 4

            self.is_alive = True

            self.n_updates = 0
            self.last_shot = 0
            self.shot_cooldown = 50

            self.angle_to_closest = 0

            self.kills = 0

        def update(self, velocity, angle, shoot):
            if not self.is_alive: return
            self.n_updates += 1
            self.angle = angle
            self.velocity = [velocity[0]*self.max_vel, velocity[1]*self.max_vel]
            self.pos = [self.pos[0]+self.velocity[0], self.pos[1]+self.velocity[1]]

            if self.pos[0] > self.game.size[0]: self.is_alive = False
            if self.pos[0] < 0: self.is_alive = False
            if self.pos[1] > self.game.size[1]: self.is_alive = False
            if self.pos[1] < 0: self.is_alive = False



            if shoot:
                if self.n_updates > self.last_shot + self.shot_cooldown:
                    self.game.projectiles.append(Projectile(self.pos[:], (math.cos(self.angle), math.sin(self.angle)), self.id))
                    self.last_shot = self.n_updates
        
        def find_closest(self):
            x, y = self.pos
            closest_dist = 9999999
            xc, yc = 0, 0
            for player in self.game.players:
                if player is self: continue
                xe, ye = player.pos
                dist = math.sqrt(pow(x-xe, 2)+pow(y-ye, 2))
                if dist < closest_dist:
                    closest_dist = dist
                    xc, yc = xe, ye
            self.angle_to_closest = math.atan2(y-yc, x-xc)

            

    COLOR_BACKGROUND = (97, 76, 54)

    def __init__(self, size=(800, 800)):
        self.size = size
        self.screen = pg.display.set_mode(self.size)
        self.clock = pg.time.Clock()
        self.fps = 60

        self.spawn_ofset = 20

        self.projectiles = []

    def check_hit(self, player):
        x, y = player.pos
        new_projetiles = []
        for projectile in self.projectiles:
            xp, yp = projectile.pos
            dist = math.sqrt(pow(x-xp, 2)+pow(y-yp, 2))
            if dist < player.radius and projectile.owner != player.id:
                self.players[projectile.owner].kills += 1
                player.is_alive = False
                continue
            if (self.size[0] > xp > 0) and (self.size[1] > yp > 0):
                new_projetiles.append(projectile)
        self.projectiles = new_projetiles
            

    def restart(self, n_players):
        self.players = []
        self.players_alive = n_players

        for idx in range(n_players):
            x = random.randint(self.spawn_ofset, self.size[0]-self.spawn_ofset)
            y = random.randint(self.spawn_ofset, self.size[1]-self.spawn_ofset)
            self.players.append(self.Player(self, (x, y), idx))

    def update_player(self, player):
        if player.is_alive:
            pg.draw.circle(self.screen, player.color_1, player.pos, player.radius)
            pg.draw.line(self.screen, player.color_2, player.pos,
            (player.pos[0]+math.cos(player.angle) * player.length, player.pos[1]+math.sin(player.angle) * player.length), player.width)
            self.check_hit(player)

    def update(self):

        self.screen.fill(self.COLOR_BACKGROUND)

        n_players_alive = 0
        for player in self.players:
            n_players_alive += 1
            self.update_player(player)
            player.find_closest()
        self.players_alive = n_players_alive
        
        for projectile in self.projectiles:
            projectile.update()
            pg.draw.circle(self.screen, (255, 0, 0), projectile.pos, 4)





        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
                pg.quit()
                quit()

        pg.display.flip()
        self.clock.tick(self.fps)


if __name__ == '__main__':
    game = Game()
    game.restart(20)
    while True:
        game.update()

        i, j = 0, 0
        if keyboard.is_pressed('s'):
            i = 1
        if keyboard.is_pressed('w'):
            i = -1
        if keyboard.is_pressed('a'):
            j = -1
        if keyboard.is_pressed('d'):
            j = 1
        angle = 0
        if keyboard.is_pressed('j'): angle = -0.07
        if keyboard.is_pressed('l'): angle = 0.07
        shoot = keyboard.is_pressed('i')
        
        game.players[0].update((j, i), angle, shoot)