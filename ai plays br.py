from numpy import angle
from deep_learning import Layer, Agent, Population
from games.battle_royale import Game
from math import pi


size = (1200, 900)
population_size = 50

max_updates = 500

rate = 1
scale = 0.1
pool_size = 6
method = Population.TOP_X
include_parents = True
mutated_layers = Agent.RANDOM

l1 = Layer(3, 8, activation=Layer.RELU)
l2 = Layer(8, 8, activation=Layer.RELU)
l3 = Layer(8, 4, activation=Layer.SIGMOID)

agent = Agent([l1, l2, l3])

population = Population(agent, population_size)

# population = Population(file_name='br_players.npy')
population.evolution_settings(include_parents, rate, scale, pool_size, method, population_size, mutated_layers=mutated_layers)

game = Game(size)

best_fitness = 0
most_kills = 0


gen = 0 
while True:
    gen += 1
    game.restart(population_size)
    players_alive = population_size
    updates = 0
    while players_alive > 1:
        updates += 1
        players_alive = 0
        for agent, player in zip(population.agents, game.players):
            if not player.is_alive: continue
            else: players_alive += 1
            inputs = [player.pos[0]/size[0], player.pos[1]/size[1], player.angle_to_closest/(pi*2)]
            agent.push_forward(inputs)
            i, j = agent.outputs[0]*2-1, agent.outputs[1]*2-1
            angle = agent.outputs[2]*2*pi
            shoot = agent.outputs[3] > 0.5
            player.update((i, j), angle, shoot)

            fitness = player.n_updates + player.kills*100
            agent.fitness = fitness
            if agent.fitness > best_fitness:
                best_fitness = agent.fitness
            if player.kills > most_kills:
                most_kills = player.kills
        if updates > max_updates:
            break
        game.update()
    print(f'Gen: {gen} | Best fitness: {best_fitness} | Most kills: {most_kills}')
    population.save(f'br_players.npy', rewrite=True)
    population.evolve()


