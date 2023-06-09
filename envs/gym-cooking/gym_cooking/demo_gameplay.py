from gym_cooking.environment.game.game import Game

from gym_cooking.environment import cooking_zoo


n_agents = 1
num_humans = 1
max_steps = 100
render = False

level = 'split_room'
seed = 1
record = False
max_num_timesteps = 1000
recipes = ["TomatoSalad"]

parallel_env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record,
                                        max_steps=max_num_timesteps, recipes=recipes, action_scheme="scheme3")

game = Game(parallel_env, num_humans, [], max_num_timesteps)
store = game.on_execute()

print("done")

