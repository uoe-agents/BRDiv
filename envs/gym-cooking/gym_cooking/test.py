from gym_cooking.environment import cooking_zoo
import pettingzoo.test as pzt

n_agents = 2
num_humans = 1
max_steps = 100
render = False

level = 'split_room'
seed = 1
record = False
max_num_timesteps = 1000
recipes = ["TomatoSalad", 'TomatoSalad']

parallel_env = cooking_zoo.parallel_env(
    level=level,
    num_agents=n_agents,
    record=record,
    max_steps=max_num_timesteps,
    recipes=recipes,
    obs_spaces=["simple"]
    )

#import pdb; pdb.set_trace()
pzt.parallel_api_test(parallel_env, num_cycles=100)
