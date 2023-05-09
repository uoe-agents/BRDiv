import gym
import time

level = 'open_room_salad2'
seed = 1
record = False
max_steps = 1000
recipe = "TomatoSalad"

env = gym.envs.make("gym_cooking:cookingEnv-v1", level=level, record=record, max_steps=max_steps, recipe=recipe,
                    obs_spaces=["feature_vector"])

obs = env.reset()

action_space = env.action_space

done = False
print(env)
while not done:
    time.sleep(0.05)
    action = action_space.sample()
    observation, reward, done, info = env.step(action)

env.render()
print('done')
