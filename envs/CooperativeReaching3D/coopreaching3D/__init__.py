from gym.envs.registration import registry, register, make, spec
from itertools import product

world_length = range(5, 20)
max_timesteps = range(15, 100)

for s, f in product(world_length, max_timesteps):
    register(
        id="MARL-CooperativeReaching3D-{0}-{1}-v0".format(s, f),
        entry_point="coopreaching3D.coopreaching3D:MARLCooperativeReachingEnv",
        kwargs={
            "world_width": s,
            "world_depth": s,
            "world_height": s,
            "max_episode_steps": f,
            "seed": 1234,
            "mode": "MARL"
        }
    )

for s, f in product(world_length, max_timesteps):
    register(
        id="MARL-CooperativeReaching3D-{0}-{1}-adhoc-v0".format(s, f),
        entry_point="coopreaching3D.coopreaching3D:MARLCooperativeReachingEnv",
        kwargs={
            "world_width": s,
            "world_depth": s,
            "world_height": s,
            "max_episode_steps": f,
            "seed": 1234,
            "mode": "adhoc-eval"
        }
    )
