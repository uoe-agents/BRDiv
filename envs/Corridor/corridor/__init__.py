from gym.envs.registration import registry, register, make, spec
from itertools import product

corridor_lengths = range(5, 10)
max_timesteps = range(15, 21)

for s, f in product(corridor_lengths, max_timesteps):
    register(
        id="MARL-Corridor-{0}-{1}-v0".format(s, f),
        entry_point="corridor.corridor:MARLCorridorEnv",
        kwargs={
            "corridor_length": s,
            "max_episode_steps": f,
            "seed": 1234
        }
    )