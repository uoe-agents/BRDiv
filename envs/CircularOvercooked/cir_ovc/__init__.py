from gym.envs.registration import registry, register, make, spec
from itertools import product

corridor_lengths = range(8, 11)
max_timesteps = [150, 200, 250, 300]

for s, f in product(corridor_lengths, max_timesteps):
    register(
        id="MARL-Circular-Overcooked-{0}-{1}-v0".format(s, f),
        entry_point="cir_ovc.cir_ovc:MARLCircularOvercookedEnv",
        kwargs={
            "corridor_length": s,
            "max_episode_steps": f,
            "seed": 1234,
            "mode": "MARL"
        }
    )

corridor_lengths = range(10, 11)
max_timesteps = [150, 200, 250, 300]
teammate_ids = range(1,13)

for s, f, t_id in product(corridor_lengths, max_timesteps, teammate_ids):
    register(
        id="MARL-Circular-Overcooked-{0}-{1}-adhoc{2}-v0".format(s, f, t_id),
        entry_point="cir_ovc.cir_ovc:MARLCircularOvercookedEnv",
        kwargs={
            "corridor_length": s,
            "max_episode_steps": f,
            "seed": 1234,
            "mode": "adhoc-eval",
            "teammate_id": t_id
        }
    )