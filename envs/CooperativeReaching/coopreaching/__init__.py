from gym.envs.registration import registry, register, make, spec
from itertools import product

corridor_lengths = range(5, 10)
max_timesteps = range(15, 60)
teammate_ids = range(1,12)

for s, f in product(corridor_lengths, max_timesteps):
    register(
        id="MARL-CooperativeReaching-{0}-{1}-v0".format(s, f),
        entry_point="coopreaching.coopreaching:MARLCooperativeReachingEnv",
        kwargs={
            "world_length": s,
            "world_height": s,
            "max_episode_steps": f,
            "seed": 1234,
            "mode": "MARL"
        }
    )

corridor_lengths = range(5, 6)
max_timesteps = range(50, 51)
teammate_ids = range(1,12)

for s, f, t_id in product(corridor_lengths, max_timesteps, teammate_ids):
    register(
        id="MARL-CooperativeReaching-{0}-{1}-adhoc{2}-v0".format(s, f, t_id),
        entry_point="coopreaching.coopreaching:MARLCooperativeReachingEnv",
        kwargs={
            "world_length": s,
            "world_height": s,
            "max_episode_steps": f,
            "seed": 1234,
            "mode": "adhoc-eval",
            "teammate_id": t_id
        }
    )
    print("MARL-CooperativeReaching-{0}-{1}-adhoc-v0{2}".format(s, f, t_id))
