from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = range(5, 10)
players = range(2, 4)
foods = range(1, 5)
coop = [True, False]
grid_observation = [True, False]

for s, p, f, c, grid_obs in product(sizes, players, foods, coop, grid_observation):
    for sight in range(1, s + 1):
        register(
            id="Foraging{5}{4}-{0}x{0}-{1}p-{2}f{3}-v2".format(s, p, f, "-coop" if c else "", "" if sight == s else f"-{sight}s", "-grid" if grid_obs else ""),
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": p,
                "max_player_level": 3,
                "field_size": (s, s),
                "max_food": f,
                "sight": sight,
                "max_episode_steps": 50,
                "mode": "MARL",
                "seed": 0,
                "force_coop": c,
                "grid_observation": grid_obs,
            },
        )

for s, p, f, c, grid_obs in product(sizes, players, foods, coop, grid_observation):
    for sight in range(1, s + 1):
        register(
            id="Foraging{5}{4}-{0}x{0}-{1}p-{2}f{3}-adhoc-v2".format(s, p, f, "-coop" if c else "", "" if sight == s else f"-{sight}s", "-grid" if grid_obs else ""),
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": p,
                "max_player_level": 3,
                "field_size": (s, s),
                "max_food": f,
                "sight": sight,
                "max_episode_steps": 50,
                "mode": "adhoc-eval",
                "seed": 0,
                "force_coop": c,
                "grid_observation": grid_obs,
            },
        )
