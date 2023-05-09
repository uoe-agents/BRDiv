from itertools import product
from gym.envs.registration import register

level_variations = ["", "_tiny", "_med", "_blocked"]
steps_variations = [250, 500, 1000]
adhoc_variations = [False, True]

def variation_string(level=None, steps=None, adhoc=False):
    if not level:
        level_string = ""
    else:
        level_string = "-" + level.strip("_")

    if steps is None:
        steps_string = ""
    else:
        steps_string = "-" + str(steps)

    if not adhoc:
        adhoc_string = ""
    else:
        adhoc_string = "-adhoc"

    return f"cookingZooEnv{level_string}{steps_string}{adhoc_string}-v0"

for level, steps, adhoc in product(level_variations, steps_variations, adhoc_variations):
    register(
        id=variation_string(level=level, steps=steps, adhoc=adhoc),
        entry_point="gym_cooking.environment:CookingZooEnvironment",
        kwargs={
            "level": f"tomato_carrot_split{level}",
            "num_agents": 2,
            "record": False,
            "max_steps": 250,
            "recipes": ["TomatoCarrotMash", "TomatoCarrotMash"],
            "obs_spaces": ["feature_vector_nc", "feature_vector_nc"],
            "allowed_objects": None,
            "action_scheme": "full_action_scheme",
            "ghost_agents": 0,
            "completion_reward_frac": 0.0,
            "time_penalty": 0.0,
            "mode": "adhoc-eval" if adhoc else "MARL",
        }
    )
