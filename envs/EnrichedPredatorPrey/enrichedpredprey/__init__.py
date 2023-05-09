from gym.envs.registration import registry, register, make, spec
from itertools import product

corridor_lengths = range(5, 10)
max_timesteps = range(15, 120)

for s, f in product(corridor_lengths, max_timesteps):
    register(
        id="MARL-EnrichedPredPrey-{0}-{1}-v0".format(s, f),
        entry_point="enrichedpredprey.enrichedpredprey:MARLEnrichedPredPreyEnv",
        kwargs={
            "world_length": s,
            "world_height": s,
            "max_episode_steps": f,
            "seed": 1234,
            "max_player_hp": 12,
            "max_prey_hp": 20,
            "player_normal_attack_accuracy": 0.85,
            "prey_normal_attack_accuracy": 0.9,
            "player_normal_attack_damage": 2,
            "prey_normal_attack_damage": 3,
            "player_normal_attack_range": 2,
            "prey_normal_attack_range": 1,
            "longsword_attack_accuracy": 0.95,
            "longsword_attack_damage": 4,
            "longsword_attack_range": 2,
            "greatsword_attack_accuracy": 0.95,
            "greatsword_attack_damage": 6,
            "greatsword_attack_range": 2,
            "greatsword_movement_penalty": 0.35,
            "bow_attack_accuracies": [1.0, 0.9, 0.85, 0.65, 0.40, 0.20, 0.10, 0.05, 0.0, 0.0, 0.0],
            "bow_attack_damage": 2,
            "bow_attack_range": 10,
            'healing_staff_cooldown_period': 8,
            'healing_staff_replenished_hp': 6,
            'healing_staff_range': 3,
            'shield_aggro_period': 5,
            'shield_aggro_cooldown_period': 8,
            'shield_range': 2,
            'shield_bash_accuracy': 1.0,
            'shield_damage_reduction': 1,
            'chain_immobilise_period': 5,
            'chain_immobilise_cooldown_period': 8,
            'chain_range' : 3,
            'chain_accuracy': 1.0,
            'scroll_nearby_dist': 2,
            'scroll_range': 4,
            'scroll_cooldown_period':8,
            'spellbook_range': 4,
            'spellbook_accuracy': 0.9,
            'spellbook_buff_increment': 0,
            'spellbook_buff_decrement': 0,
            'spellbook_buff_accuracy': 0.35,
            'spellbook_debuff_accuracy': 0.35,
            'spellbook_buff_effect_period': 5,
            'spellbook_buff_cooldown_period': 8
        }
    )
