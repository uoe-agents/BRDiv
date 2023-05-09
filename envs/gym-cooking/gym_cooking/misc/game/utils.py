import pygame
from gym_cooking.cooking_world.actions import *


class Color:
    BLACK = (0, 0, 0)
    FLOOR = (245, 230, 210)  # light gray
    COUNTER = (220, 170, 110)  # tan/gray
    COUNTER_BORDER = (114, 93, 51)  # darker tan
    DELIVERY = (96, 96, 96)  # grey


KeyToTuple = {
    pygame.K_UP: (0, -1),  # 273
    pygame.K_DOWN: (0, 1),  # 274
    pygame.K_RIGHT: (1, 0),  # 275
    pygame.K_LEFT: (-1, 0),  # 276
}

KeyToTuple_human1 = {
    pygame.K_UP: FullActionScheme.WALK_UP,  # 273
    pygame.K_DOWN: FullActionScheme.WALK_DOWN,  # 274
    pygame.K_RIGHT: FullActionScheme.WALK_RIGHT,  # 275
    pygame.K_LEFT: FullActionScheme.WALK_LEFT,  # 276
    pygame.K_SPACE: FullActionScheme.NO_OP,
    pygame.K_f: FullActionScheme.INTERACT_PRIMARY,
    pygame.K_g: FullActionScheme.INTERACT_PICK_UP_SPECIAL,
    pygame.K_e: FullActionScheme.EXECUTE_ACTION
}

KeyToTuple_Scheme2_human1 = {
    pygame.K_UP: EgoTurnScheme.WALK,  # 273
    pygame.K_RIGHT: EgoTurnScheme.TURN_RIGHT,  # 275
    pygame.K_LEFT: EgoTurnScheme.TURN_LEFT,  # 276
    pygame.K_SPACE: EgoTurnScheme.NO_OP,
    pygame.K_f: EgoTurnScheme.INTERACT_PRIMARY,
    pygame.K_g: EgoTurnScheme.INTERACT_PICK_UP_SPECIAL,
    pygame.K_e: EgoTurnScheme.EXECUTE_ACTION
}

KeyToTuple_human2 = {
    pygame.K_w: (0, -1),
    pygame.K_s: (0, 1),
    pygame.K_d: (1, 0),
    pygame.K_a: (-1, 0),
}
