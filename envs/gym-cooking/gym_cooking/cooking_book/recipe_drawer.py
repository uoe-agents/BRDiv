from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_book.recipe import Recipe, RecipeNode
from copy import deepcopy


def id_num_generator():
    num = 0
    while True:
        yield num
        num += 1


id_generator = id_num_generator()

#  Basic food Items
# root_type, id_num, parent=None, conditions=None, contains=None
ChoppedLettuce = RecipeNode(root_type=Lettuce, id_num=next(id_generator), name="Lettuce",
                            conditions=[("chop_state", ChopFoodStates.CHOPPED)], objects_to_seek=[Lettuce, Cutboard])
ChoppedOnion = RecipeNode(root_type=Onion, id_num=next(id_generator), name="Onion",
                          conditions=[("chop_state", ChopFoodStates.CHOPPED)], objects_to_seek=[Onion, Cutboard])
ChoppedTomato = RecipeNode(root_type=Tomato, id_num=next(id_generator), name="Tomato",
                           conditions=[("chop_state", ChopFoodStates.CHOPPED)], objects_to_seek=[Tomato, Cutboard])
ChoppedApple = RecipeNode(root_type=Apple, id_num=next(id_generator), name="Apple",
                          conditions=[("chop_state", ChopFoodStates.CHOPPED)], objects_to_seek=[Apple, Cutboard])
ChoppedCucumber = RecipeNode(root_type=Cucumber, id_num=next(id_generator), name="Cucumber",
                             conditions=[("chop_state", ChopFoodStates.CHOPPED)], objects_to_seek=[Cucumber, Cutboard])
ChoppedWatermelon = RecipeNode(root_type=Watermelon, id_num=next(id_generator), name="Watermelon",
                               conditions=[("chop_state", ChopFoodStates.CHOPPED)],
                               objects_to_seek=[Watermelon, Cutboard])
ChoppedBanana = RecipeNode(root_type=Banana, id_num=next(id_generator), name="Banana",
                           conditions=[("chop_state", ChopFoodStates.CHOPPED)], objects_to_seek=[Banana, Cutboard])
ChoppedCarrot = RecipeNode(root_type=Carrot, id_num=next(id_generator), name="Carrot",
                           conditions=[("chop_state", ChopFoodStates.CHOPPED)], objects_to_seek=[Carrot, Cutboard])
MashedCarrot = RecipeNode(root_type=Carrot, id_num=next(id_generator), name="Carrot",
                          conditions=[("blend_state", BlenderFoodStates.MASHED)], objects_to_seek=[Carrot, Blender])

# Salad Plates
TomatoSaladPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedTomato], objects_to_seek=[Tomato, Plate])
CarrotMashPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                             contains=[MashedCarrot], objects_to_seek=[Carrot, Plate])
TomatoLettucePlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedTomato, ChoppedLettuce], objects_to_seek=[(Tomato, Lettuce), Plate,
                                                                                           (Tomato, Lettuce), Plate])
TomatoLettuceOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                     contains=[ChoppedTomato, ChoppedLettuce, ChoppedOnion],
                                     objects_to_seek=[Tomato, Plate, Lettuce, Plate, Onion, Plate])

CarrotBananaPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[ChoppedCarrot, ChoppedBanana], objects_to_seek=[(Carrot, Banana), Plate,
                                                                                         (Carrot, Banana), Plate])

CucumberOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedCucumber, ChoppedOnion], objects_to_seek=[(Cucumber, Onion), Plate,
                                                                                           (Cucumber, Onion), Plate])

AppleWatermelonPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                  contains=[ChoppedApple, ChoppedWatermelon],
                                  objects_to_seek=[(Apple, Watermelon), Plate,
                                                   (Apple, Watermelon), Plate])
TomatoCarrotMashPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                   contains=[ChoppedTomato, MashedCarrot],
                                   objects_to_seek=[(Tomato, Carrot), Plate,
                                                    (Tomato, Carrot), Plate])

# Delivered Salads
TomatoSalad = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare", conditions=None,
                         contains=[TomatoSaladPlate], objects_to_seek=[Plate, Deliversquare])
CarrotMash = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare", conditions=None,
                        contains=[CarrotMashPlate], objects_to_seek=[Plate, Deliversquare])
TomatoLettuceSalad = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare",
                                conditions=None, contains=[TomatoLettucePlate]
                                , objects_to_seek=[Plate, Deliversquare])
TomatoLettuceOnionSalad = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare",
                                     conditions=None, contains=[TomatoLettuceOnionPlate],
                                     objects_to_seek=[Plate, Deliversquare])

CarrotBanana = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare", conditions=None,
                          contains=[CarrotBananaPlate], objects_to_seek=[Plate, Deliversquare])
CucumberOnion = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare", conditions=None,
                           contains=[CucumberOnionPlate], objects_to_seek=[Plate, Deliversquare])
AppleWatermelon = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare", conditions=None,
                             contains=[AppleWatermelonPlate], objects_to_seek=[Plate, Deliversquare])
TomatoCarrotMash = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare", conditions=None,
                             contains=[TomatoCarrotMashPlate], objects_to_seek=[Plate, Deliversquare])

floor = RecipeNode(root_type=Floor, id_num=next(id_generator), name="Floor", conditions=None, contains=[])
no_recipe_node = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name='Deliversquare', conditions=None,
                            contains=[floor], objects_to_seek=[])

# this one increments one further and is thus the amount of ids we have given since
# we started counting at zero.
NUM_GOALS = next(id_generator)

RECIPES = {"TomatoSalad": lambda: deepcopy(Recipe(TomatoSalad, NUM_GOALS)),
           "TomatoSaladPlate": lambda: deepcopy(Recipe(TomatoSaladPlate, NUM_GOALS)),
           "CarrotMash": lambda: deepcopy(Recipe(CarrotMash, NUM_GOALS)),
           "TomatoLettuceSalad": lambda: deepcopy(Recipe(TomatoLettuceSalad, NUM_GOALS)),
           "CarrotBanana": lambda: deepcopy(Recipe(CarrotBanana, NUM_GOALS)),
           "CucumberOnion": lambda: deepcopy(Recipe(CucumberOnion, NUM_GOALS)),
           "AppleWatermelon": lambda: deepcopy(Recipe(AppleWatermelon, NUM_GOALS)),
           "TomatoLettuceOnionSalad": lambda: deepcopy(Recipe(TomatoLettuceOnionSalad, NUM_GOALS)),
           "TomatoCarrotMash": lambda: deepcopy(Recipe(TomatoCarrotMash, NUM_GOALS)),
           "no_recipe": lambda: deepcopy(Recipe(no_recipe_node, NUM_GOALS))
           }
