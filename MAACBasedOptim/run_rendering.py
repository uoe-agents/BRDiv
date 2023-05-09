from render import PolicyRender
import yaml
from collections import namedtuple

if __name__ == '__main__':
    with open("/Volumesdiversity_paper_experiments/DiversityExperiments/MAACBasedOptim/configs/overcooked.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("error")


    def convert(dictionary):
        return namedtuple('GenericDict', dictionary.keys())(**dictionary)
    config = convert(config)

    renderer = PolicyRender(config)
