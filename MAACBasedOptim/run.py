import hydra
import torch
from train import DiversityTraining

@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg):
    #args = vars(args)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    torch.set_num_threads(hydra_cfg.launcher.get("cpus_per_task", 4))
    model_trainer = DiversityTraining(cfg)
    div_loss = model_trainer.run()
    return div_loss


if __name__ == '__main__':
    run()
