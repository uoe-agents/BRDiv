# Diversity Experiments
This repository contains the implementation of the following methods:

- Our proposed approach (which optimises BRDiv)
- TrajeDi
- Any-Play
- Independent learning (No diversity metric maximisation)

All methods share the same implementation. The only difference lies in the different weights associated to the optimised loss functions within each method.

# Our Proposed Approach

To run our proposed approach, run the code with ``--jsd-weights=0``.

# Hybrid Approach

To run the hybrid approach, run the code with all hyperparameter's values set to non-zero.

# Independent Learning

To run this approach, change ``--jsd-weights=0``, ``--xp-loss-weights=0`` and do not include ``--gamma-act-jsd`` for optimisation.

# TrajeDi

To run this approach, set ``--xp-loss-weights=0``.

# Installing Required Packages
To setup the code for the experiments conducted in this paper, follow these instructions to install the required packages:
- Use the following command to create a clean conda environment (can be skipped if you want to install on top of an existing conda/virtualenv):
```bash
conda create -n <env name> python=3.9
```

- Install the necessary libraries to run the code via the following commands:
```bash
pip3 install wandb
wandb login
<Type in shared wandb password>
pip3 install pathlib
pip3 install gym
pip3 install hydra-core==1.2
pip3 install torch torchvision torchaudio
```

- Go inside every folder inside the ```envs``` folder and type the following command to install the environments used in our experiments:
```bash
pip install -e .
```

# Running Teammate Generation Experiments

- To run the teammate generation experiments, go inside the ```MAACBasedOptim``` folder and run the following command:
```bash
python3 run.py -cn <config_name>
```
In the above command ```<config_name>``` must correspond to your intended method to run. The name of the config file for each teammate generation method is provided below:
```
BRDiv       --> brdiv_*.yaml
Any-Play    --> anyplay.yaml
Independent --> independent.yaml
TrajeDi0    --> trajedi0.yaml
TrajeDi025  --> trajedi025.yaml
TrajeDi05   --> trajedi05.yaml
TrajeDi075  --> trajedi075.yaml
TrajeDi1    --> trajedi1.yaml
```
Note that these files are contained within the configs folder.

# Important Hyperparameters for Teammate Generation

A few important hyperparametersfor teammate generation are listed below:
- ``env.name``: Name of environment being tested. Note that it must belong from the following list: ``["MARL-CooperativeReaching-5-50-v0", "Foraging-6x6-2p-3f-coop-v2", "MARL-Circular-Overcooked-10-250-v0"]``. 
- ``train.lr``: Learning rate.
- ``env.parallel.sp_collection``: Number of threads for self-play data collection.
- ``env.parallel.xp_collection``: Number of threads for cross-play data collection.
- ``populations.num_populations``: Number of distinct policies to generate.
- ``loss_weights.xp_val_loss_weight``: Weights associated to training centralised critic based on XP data. Make sure this term is 0 for other methods than BRDiv.
- ``loss_weights.sp_val_loss_weight``: Weights associated to training centralised critic based on SP data.
- ``loss_weights.sp_rew_weight``: Weights associated to maximising trace of BRDiv's XP matrix.
- ``loss_weights.xp_loss_weight``: Weights associated to minimizing non-diagonal elements of BRDiv's XP matrix.
- ``loss_weights.jsd_weight``: Weights associated to maximising the Jensen-Shannon Divergence for TrajeDi. Make sure to set this to 0 for other methods than TrajeDi.
- ``loss_weights.entropy_regularizer_loss``: Weights associated to entropy regularizer to prevent policies from ceasing exploration too early.
- ``any_play.with_any_play``: Boolean flag on whether Any-Play is used during training or not. Set this to false when using other methods than Any-Play.
- ``any_play.any_play_classifier_loss_weight``: Weights associated to optimising the classifier used for Any-Play (to distinguish different policies based on state). Only matters when Any-Play is being used for training.
- ``any_play.any_play_lambda``: Before the output of Any-Play's classifier is used as an intrinsic reward to train actors, it is first multiplied with this weight constant.
- ``any_play.classifier_dims``: Size of hidden networks of the MLP used as Any-Play's classifier.  
- ``model.actor_dims``: The dimension of hidden layers of actor networks optimised by the teammate generation methods.
- ``model.critic_dims``: The dimension of hidden layers of critic networks optimised by the teammate generation methods.

# AHT Experiments With PLASTIC Policy

Running the AHT experiments requires some preparation steps that ensures that the evaluated generated teammates are provided in the correct folder. Only by positioning it correctly will the AHT code be able to retrieve them for the AHT experiments. 

- First, you must organize the models of your parameters under the following folder: 
```<folder_name_of_choice/<Approach Name>/<seed_id>/models>```.

- To run the codes AHT experiments based on the generated teammate policies, use the following command:
```
python3 run_plastic_adhoc_eval.py -cn <aht config filename in configs>
```

# Parameters of the AHT Experiments
Inside the AHT config yaml code in the configs folder, make sure to set the following fields correctly befor running the AHT code:
- ``env.name``: Make sure you use the same environment used during teammate generation.
- ``env.model_id``: Highlight the last checkpoint ID of the teammate generation algorithm.
- ``env.model_load_dir``: Provide the folder where the teammate policies can be loaded from (i.e. see first instruction from previous section).
- ``env_eval.name``: Provide the same string as ``env.name``, except that you add ``adhoc`` before ``-v0``.
- ``env_eval.eval_mode``: Choose ``heuristic`` to evaluate against heuristic-based teammates. Otherwise by default it will choose to evaluate against teammates generated by other methods if other strings are provided as value.
- ``env_eval.num_eval_heuristics``: Set the number of heuristics used in evaluation. 11 for coop reaching, 10 for LBF, and 12 for Simple Cooking.
- ``eval_params.all_params_dir``: Set this to the directory of the folder containing model parameters for all evaluated methods (i.e. ``<folder_name_of_choice>`` from the first instruction from previous section).
- ``populations.num_populations``: Set this to the number of generated policies during the teammate generation process. 

# Visualising AHT results
To generate the heatmaps that we provided in our experiments, you must follow these steps:

- In WandB, tag the runs resulting from the AHT experiments following this name:
```
<2DCorridor/LBF/CircularOvercooked>-<BRDiv/Independent/AnyPlay/TrajeDi0/TrajeDi025/TrajeDi05/TrajeDi075/TrajeDi1>-Plastic-<Heuristic (when evaluating against heuristic)/XAlg (otherwise)>
```

-. Run ``vis_tool.py``. Your resulting visualisation will be inside the ``output`` folder in ``MAACBasedOptim``.