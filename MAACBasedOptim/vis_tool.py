import os
import wandb
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from collections import defaultdict
api = wandb.Api()

plot_ci = True
normalise_cbar = True
output_dir = "output"
os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "plot"), exist_ok=True)
env_names = [
        #"LBF",
        #"2DCorridor",
        #"Overcooked",
        "CircularOvercooked"
        ]
gen_methods = [
        "BRDiv",
        "AnyPlay",
        "TrajeDi0",
        "TrajeDi025",
        "TrajeDi05",
        "TrajeDi075",
        "TrajeDi1",
        "Independent"
    ]
eval_types = [
        "XAlg",
        "Heuristic"
    ]

project = "uoe-agents-div-team/diversity-gen"
summary_group_heuristic = "Returns/generalise/nondiscounted"
#summary_group_heuristic = "Returns/generalise/discounted"
summary_group_xalg = "Returns/gen_xp/nondiscounted"
#summary_group_xalg = "Returns/gen_xp/discounted"

def annotate(x, cl=0.95):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        ci = (mean, mean)
    else:
        ci = st.t.interval(cl, len(x), loc=mean, scale=st.sem(x))

    mean_str = f"{mean:.2f}"
    std_str = f"{std:.2f}"
    ci_str = f"({ci[0]:.2f},{ci[1]:.2f})"
    return f"{mean_str}\n{ci_str}"


returns = {}
for env_name, eval_type in product(env_names, eval_types):
    pair_key = f"{env_name}-{eval_type}"
    returns[pair_key] = {}
    for gen_method in gen_methods:
        tag_name = f"{env_name}-{gen_method}-Plastic-{eval_type}"
        runs = api.runs(project,
                       filters={"tags": tag_name}
                       )
        if runs:
            returns[pair_key][gen_method] = defaultdict(list)
            for run in runs:
                summary = run.summary
                for key, val in summary.items():
                    alg = key.split("/")[-1]
                    if key.startswith(summary_group_heuristic):
                        alg = f"H{int(alg.strip('H')):02}"
                        returns[pair_key][gen_method][alg].append(val)
                    elif key.startswith(summary_group_xalg):
                        returns[pair_key][gen_method][alg].append(val)

for pair_key, data in returns.items():
    if data == {}:
        continue
    df = pd.DataFrame(returns[pair_key])
    if "Heuristic" in pair_key:
        df = pd.DataFrame(returns[pair_key]).sort_index(axis=0)
    json_filename = os.path.join(output_dir, "data", f"{pair_key}.json")
    df.to_json(json_filename)
    df_mean = df.applymap(np.mean)
    df_annot = df.applymap(annotate)

    #print("DF Mean : ", df_mean)
    #print("DF Annot : ", df_annot)
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 12))
    if normalise_cbar:
        sfig = sns.heatmap(
                df_mean,
                vmin=0,
                vmax=1,
                cmap="mako",
                square=True,
                annot=df_annot if plot_ci else False,
                fmt="",
                )
    else:
        sfig = sns.heatmap(
                df_mean,
                cmap="mako",
                square=True,
                annot=df_annot if plot_ci else False,
                fmt="",
                )
    parsed_pair_key = pair_key.split("-")
 
    env_name = parsed_pair_key[0]
    if parsed_pair_key[0] == "2DCorridor":
       env_name = "Cooperative Reaching"
    elif parsed_pair_key[0] == "CircularOvercooked":
       env_name = "Simple Cooking"

    title_string = "Evaluation in "+env_name+" Against "
    
    y_label = "Evaluation Teammate Heuristic Name"
    if parsed_pair_key[1] == "Heuristic":
        title_string = title_string + "Heuristic Agents"
    else:
        title_string = title_string + "Generated Agents"     
        y_label = "Evaluation Teammate Generator Algorithm"

    #sfig.set(
    #    xlabel="Training Teammate Generator Algorithm",
    #    ylabel=y_label,
    #    title=title_string,
    #    )
    
    sfig.set_xlabel("Training Teammate Generator Algorithm", fontsize=18)
    sfig.set_ylabel(y_label, fontsize=18)
    sfig.set_title(title_string, fontsize=18)

    plot_filename = os.path.join(output_dir, "plot", f"{pair_key}.pdf")
    plt.savefig(plot_filename)

