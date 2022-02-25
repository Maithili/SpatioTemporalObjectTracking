# %%
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

red = np.array([164,22,35])/256
green = np.array([20,89,29])/256
neutral = np.array([245,226,200])/256

redder = red-neutral
greener = green-neutral

change_colors = ['tab:blue', 'tab:orange', 'tab:purple']
change_names = ['taking out', 'other', 'putting away']

sample_data = {
    "recall_breakdown": [[3.0, 2.0], [1.0, 0.9], [1.5, 0.9], [1.0, 0.9], [0.2, 0.9]], 
    "precision_breakdown": {
        "missed_changes": 5, 
        "by_lookahead": [[3.0, 2.0], [1.0, 1], [1.5, 1], [1.0, 1], [0.5, 1]], 
        "by_change_type": [[4.0, 2.0], [2.0, 3.0], [1.0, 1.0]]
        }
}
sample_num=0



def visualize_eval_breakdowns(data, names):
    fig, ax = plt.subplots(1,2)
    ax_r, ax_p = ax[0], ax[1]

    for sample_num, sample_data in enumerate(data):
        recall_steps = len(sample_data['recall_breakdown'])

        prev_pos, prev_neg = 0, 0
        for step, (num_pos, num_neg) in enumerate(sample_data['recall_breakdown']) :
            ax_r.bar(sample_num, num_pos, bottom=prev_pos, color=green-greener*step/recall_steps) #, label=str(step)+'-proactivity correct predictions')
            ax_r.bar(sample_num, -num_neg, bottom=prev_neg, color=red-redder*step/recall_steps) #, label=str(step)+'-proactivity wrong predictions')
            prev_pos += num_pos
            prev_neg -= num_neg

        prec_steps = len(sample_data['precision_breakdown']['by_lookahead'])

        bottom=0
        for step, (pos, _) in enumerate(sample_data['precision_breakdown']['by_lookahead']):
            ax_p.bar(sample_num-0.2, pos, bottom=bottom, color=green-greener*step/prec_steps, width=0.5)#, label=str(step)+'-proactivity correct predictions')
            bottom += pos

        ax_p.bar(sample_num-0.2, sample_data['precision_breakdown']['missed_changes'], bottom=bottom, color=neutral, width=0.5)#, label='missed changes')
        bottom += sample_data['precision_breakdown']['missed_changes']

        for step in range(len(sample_data['precision_breakdown']['by_lookahead'])-1, -1, -1):
            neg = sample_data['precision_breakdown']['by_lookahead'][step][1]
            ax_p.bar(sample_num-0.2, neg, bottom=bottom, color=red-redder*step/prec_steps, width=0.5)#, label=str(step)+'-proactivity wrong predictions')
            bottom += neg

        bottom=0
        for ch, (pos, _) in enumerate(sample_data['precision_breakdown']['by_change_type']):
            if sample_num == 1:
                ax_p.bar(sample_num+0.3, pos, bottom=bottom, color=change_colors[ch], width=0.3, label=change_names[ch])#, label=str(step)+'-change correct predictions')
            else:
                ax_p.bar(sample_num+0.3, pos, bottom=bottom, color=change_colors[ch], width=0.3)#, label=str(step)+'-change correct predictions')
            bottom += pos

        ax_p.bar(sample_num+0.3, sample_data['precision_breakdown']['missed_changes'], bottom=bottom, color=neutral, width=0.3)#, label='missed changes')
        bottom += sample_data['precision_breakdown']['missed_changes']

        for ch, (_, neg) in enumerate(sample_data['precision_breakdown']['by_change_type']):
            ax_p.bar(sample_num+0.3, neg, bottom=bottom, color=change_colors[ch], width=0.3)#, label=str(step)+'-change correct predictions')
            bottom += neg

    ax_p.legend()
    ax_p.set_xticks(np.arange(len(names)))
    ax_r.set_xticks(np.arange(len(names)))
    ax_p.set_xticklabels(names, rotation=90)
    ax_r.set_xticklabels(names, rotation=90)

    return fig


directory = 'logs/stochastic/basic'

data = []
names = []
files=glob.glob(os.path.join(directory,'*','evaluation.json'))
files.sort()
for f in files:
    with open(f) as openfile:
        data.append(json.load(openfile))
    names.append(f.split('/')[-2])
fig = visualize_eval_breakdowns(data, names)
plt.show()