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


def visualize_eval_breakdowns(data, names, without_types=False):
    fig, ax = plt.subplots(1,2)
    ax_r, ax_p = ax[0], ax[1]

    for sample_num, sample_data in enumerate(data):
        # if sample_data is None:
        #     continue

        recall_steps = len(sample_data['recall_breakdown'])

        prev_pos, prev_neg = 0, 0
        for step, (num_pos, num_neg) in enumerate(sample_data['recall_breakdown']) :
            ax_r.bar(sample_num, num_pos, bottom=prev_pos, color=green-greener*step/recall_steps) #, label=str(step)+'-proactivity correct predictions')
            ax_r.bar(sample_num, -num_neg, bottom=prev_neg, color=red-redder*step/recall_steps) #, label=str(step)+'-proactivity wrong predictions')
            prev_pos += num_pos
            prev_neg -= num_neg

        prec_steps = len(sample_data['precision_breakdown']['by_lookahead'])


        if without_types:
            bottom=0
            for step, (pos, _) in enumerate(sample_data['precision_breakdown']['by_lookahead']):
                ax_p.bar(sample_num, pos, bottom=bottom, color=green-greener*step/prec_steps)#, label=str(step)+'-proactivity correct predictions')
                bottom += pos

            ax_p.bar(sample_num, sample_data['precision_breakdown']['missed_changes'], bottom=bottom, color=neutral)#, label='missed changes')
            bottom += sample_data['precision_breakdown']['missed_changes']

            for step in range(len(sample_data['precision_breakdown']['by_lookahead'])-1, -1, -1):
                neg = sample_data['precision_breakdown']['by_lookahead'][step][1]
                ax_p.bar(sample_num, neg, bottom=bottom, color=red-redder*step/prec_steps)#, label=str(step)+'-proactivity wrong predictions')
                bottom += neg
            
        else:
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
            for ch, (pos, _, _) in enumerate(sample_data['precision_breakdown']['by_change_type']):
                if sample_num == 1:
                    ax_p.bar(sample_num+0.25, pos, bottom=bottom, color=change_colors[ch], alpha=0.5, width=0.3, label=change_names[ch])#, label=str(step)+'-change correct predictions')
                else:
                    ax_p.bar(sample_num+0.25, pos, bottom=bottom, color=change_colors[ch], alpha=0.5, width=0.3)#, label=str(step)+'-change correct predictions')
                bottom += pos

            for ch, (_, _, mis) in enumerate(sample_data['precision_breakdown']['by_change_type']):
                ax_p.bar(sample_num+0.25, mis, bottom=bottom, color=change_colors[ch], alpha=0.5, width=0.3)#, label=str(step)+'-change correct predictions')
                bottom += mis
            # ax_p.bar(sample_num+0.3, sample_data['precision_breakdown']['missed_changes'], bottom=bottom, color=neutral, width=0.3)#, label='missed changes')
            # bottom += sample_data['precision_breakdown']['missed_changes']

            for ch, (_, neg, _) in enumerate(sample_data['precision_breakdown']['by_change_type']):
                ax_p.bar(sample_num+0.25, neg, bottom=bottom, color=change_colors[ch], alpha=0.5, width=0.3)#, label=str(step)+'-change correct predictions')
                bottom += neg

    if not without_types:
        ax_p.legend()

    ax_p.set_xticks(np.arange(len(names)))
    ax_r.set_title('Predictions : Correct v.s. Incorrect', fontsize=30)
    ax_r.set_xticks(np.arange(len(names)))
    ax_p.set_xticklabels(names, rotation=90, fontsize=30)
    ax_r.set_xticklabels(names, rotation=90, fontsize=30)
    ax_p.set_title('Expected Predictions : Correct v.s. Missed v.s. Incorrect', fontsize=30)

    fig.set_size_inches(30,20)
    fig.tight_layout()


    return fig


# {
#   "recall_breakdown": [[28.500000000000004, 318.59999999999997], [0.30000000000000004, 2.3000000000000003], [0.0, 2.5999999999999996], [0.1, 2.5999999999999996], [0.1, 2.6999999999999993]], 
#   "precision_breakdown": {"missed_changes": 21.6, 
#           "by_lookahead": [[28.500000000000004, 21.7], [0.30000000000000004, 0.0], [0.0, 0.0], [0.1, 0.0], [0.1, 0.0]], 
#           "by_change_type": [[9.8, 0.3, 19.799999952316284], [2.1, 21.4, 1.8000000715255737], [17.099999999999998, 0.0, 0.0]]
# }

def average_stats(stats_list):
    avg = {}
    num_stats = len(stats_list)
    if num_stats == 0:
        return None
    avg['recall_breakdown'] = [[ sum([sl['recall_breakdown'][s][c] for sl in stats_list])/num_stats for c in range(2)]for s in range(5)]
    avg['precision_breakdown'] = {}
    avg['precision_breakdown']['by_lookahead'] = [[ sum([sl['precision_breakdown']['by_lookahead'][s][c] for sl in stats_list])/num_stats for c in range(2)]for s in range(5)]
    avg['precision_breakdown']['by_change_type'] = [[ sum([sl['precision_breakdown']['by_change_type'][t][c] for sl in stats_list])/num_stats for c in range(3)]for t in range(3)]
    avg['precision_breakdown']['missed_changes'] = sum([sl['precision_breakdown']['missed_changes'] for sl in stats_list])/num_stats
    return avg


dirs = ['logs/']

directory_list = []
for dir in dirs:
    directory_list += [os.path.join(dir,d) for d in os.listdir(dir)]
dir_out = 'visuals/all'
if not os.path.exists(dir_out): os.makedirs(dir_out)

data = []
names = []
datasets = []
for directory in directory_list:
    d = os.path.basename(directory)
    files=glob.glob(os.path.join(directory,'*','evaluation.json'))
    files.sort()
    for f in files:
        with open(f) as openfile:
            data.append(json.load(openfile))
        names.append(f.split('/')[-2])
        datasets.append(d)

combined_names = []
combined_names = list(set([n for n in names if n[-2]!='_']))
combined_names.sort()

gen_names = [n[:-2] if n[-2]=='_' else n for n in names]

import random
def get_combined_data(name, filter_dataset):
    data_list = [(ds+'-'+n, d) for d,n,ds in zip(data, gen_names, datasets) if n==name and filter_dataset(ds)]
    print([d[0] for d in data_list])
    cdata = average_stats([d[1] for d in data_list])
    # return (random.choice(data_list))[1]
    return cdata

## per dataset
print('Datasets : ')
for dataset in set(datasets):
    print('Plotting :',dataset)
    combined_data = [get_combined_data(name, lambda x: x==dataset) for name in combined_names]
    fig = visualize_eval_breakdowns(combined_data, combined_names)
    plt.savefig(os.path.join(dir_out,dataset+'.jpg'))
    fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
    plt.savefig(os.path.join(dir_out,dataset+'_without_types.jpg'))

## all data
print('All datasets : ')
combined_data = [get_combined_data(name, lambda x: True) for name in combined_names]
fig = visualize_eval_breakdowns(combined_data, combined_names)
plt.savefig(os.path.join(dir_out,'all.jpg'))
fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
plt.savefig(os.path.join(dir_out,'all_without_types.jpg'))


print('Individual datasets : ')
combined_data = [get_combined_data(name, lambda x: x.startswith('A')) for name in combined_names]
fig = visualize_eval_breakdowns(combined_data, combined_names)
plt.savefig(os.path.join(dir_out,'allIndividual.jpg'))
fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
plt.savefig(os.path.join(dir_out,'allIndividual_without_types.jpg'))

print('Persona datasets : ')
combined_data = [get_combined_data(name, lambda x: not x.startswith('A')) for name in combined_names]
fig = visualize_eval_breakdowns(combined_data, combined_names)
plt.savefig(os.path.join(dir_out,'allPersona.jpg'))
fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
plt.savefig(os.path.join(dir_out,'allPersona_without_types.jpg'))