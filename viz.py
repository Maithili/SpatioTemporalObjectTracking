import os
import glob
import json
from unicodedata import name
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

method_colors = {
    'LastSeenAndStaticSemantic':'tab:green',
    'FremenStateConditioned':'tab:blue',
    'ours':'tab:red',
    'ours_timeLinear':'tab:orange'
}

method_labels = {
    # 'StaticSemantic':'Static Priors only',
    'LastSeenAndStaticSemantic':'Static\nSemantic',
    'FremenStateConditioned':'FreMeN',
    # 'Fremen':'FreMeN Priors only',
    'ours':'Ours',
    'ours_timeLinear':'Ours w/ \nLinear Time'
}


def visualize_eval_breakdowns(data, names, without_types=False):
    fig, ax = plt.subplots(2,2)
    ax_comp_t_tl, ax_prec = ax[0,0], ax[0,1]
    ax_comp_tl_prec, ax_comp_t_prec = ax[1,0], ax[1,1]

    for sample_num, sample_data in enumerate(data):
        if sample_data is None:
            continue

        quality_steps = len(sample_data['quality_breakdown'])
        offsets = np.linspace(-0.45,0.45,quality_steps+1)
        offsets = (offsets[1:]+offsets[:-1])/2
        width = offsets[1] - offsets[0] - 0.01
        for step in range(quality_steps-1, -1, -1):
            ax_prec.bar(sample_num + offsets[step], sum(sample_data['quality_breakdown'][step])/1080, color=red, width=width)
            ax_prec.bar(sample_num + offsets[step], sample_data['quality_breakdown'][step][0]/1080, color=green, width=width)
        precisions = [qb[0]/(sum(qb)+1e-8) for qb in sample_data['quality_breakdown']]

        comp_steps = len(sample_data['completeness_breakdown']['by_lookahead'])
        assert quality_steps==comp_steps

        # if without_types:
        offsets = np.linspace(-0.45,0.45,comp_steps+1)
        offsets = (offsets[1:]+offsets[:-1])/2
        width = offsets[1] - offsets[0] - 0.01
        for step in range(comp_steps-1, -1, -1):
            ax_comp_t_tl.bar(sample_num + offsets[step], sum(sample_data['completeness_breakdown']['by_lookahead'][step])/1080, color=red-redder*0.9, width=width)
            ax_comp_t_tl.bar(sample_num + offsets[step], (sample_data['completeness_breakdown']['by_lookahead'][step][0]+sample_data['completeness_breakdown']['by_lookahead'][step][1])/1080, color=green-greener*0.5, width=width)
            ax_comp_t_tl.bar(sample_num + offsets[step], sample_data['completeness_breakdown_1step']['by_lookahead'][step][0]/1080, color=green, width=width)
        completeness_t = [cb[0]/(sum(cb)+1e-8) for cb in sample_data['completeness_breakdown']['by_lookahead']]
        completeness_tl = [(cb[0]+cb[1])/(sum(cb)+1e-8) for cb in sample_data['completeness_breakdown_1step']['by_lookahead']]

        alphas = np.linspace(1,0.5,quality_steps)
        for i in range(quality_steps):
            label = method_labels[names[sample_num]] if i==0 else None
            ax_comp_tl_prec.plot(completeness_tl[i], precisions[i], 'x', markersize=20, markeredgewidth = 5, label=label, color=method_colors[names[sample_num]], alpha=alphas[i])
            ax_comp_t_prec.plot(completeness_t[i], precisions[i], 'x', markersize=20, markeredgewidth = 5, label=label, color=method_colors[names[sample_num]], alpha=alphas[i])
            


    ax_comp_t_prec.legend(fontsize=30)
    ax_comp_t_prec.set_xlabel('Completeness (Time)', fontsize=30)
    ax_comp_t_prec.set_ylabel('Precision', fontsize=30)
    ax_comp_t_prec.set_xlim([0,1])
    ax_comp_t_prec.set_ylim([0,1])

    ax_comp_tl_prec.legend(fontsize=30)
    ax_comp_tl_prec.set_xlabel('Completeness (Time + Destination)', fontsize=30)
    ax_comp_tl_prec.set_ylabel('Precision', fontsize=30)
    ax_comp_tl_prec.set_xlim([0,1])
    ax_comp_tl_prec.set_ylim([0,1])
    
    ax_comp_t_tl.set_xticks(np.arange(len(names)))
    ax_comp_t_tl.set_xticklabels([method_labels[n] for n in names], fontsize=30)
    ax_comp_t_tl.set_ylabel('Num. changes per step', fontsize=20)
    ax_comp_t_tl.tick_params(axis = 'y', labelsize=20)
    ax_comp_t_tl.tick_params(axis = 'x', labelsize=30)
    ax_comp_t_tl.set_title('Completeness (Time, Time+Destination)', fontsize=30)
    
    ax_prec.set_xticks(np.arange(len(names)))
    ax_prec.set_ylabel('Num. changes per step', fontsize=20)
    ax_prec.set_xticklabels([method_labels[n] for n in names], fontsize=30)
    ax_prec.tick_params(axis = 'y', labelsize=20)
    ax_prec.tick_params(axis = 'x', labelsize=30)
    ax_prec.set_title('Precision', fontsize=30)

    fig.set_size_inches(30,15)
    fig.tight_layout()


    return fig



def average_stats(stats_list):
    avg = {}
    num_stats = len(stats_list)
    if num_stats == 0:
        return None
    lookahead_steps = len(stats_list[0]['quality_breakdown'])
    avg['quality_breakdown'] = [[ sum([sl['quality_breakdown'][s][c] for sl in stats_list])/num_stats for c in range(2)]for s in range(lookahead_steps)]
    lookahead_steps = len(stats_list[0]['completeness_breakdown']['by_lookahead'])
    avg['completeness_breakdown'] = {
        'by_lookahead' : [[ sum([sl['completeness_breakdown']['by_lookahead'][s][c] for sl in stats_list])/num_stats for c in range(3)]for s in range(lookahead_steps)],
        'by_change_type' : [[ sum([sl['completeness_breakdown']['by_change_type'][t][c] for sl in stats_list])/num_stats for c in range(3)]for t in range(3)]
    }
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
combined_names = [n for n in combined_names if n in method_labels]
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
    plt.savefig(os.path.join(dir_out,dataset+'_with_types.jpg'))
    fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
    plt.savefig(os.path.join(dir_out,dataset+'.jpg'))

## all data
print('All datasets : ')
combined_data = [get_combined_data(name, lambda x: True) for name in combined_names]
fig = visualize_eval_breakdowns(combined_data, combined_names)
plt.savefig(os.path.join(dir_out,'all_with_types.jpg'))
fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
plt.savefig(os.path.join(dir_out,'all.jpg'))


print('Individual datasets : ')
combined_data = [get_combined_data(name, lambda x: x.startswith('A')) for name in combined_names]
fig = visualize_eval_breakdowns(combined_data, combined_names)
plt.savefig(os.path.join(dir_out,'allIndividual_with_types.jpg'))
fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
plt.savefig(os.path.join(dir_out,'allIndividual.jpg'))

print('Persona datasets : ')
combined_data = [get_combined_data(name, lambda x: not x.startswith('A')) for name in combined_names]
fig = visualize_eval_breakdowns(combined_data, combined_names)
plt.savefig(os.path.join(dir_out,'allPersona_with_types.jpg'))
fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
plt.savefig(os.path.join(dir_out,'allPersona.jpg'))