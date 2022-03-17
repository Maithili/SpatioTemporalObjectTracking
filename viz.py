# %%
from ast import excepthandler
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

# sample_data = {
#     "quality_breakdown": [[3.0, 2.0], [1.0, 0.9], [1.5, 0.9], [1.0, 0.9], [0.2, 0.9]], 
#     "completeness_breakdown": {
#         "missed_changes": 5, 
#         "by_lookahead": [[3.0, 2.0], [1.0, 1], [1.5, 1], [1.0, 1], [0.5, 1]], 
#         "by_change_type": [[4.0, 2.0], [2.0, 3.0], [1.0, 1.0]]
#         }
# }
# sample_num=0


def visualize_eval_breakdowns(data, names, without_types=False):
    fig, ax = plt.subplots(2,3)
    ax_qual, ax_comp, ax_comp_qual = ax[0,0], ax[0,1], ax[0,2]
    ax_qual_trun, ax_comp_1step, ax_comp_qual2 = ax[1,0], ax[1,1], ax[1,2]

    for sample_num, sample_data in enumerate(data):
        if sample_data is None:
            continue

        qualities = []

        quality_steps = len(sample_data['quality_breakdown'])

        offsets = np.linspace(-0.45,0.45,quality_steps+1)
        offsets = (offsets[1:]+offsets[:-1])/2
        width = offsets[1] - offsets[0] - 0.01
        for step in range(quality_steps-1, -1, -1):
            ax_qual.bar(sample_num + offsets[step], sum(sample_data['quality_breakdown'][step])/1080, color=red-redder*step*0.5/quality_steps, width=width)
            ax_qual.bar(sample_num + offsets[step], sample_data['quality_breakdown'][step][0]/1080, color=green-greener*step*0.5/quality_steps, width=width)
            ax_qual_trun.bar(sample_num + offsets[step], sum(sample_data['quality_breakdown_truncated'][step])/1080, color=red-redder*step*0.5/quality_steps, width=width)
            ax_qual_trun.bar(sample_num + offsets[step], sample_data['quality_breakdown_truncated'][step][0]/1080, color=green-greener*step*0.5/quality_steps, width=width)
        qualities = [qb[0]/(sum(qb)+1e-8) for qb in sample_data['quality_breakdown']]
        qualities_trunc = [qb[0]/(sum(qb)+1e-8) for qb in sample_data['quality_breakdown_truncated']]

        comp_steps = len(sample_data['completeness_breakdown']['by_lookahead'])
        assert quality_steps==comp_steps

        completeness = []

        # if without_types:
        offsets = np.linspace(-0.45,0.45,comp_steps+1)
        offsets = (offsets[1:]+offsets[:-1])/2
        width = offsets[1] - offsets[0] - 0.01
        for step in range(comp_steps-1, -1, -1):
            ax_comp.bar(sample_num + offsets[step], sum(sample_data['completeness_breakdown']['by_lookahead'][step])/1080, color=red-redder*step*0.5/comp_steps, width=width)
            ax_comp.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/1080, color=green-greener*step*0.5/comp_steps, width=width)
            ax_comp_1step.bar(sample_num + offsets[step], sum(sample_data['completeness_breakdown_1step']['by_lookahead'][step])/1080, color=red-redder*step*0.5/comp_steps, width=width)
            ax_comp_1step.bar(sample_num + offsets[step], sample_data['completeness_breakdown_1step']['by_lookahead'][step][0]/1080, color=green-greener*step*0.5/comp_steps, width=width)
        completeness = [cb[0]/(sum(cb)+1e-8) for cb in sample_data['completeness_breakdown']['by_lookahead']]
        completeness_1step = [cb[0]/(sum(cb)+1e-8) for cb in sample_data['completeness_breakdown_1step']['by_lookahead']]
            
        # else:
        #     offsets = np.linspace(-0.45,0.45,comp_steps+2)
        #     offsets = (offsets[1:]+offsets[:-1])/2
        #     width = offsets[1] - offsets[0] - 0.01
        #     for step in range(comp_steps-1, -1, -1):
        #         ax_comp.bar(sample_num + offsets[step], sum(sample_data['completeness_breakdown']['by_lookahead'][step])/1080, color=red-redder*step*0.5/comp_steps, width=width) #, label=str(step)+'-proactivity wrong predictions')
        #         ax_comp.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/1080, color=green-greener*step*0.5/comp_steps, width=width)#, label=str(step)+'-proactivity correct predictions')
        #     completeness = [cb[0]/(cb[0]+cb[1]+cb[2]+1e-8) for cb in sample_data['completeness_breakdown']['by_lookahead']]

        #     bottom=0
        #     for ch, (pos, _, _) in enumerate(sample_data['completeness_breakdown']['by_change_type']):
        #         if sample_num == 1:
        #             ax_comp.bar(sample_num+offsets[-1], pos/1080, bottom=bottom, color=change_colors[ch], alpha=0.5, width=width, label=change_names[ch])#, label=str(step)+'-change correct predictions')
        #         else:
        #             ax_comp.bar(sample_num+offsets[-1], pos/1080, bottom=bottom, color=change_colors[ch], alpha=0.5, width=width)#, label=str(step)+'-change correct predictions')
        #         bottom += pos/1080

        #     for ch, (_, _, mis) in enumerate(sample_data['completeness_breakdown']['by_change_type']):
        #         ax_comp.bar(sample_num+offsets[-1], mis/1080, bottom=bottom, color=change_colors[ch], alpha=0.5, width=width)#, label=str(step)+'-change correct predictions')
        #         bottom += mis/1080

        #     for ch, (_, neg, _) in enumerate(sample_data['completeness_breakdown']['by_change_type']):
        #         ax_comp.bar(sample_num+offsets[-1], neg/1080, bottom=bottom, color=change_colors[ch], alpha=0.5, width=width)#, label=str(step)+'-change correct predictions')
        #         bottom += neg/1080

        alphas = np.linspace(1,0.5,quality_steps)
        for i in range(quality_steps):
            label = method_labels[names[sample_num]] if i==0 else None
            ax_comp_qual.plot(completeness[i], qualities[i], 'x', markersize=20, markeredgewidth = 5, label=label, color=method_colors[names[sample_num]], alpha=alphas[i])
            ax_comp_qual2.plot(completeness[i], qualities_trunc[i], 'x', markersize=20, markeredgewidth = 5, label=label, color=method_colors[names[sample_num]], alpha=alphas[i])
            

    # if not without_types:
    #     ax_comp.legend(fontsize=30)

    ax_comp_qual.legend(fontsize=30)
    ax_comp_qual.set_xlabel('Completeness', fontsize=30)
    ax_comp_qual.set_ylabel('Quality', fontsize=30)
    ax_comp_qual.set_xlim([0,1])
    ax_comp_qual.set_ylim([0,1])

    ax_comp_qual2.legend(fontsize=30)
    ax_comp_qual2.set_xlabel('Completeness', fontsize=30)
    ax_comp_qual2.set_ylabel('Quality Truncated', fontsize=30)
    ax_comp_qual2.set_xlim([0,1])
    ax_comp_qual2.set_ylim([0,1])
    
    ax_comp.set_xticks(np.arange(len(names)))
    ax_comp.set_xticklabels([method_labels[n] for n in names], fontsize=30)
    ax_comp.set_ylabel('Num. changes per step', fontsize=20)
    ax_comp.tick_params(axis = 'y', labelsize=20)
    ax_comp.tick_params(axis = 'x', labelsize=30)
    ax_comp.set_title('Completeness', fontsize=30)
    
    ax_qual.set_xticks(np.arange(len(names)))
    ax_qual.set_ylabel('Num. changes per step', fontsize=20)
    ax_qual.set_xticklabels([method_labels[n] for n in names], fontsize=30)
    ax_qual.tick_params(axis = 'y', labelsize=20)
    ax_qual.tick_params(axis = 'x', labelsize=30)
    ax_qual.set_title('Quality', fontsize=30)

    ax_comp_1step.set_xticks(np.arange(len(names)))
    ax_comp_1step.set_xticklabels([method_labels[n] for n in names], fontsize=30)
    ax_comp_1step.set_ylabel('Num. changes per step', fontsize=20)
    ax_comp_1step.tick_params(axis = 'y', labelsize=20)
    ax_comp_1step.tick_params(axis = 'x', labelsize=30)
    ax_comp_1step.set_title('Completeness (1 step)', fontsize=30)
    
    ax_qual_trun.set_xticks(np.arange(len(names)))
    ax_qual_trun.set_ylabel('Num. changes per step', fontsize=20)
    ax_qual_trun.set_xticklabels([method_labels[n] for n in names], fontsize=30)
    ax_qual_trun.tick_params(axis = 'y', labelsize=20)
    ax_qual_trun.tick_params(axis = 'x', labelsize=30)
    ax_qual_trun.set_title('Quality Truncated', fontsize=30)

    fig.set_size_inches(30,15)
    fig.tight_layout()


    return fig


# {
#   "quality_breakdown": [[28.500000000000004, 318.59999999999997], [0.30000000000000004, 2.3000000000000003], [0.0, 2.5999999999999996], [0.1, 2.5999999999999996], [0.1, 2.6999999999999993]], 
#   "completeness_breakdown": {"missed_changes": 21.6, 
#           "by_lookahead": [[28.500000000000004, 21.7], [0.30000000000000004, 0.0], [0.0, 0.0], [0.1, 0.0], [0.1, 0.0]], 
#           "by_change_type": [[9.8, 0.3, 19.799999952316284], [2.1, 21.4, 1.8000000715255737], [17.099999999999998, 0.0, 0.0]]
# }

def average_stats(stats_list):
    avg = {}
    num_stats = len(stats_list)
    if num_stats == 0:
        return None
    lookahead_steps = len(stats_list[0]['quality_breakdown'])
    avg['quality_breakdown'] = [[ sum([sl['quality_breakdown'][s][c] for sl in stats_list])/num_stats for c in range(2)]for s in range(lookahead_steps)]
    avg['quality_breakdown_truncated'] = [[ sum([sl['quality_breakdown_truncated'][s][c] for sl in stats_list])/num_stats for c in range(2)]for s in range(lookahead_steps)]
    lookahead_steps = len(stats_list[0]['completeness_breakdown']['by_lookahead'])
    avg['completeness_breakdown'] = {
        'by_lookahead' : [[ sum([sl['completeness_breakdown']['by_lookahead'][s][c] for sl in stats_list])/num_stats for c in range(3)]for s in range(lookahead_steps)],
        'by_change_type' : [[ sum([sl['completeness_breakdown']['by_change_type'][t][c] for sl in stats_list])/num_stats for c in range(3)]for t in range(3)]
    }
    avg['completeness_breakdown_1step'] = {
        'by_lookahead' : [[ sum([sl['completeness_breakdown_1step']['by_lookahead'][s][c] for sl in stats_list])/num_stats for c in range(3)]for s in range(lookahead_steps)],
        'by_change_type' : [[ sum([sl['completeness_breakdown_1step']['by_change_type'][t][c] for sl in stats_list])/num_stats for c in range(3)]for t in range(3)]
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