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
    # 'ours_timeLinear':'Ours w/ \nLinear Time'
}

ablation=False
filenames = ['recall_accuracy','precision','f1','precision_accuracy', 'precision_recall', 'recall_accuracy_norm', 'precision_norm']
if ablation:
    filenames = ['ablation_'+f for f in filenames]
    method_labels = {
    'ours':'Ours',
    'ours_timeLinear':'Ours w/ \nLinear Time'
    }

def visualize_eval_breakdowns(data, names, without_types=False):
    # fig, ax = plt.subplots(2,3)
    f1, ax_comp_t_tl = plt.subplots()
    f2, ax_prec = plt.subplots()
    f3, ax_f1 = plt.subplots()
    f4, ax_comp_tl_prec = plt.subplots()
    f5, ax_comp_t_prec = plt.subplots()
    f6, ax_dest_acc_recl_norm = plt.subplots()
    f7, ax_prec_norm = plt.subplots()
    figs =[f1,f2,f3,f4,f5,f6,f7]

    info = {}

    offsets = np.linspace(-0.45,0.45,len(data[0]['precision_breakdown'])+1)
    offsets = (offsets[1:]+offsets[:-1])/2
    width = offsets[1] - offsets[0] - 0.01

    for sample_num, sample_data in enumerate(data):
        if sample_data is None:
            continue

        quality_steps = len(sample_data['precision_breakdown'])
        for step in range(quality_steps-1, -1, -1):
            if sample_num == 0 and step == 0:
                ax_prec.bar(sample_num + offsets[step], sum(sample_data['precision_breakdown'][step])/1080, color=red-redder*0.3, width=width, label='False Positives')
                ax_prec.bar(sample_num + offsets[step], sample_data['precision_breakdown'][step][0]/1080, color=green, width=width, label='Correct Time')
                ax_prec_norm.bar(sample_num + offsets[step], sample_data['precision_breakdown'][step][0]/(sum(sample_data['precision_breakdown'][step])+1e-8), color=green-greener*0.2, width=width, label='Precision')
            else:
                ax_prec.bar(sample_num + offsets[step], sum(sample_data['precision_breakdown'][step])/1080, color=red-redder*0.3, width=width)
                ax_prec.bar(sample_num + offsets[step], sample_data['precision_breakdown'][step][0]/1080, color=green, width=width)
                ax_prec_norm.bar(sample_num + offsets[step], sample_data['precision_breakdown'][step][0]/(sum(sample_data['precision_breakdown'][step])+1e-8), color=green-greener*0.2, width=width)
        precisions = [qb[0]/(sum(qb)+1e-8) for qb in sample_data['precision_breakdown']]

        comp_steps = len(sample_data['completeness_breakdown']['by_lookahead'])
        assert quality_steps==comp_steps

        # if without_types:
        for step in range(comp_steps-1, -1, -1):
            if sample_num == 0 and step == 0:
                ax_comp_t_tl.bar(sample_num + offsets[step], sum(sample_data['completeness_breakdown']['by_lookahead'][step])/1080, color=red-redder*0.3, width=width, label='Wrong Time')
                ax_comp_t_tl.bar(sample_num + offsets[step], (sample_data['completeness_breakdown']['by_lookahead'][step][0]+sample_data['completeness_breakdown']['by_lookahead'][step][1])/1080, color=green-greener*0.3, width=width, label='Correct Time')
                ax_comp_t_tl.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/1080, color=green, width=width, label='Correct Time \n+ Destination')
                ax_dest_acc_recl_norm.bar(sample_num + offsets[step], (sample_data['completeness_breakdown']['by_lookahead'][step][0]+sample_data['completeness_breakdown']['by_lookahead'][step][1])/sum(sample_data['completeness_breakdown']['by_lookahead'][step]), color=green-greener*0.5, width=width, label='Recall')
                ax_dest_acc_recl_norm.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/sum(sample_data['completeness_breakdown']['by_lookahead'][step]), color=green-greener*0.0, width=width, label='Destination\nAccuracy')
        
            else:
                ax_comp_t_tl.bar(sample_num + offsets[step], sum(sample_data['completeness_breakdown']['by_lookahead'][step])/1080, color=red-redder*0.3, width=width)
                ax_comp_t_tl.bar(sample_num + offsets[step], (sample_data['completeness_breakdown']['by_lookahead'][step][0]+sample_data['completeness_breakdown']['by_lookahead'][step][1])/1080, color=green-greener*0.3, width=width)
                ax_comp_t_tl.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/1080, color=green, width=width)
                ax_dest_acc_recl_norm.bar(sample_num + offsets[step], (sample_data['completeness_breakdown']['by_lookahead'][step][0]+sample_data['completeness_breakdown']['by_lookahead'][step][1])/sum(sample_data['completeness_breakdown']['by_lookahead'][step]), color=green-greener*0.5, width=width)
                ax_dest_acc_recl_norm.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/sum(sample_data['completeness_breakdown']['by_lookahead'][step]), color=green-greener*0.0, width=width)
        completeness_tl = [cb[0]/(sum(cb)+1e-8) for cb in sample_data['completeness_breakdown']['by_lookahead']]
        completeness_t = [(cb[0]+cb[1])/(sum(cb)+1e-8) for cb in sample_data['completeness_breakdown']['by_lookahead']]

        f1 = [2*p*r/(p+r+1e-8) for p,r in zip(precisions, completeness_t)]
        ax_f1.bar(sample_num+offsets, f1, color=green-greener*0.2, width=width)

        alphas = np.linspace(1,0.5,quality_steps)
        for i in range(quality_steps):
            label = method_labels[names[sample_num]] if i==0 else None
            ax_comp_tl_prec.plot(completeness_tl[i], precisions[i], 'x', markersize=20, markeredgewidth = 5, label=label, color=method_colors[names[sample_num]], alpha=alphas[i])
            ax_comp_t_prec.plot(completeness_t[i], precisions[i], 'x', markersize=20, markeredgewidth = 5, label=label, color=method_colors[names[sample_num]], alpha=alphas[i])
            
        info[names[sample_num]] = {}
        info[names[sample_num]]['precision'] = precisions
        info[names[sample_num]]['recall'] = completeness_t
        info[names[sample_num]]['destination_accuracy'] = completeness_tl
        info[names[sample_num]]['f1_score'] = f1


    ax_f1.set_xticks(np.arange(len(names)))
    ax_f1.set_xticklabels([method_labels[n] for n in names], fontsize=30)
    ax_f1.tick_params(axis = 'y', labelsize=20)
    ax_f1.tick_params(axis = 'x', labelsize=30)
    # ax_f1.set_title('F-1 Score', fontsize=30)

    ax_comp_t_prec.legend(fontsize=30)
    ax_comp_t_prec.set_xlabel('Recall', fontsize=30)
    ax_comp_t_prec.set_ylabel('Precision', fontsize=30)
    ax_comp_t_prec.set_xlim([0,0.5])
    ax_comp_t_prec.set_ylim([0,0.5])

    ax_comp_tl_prec.legend(fontsize=30)
    ax_comp_tl_prec.set_xlabel('Destination Accuracy', fontsize=30)
    ax_comp_tl_prec.set_ylabel('Precision', fontsize=30)
    # ax_comp_tl_prec.set_xlim([0,1])
    # ax_comp_tl_prec.set_ylim([0,1])
    
    ax_comp_t_tl.legend(fontsize=35)
    ax_comp_t_tl.set_xticks(np.arange(len(names)))
    ax_comp_t_tl.set_xticklabels([method_labels[n] for n in names], fontsize=45)
    ax_comp_t_tl.set_ylabel('Num. changes per step', fontsize=35)
    ax_comp_t_tl.tick_params(axis = 'y', labelsize=30)
    ax_comp_t_tl.tick_params(axis = 'x', labelsize=40)
    # ax_comp_t_tl.set_title('Fraction of changes correctly predicted', fontsize=30)
    
    ax_prec.legend(fontsize=35)
    ax_prec.set_xticks(np.arange(len(names)))
    ax_prec.set_ylabel('Num. changes per step', fontsize=35)
    ax_prec.set_xticklabels([method_labels[n] for n in names], fontsize=45)
    ax_prec.tick_params(axis = 'y', labelsize=30)
    ax_prec.tick_params(axis = 'x', labelsize=40)
    # ax_prec.set_title('Correct fraction of predictions', fontsize=30)

    ax_prec_norm.legend(fontsize=35)
    ax_prec_norm.set_xticks(np.arange(len(names)))
    ax_prec_norm.set_xticklabels([method_labels[n] for n in names], fontsize=45)
    ax_prec_norm.tick_params(axis = 'y', labelsize=30)
    ax_prec_norm.tick_params(axis = 'x', labelsize=40)
    # ax_prec_norm.set_title('Precision', fontsize=30)

    ax_dest_acc_recl_norm.legend(fontsize=35)
    ax_dest_acc_recl_norm.set_xticklabels([method_labels[n] for n in names], fontsize=45)
    ax_dest_acc_recl_norm.set_xticks(np.arange(len(names)))
    ax_dest_acc_recl_norm.tick_params(axis = 'y', labelsize=30)
    ax_dest_acc_recl_norm.tick_params(axis = 'x', labelsize=40)
    # ax_dest_acc_recl_norm.set_title('Recall & Destination Accuracy', fontsize=30)

    for fig in figs:
        fig.set_size_inches(12,10)
        fig.tight_layout()

    f3.set_size_inches(15,8)
    f3.tight_layout()

    f4.set_size_inches(12,12)
    f4.tight_layout()
    f5.set_size_inches(12,12)
    f5.tight_layout()
    
    if ablation:
        for fig in figs:
            fig.set_size_inches(8,10)
            fig.tight_layout()

        f3.set_size_inches(8,5)
        f3.tight_layout()

        f4.set_size_inches(8,8)
        f4.tight_layout()
        f5.set_size_inches(8,8)
        f5.tight_layout()

    return figs, info



def average_stats(stats_list):
    avg = {}
    num_stats = len(stats_list)
    if num_stats == 0:
        return None
    lookahead_steps = len(stats_list[0]['precision_breakdown'])
    avg['precision_breakdown'] = [[ sum([sl['precision_breakdown'][s][c] for sl in stats_list])/num_stats for c in range(2)]for s in range(lookahead_steps)]
    lookahead_steps = len(stats_list[0]['completeness_breakdown']['by_lookahead'])
    avg['completeness_breakdown'] = {
        'by_lookahead' : [[ sum([sl['completeness_breakdown']['by_lookahead'][s][c] for sl in stats_list])/num_stats for c in range(3)]for s in range(lookahead_steps)],
        'by_change_type' : [[ sum([sl['completeness_breakdown']['by_change_type'][t][c] for sl in stats_list])/num_stats for c in range(3)]for t in range(3)]
    }
    return avg


dirs = ['logs0317newMetrics/']

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

# ## per dataset
# print('Datasets : ')
# for dataset in set(datasets):
#     print('Plotting :',dataset)
#     combined_data = [get_combined_data(name, lambda x: x==dataset) for name in combined_names]
#     # fig = visualize_eval_breakdowns(combined_data, combined_names)
#     # plt.savefig(os.path.join(dir_out,dataset+'_with_types.jpg'))
#     fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
#     plt.savefig(os.path.join(dir_out,dataset+'.jpg'))

## all data
print('Persona datasets : ')
combined_data = [get_combined_data(name, lambda x: not x.startswith('A')) for name in combined_names]
figs, info = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
for i,fig in enumerate(figs):
    fig.savefig(os.path.join(dir_out,filenames[i]+'.jpg'))
with open(os.path.join(dir_out,'info.json'), 'w') as f:
    json.dump(info, f)


# print('Individual datasets : ')
# combined_data = [get_combined_data(name, lambda x: x.startswith('A')) for name in combined_names]
# # fig = visualize_eval_breakdowns(combined_data, combined_names)
# # plt.savefig(os.path.join(dir_out,'allIndividual_with_types.jpg'))
# fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
# plt.savefig(os.path.join(dir_out,'allIndividual.jpg'))

# print('Persona datasets : ')
# combined_data = [get_combined_data(name, lambda x: not x.startswith('A')) for name in combined_names]
# # fig = visualize_eval_breakdowns(combined_data, combined_names)
# # plt.savefig(os.path.join(dir_out,'allPersona_with_types.jpg'))
# fig = visualize_eval_breakdowns(combined_data, combined_names, without_types=True)
# plt.savefig(os.path.join(dir_out,'allPersona.jpg'))