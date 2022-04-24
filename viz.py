from copy import deepcopy
import os
import argparse
import glob
import random
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

method_colors = {
    'LastSeenAndStaticSemantic':'tab:green',
    'StaticSemantic':'tab:green',
    'FremenStateConditioned':'tab:blue',
    'Fremen':'tab:blue',
    'ours':'tab:red',
    'ours_timeLinear':'tab:orange',
    'ours_allEdges':'tab:pink',
    'ours_3epochs':'tab:orange',
    'ours_5epochs':'tab:orange',
    'ours_10epochs':'tab:orange',
    'ours_15epochs':'tab:pink',
    'ours_20epochs':'tab:pink',
    'ours_25epochs':'tab:pink',
    'ours_30epochs':'tab:pink'
}

def get_method_labels(ablation = ''):
    if ablation.lower() == 'ablation_time_':
        return {
        'ours':'Ours',
        'ours_timeLinear':'Ours w/ \nLinear Time',
        }
    if ablation.lower() == 'ablation_edges_':
        return {
        'ours':'Ours',
        'ours_allEdges':'Ours w/ \n All Edges'
        }
    if ablation == '': 
        return {
                # 'StaticSemantic':'Static\n Priors only',
                'LastSeenAndStaticSemantic':'Static\nSemantic',
                'FremenStateConditioned':'FreMEn',
                # 'Fremen':'FreMeN\n Priors only',
                'ours':'Ours w/ \n3 epochs',
                'ours_3epochs':'Ours w/ \n 3 epochs',
                'ours_5epochs':'Ours w/ \n 5 epochs',
                'ours_10epochs':'Ours w/ \n 10 epochs',
                'ours_15epochs':'Ours w/ \n 15 epochs',
                'ours_20epochs':'Ours w/ \n 20 epochs',
                'ours_25epochs':'Ours w/ \n 25 epochs',
                'ours_30epochs':'Ours w/ \n 30 epochs',
                }

filenames = ['recall_accuracy','precision','f1','precision_accuracy', 'precision_recall', 'recall_accuracy_norm', 'precision_norm', 'time_only_prediction']


def visualize_eval_breakdowns(data, names, ablation=''):
    # fig, ax = plt.subplots(2,3)
    f1, ax_comp_t_tl = plt.subplots()
    f2, ax_prec = plt.subplots()
    f3, ax_f1 = plt.subplots()
    f4, ax_comp_tl_prec = plt.subplots()
    f5, ax_comp_t_prec = plt.subplots()
    f6, ax_dest_acc_recl_norm = plt.subplots()
    f7, ax_prec_norm = plt.subplots()
    f8, ax_time_only = plt.subplots()
    figs =[f1,f2,f3,f4,f5,f6,f7, f8]

    method_labels = get_method_labels(ablation)

    info = {}

    offsets = np.linspace(-0.45,0.45,len(data[0]['precision_breakdown'])+1)
    offsets = (offsets[1:]+offsets[:-1])/2
    width = offsets[1] - offsets[0] - 0.01

    for sample_num, sample_data in enumerate(data):
        if sample_data is None or names[sample_num] not in method_labels:
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
            
        ax_time_only.bar(sample_num-0.21, sample_data['timeonly_breakdown_direct']['correct']/1080, color=green-greener*0.3, width=0.4)
        ax_time_only.bar(sample_num-0.21, sample_data['timeonly_breakdown_direct']['wrong']/1080, bottom=sample_data['timeonly_breakdown_direct']['correct']/1080, color=red-redder*0.3, width=0.4)
        ax_time_only.bar(sample_num+0.21, sample_data['timeonly_breakdown_playahead']['correct']/1080, color=green-greener*0.3, width=0.4)
        ax_time_only.bar(sample_num+0.21, sample_data['timeonly_breakdown_playahead']['wrong']/1080, bottom=sample_data['timeonly_breakdown_playahead']['correct']/1080, color=red-redder*0.3, width=0.4)

        info[names[sample_num]] = {}
        info[names[sample_num]]['precision'] = precisions
        info[names[sample_num]]['recall'] = completeness_t
        info[names[sample_num]]['destination_accuracy'] = completeness_tl
        info[names[sample_num]]['f1_score'] = f1
        info[names[sample_num]]['time_only_accuracy'] = {'direct':sample_data['timeonly_breakdown_direct']['correct']/(sample_data['timeonly_breakdown_direct']['correct']+sample_data['timeonly_breakdown_direct']['wrong']),
                                                         'direct':sample_data['timeonly_breakdown_playahead']['correct']/(sample_data['timeonly_breakdown_playahead']['correct']+sample_data['timeonly_breakdown_playahead']['wrong'])}


    ax_f1.set_xticks(np.arange(len(names)))
    ax_f1.set_xticklabels([method_labels[n] for n in names], fontsize=30)
    ax_f1.tick_params(axis = 'y', labelsize=20)
    ax_f1.tick_params(axis = 'x', labelsize=30)
    # ax_f1.set_title('F-1 Score', fontsize=30)
    ax_f1.set_ylim([0,1])

    ax_comp_t_prec.legend(fontsize=30)
    ax_comp_t_prec.set_xlabel('Recall', fontsize=30)
    ax_comp_t_prec.set_ylabel('Precision', fontsize=30)
    ax_comp_t_prec.set_xlim([0,1])
    ax_comp_t_prec.set_ylim([0,1])

    ax_comp_tl_prec.legend(fontsize=30)
    ax_comp_tl_prec.set_xlabel('Destination Accuracy', fontsize=30)
    ax_comp_tl_prec.set_ylabel('Precision', fontsize=30)
    ax_comp_tl_prec.set_xlim([0,1])
    ax_comp_tl_prec.set_ylim([0,1])
    
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
    # ax_prec.set_ylim([0,10])

    ax_prec_norm.legend(fontsize=35)
    ax_prec_norm.set_xticks(np.arange(len(names)))
    ax_prec_norm.set_xticklabels([method_labels[n] for n in names], fontsize=45)
    ax_prec_norm.tick_params(axis = 'y', labelsize=30)
    ax_prec_norm.tick_params(axis = 'x', labelsize=40)
    # ax_prec_norm.set_title('Precision', fontsize=30)
    ax_prec_norm.set_ylim([0,1])

    ax_dest_acc_recl_norm.legend(fontsize=35)
    ax_dest_acc_recl_norm.set_xticklabels([method_labels[n] for n in names], fontsize=45)
    ax_dest_acc_recl_norm.set_xticks(np.arange(len(names)))
    ax_dest_acc_recl_norm.tick_params(axis = 'y', labelsize=30)
    ax_dest_acc_recl_norm.tick_params(axis = 'x', labelsize=40)
    ax_dest_acc_recl_norm.set_ylim([ax_dest_acc_recl_norm.get_ylim()[0], ax_dest_acc_recl_norm.get_ylim()[1]+0.12])
    # ax_dest_acc_recl_norm.set_title('Recall & Destination Accuracy', fontsize=30)
    ax_dest_acc_recl_norm.set_ylim([0,1])

    ax_time_only.legend(fontsize=35)
    ax_time_only.set_xticks(np.arange(len(names)))
    ax_time_only.set_ylabel('Num. changes per step', fontsize=35)
    ax_time_only.set_xticklabels([method_labels[n] for n in names], fontsize=45)
    ax_time_only.tick_params(axis = 'y', labelsize=30)
    ax_time_only.tick_params(axis = 'x', labelsize=40)
    # ax_time_only.set_title('Time-based predictions', fontsize=30)

    for fig in figs:
        fig.set_size_inches(25,10)
        fig.tight_layout()

    f3.set_size_inches(25,8)
    f3.tight_layout()

    f4.set_size_inches(12,12)
    f4.tight_layout()
    f5.set_size_inches(12,12)
    f5.tight_layout()
    
    if ablation.startswith('ablation'):
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
    avg['timeonly_breakdown_direct'] = {k:sum([sl['timeonly_breakdown_direct'][k] for sl in stats_list])/num_stats for k in ['correct','wrong']}
    avg['timeonly_breakdown_playahead'] = {k:sum([sl['timeonly_breakdown_playahead'][k] for sl in stats_list])/num_stats for k in ['correct','wrong']}
    return avg

def result_string_from_info(info):
    info_averages = {kk:{k:np.mean(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()}
    info_mins = {kk:{k:min(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()}
    info_maxs = {kk:{k:max(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()}
    info_stds = {kk:{k:np.std(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()}
    methods = info.keys()
    string = ''
    for res in ['precision', 'recall', 'destination_accuracy', 'f1_score']:
        string += ('\n----- '+ res +' -----')
        for m in methods:
            string += ('\n{} : {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(m+' '*(25-len(m)), info_mins[m][res], info_averages[m][res]-info_stds[m][res], info_averages[m][res], info_averages[m][res]+info_stds[m][res], info_maxs[m][res]))
    string += '\n\n\n\n'
    string += '\n precision  recall  destination_accuracy  f1_score'
    second_best = {'precision':0,  'recall':0,  'destination_accuracy':0,  'f1_score':0}
    for m in methods:
        if m != 'ours':
            for k in second_best.keys():
                second_best[k] = max(second_best[k], info_averages[m][k])
        string += ('\n{} : {:.4f} & {:.4f} & {:.4f} & {:.4f} \\'.format(m+' '*(25-len(m)), info_averages[m]['precision'], info_averages[m]['recall'], info_averages[m]['destination_accuracy'], info_averages[m]['f1_score']))
    # m = 'second_best'
    # string += ('\n{} : {:.4f} & {:.4f} & {:.4f} & {:.4f} \\'.format(m+' '*(25-len(m)), second_best['precision'], second_best['recall'], second_best['destination_accuracy'], second_best['f1_score']))
    # perc_imp = [(info_averages['ours'][k] - second_best[k])/second_best[k] * 100 for k in ['precision', 'recall', 'destination_accuracy', 'f1_score']]
    # m = 'perc_improvement'
    # string += ('\n{} : {:2.2f} & {:2.2f} & {:2.2f} & {:2.2f} \\'.format(m+' '*(25-len(m)), perc_imp[0], perc_imp[1], perc_imp[2], perc_imp[3]))
    return string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--paths', type=str, default='logs/', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--combined_dir_out', type=str, help='Combining data from all dirs')
    args = parser.parse_args()

    f_f1, ax_f1 = plt.subplots()
    f_pr, ax_pr = plt.subplots()
    f_rc, ax_rc = plt.subplots()
    f_da, ax_da = plt.subplots()

    dirs = args.paths.split(',')
    master_combined_data = {k:{'precision':[], 'recall':[], 'destination_accuracy':[], 'f1_score':[]} for k in method_colors.keys()}

    for dir in dirs:
        if not dir.endswith('/'):
            dir += '/'
        directory_list = []
        directory_list += [os.path.join(dir,d) for d in os.listdir(dir)]
        dir_out = dir+'all'
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

        if len(datasets) == 0:
            continue


        gen_names = [n[:-2] if n[-2]=='_' else n for n in names]

        def get_combined_data(name, filter_dataset=lambda _: True):
            data_list = [(ds+'-'+n, d) for d,n,ds in zip(data, gen_names, datasets) if n==name and filter_dataset(ds)]
            cdata = average_stats([d[1] for d in data_list])
            return cdata

        ## per dataset
        print('Datasets : ')
        for dataset in set(datasets):
            ablation = ''
            print('Plotting :',dataset)
            combined_names = []
            combined_names = list(set([n for n in names if n[-2]!='_']))
            combined_names = [n for n in combined_names if n in get_method_labels(ablation)]
            combined_names.sort()
            print(combined_names)
            combined_data = [get_combined_data(name, lambda x: x==dataset) for name in combined_names]
            figs, info = visualize_eval_breakdowns(combined_data, combined_names, ablation=ablation)
            for i,fig in enumerate(figs):
                fig.savefig(os.path.join(dir_out.replace('all',dataset),filenames[i]+'.jpg'))
            with open(os.path.join(dir_out.replace('all',dataset),'info.json'), 'w') as f:
                json.dump(info, f)
            with open(os.path.join(dir_out.replace('all',dataset),'result.txt'), 'w') as f:
                f.write(result_string_from_info(info))
            info_averages = deepcopy({kk:{k:np.mean(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()})

        ## all data
        for ablation in ['']: #, 'ablation_time_', 'ablation_edges_']:
            print(ablation)
            combined_names = []
            combined_names = list(set([n for n in names if n[-2]!='_']))
            combined_names = [n for n in combined_names if n in get_method_labels(ablation)]
            combined_names.sort()
            combined_data = [get_combined_data(name, lambda x: True) for name in combined_names]
            figs, info = visualize_eval_breakdowns(combined_data, combined_names, ablation=ablation)
            for i,fig in enumerate(figs):
                fig.savefig(os.path.join(dir_out,ablation+filenames[i]+'.jpg'))
            with open(os.path.join(dir_out,ablation+'info.json'), 'w') as f:
                json.dump(info, f)
            with open(os.path.join(dir_out,ablation+'result.txt'), 'w') as f:
                f.write(result_string_from_info(info))

        print(info_averages.keys())
        for m in info_averages.keys():
            for res in info_averages[m].keys():
                master_combined_data[m][res].append(info_averages[m][res])

    for m in info_averages:
        ax_f1.plot(master_combined_data[m]['f1_score'], color=method_colors[m], label=m)
        ax_pr.plot(master_combined_data[m]['precision'], color=method_colors[m], label=m)
        ax_rc.plot(master_combined_data[m]['recall'], color=method_colors[m], label=m)
        ax_da.plot(master_combined_data[m]['destination_accuracy'], color=method_colors[m], label=m)


    labels = [os.path.basename(dir) for dir in dirs]
    for ax in [ax_f1, ax_pr, ax_rc, ax_da]:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Number of training days')
        ax.legend()

    if args.combined_dir_out:
        f_f1.savefig(os.path.join(args.combined_dir_out,'f1-score.jpg'))
        f_pr.savefig(os.path.join(args.combined_dir_out,'precision.jpg'))
        f_rc.savefig(os.path.join(args.combined_dir_out,'recall.jpg'))
        f_da.savefig(os.path.join(args.combined_dir_out,'destination-accuracy.jpg'))
