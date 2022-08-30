import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, sqrt

# TYPE = 'ablation'
TYPE = 'baselines'
# TYPE = 'pretrain'
# TYPE = 'without4'


red = np.array([164,22,35])/256
green = np.array([20,89,29])/256
neutral = np.array([245,226,200])/256


def method_color(name):
    if name == 'LastSeen':
        return 'tab:grey'
    if name.startswith('LastSeenAnd'):
        return 'tab:green'
    if name.startswith('StaticSemantic'):
        return 'tab:olive'
    if name.startswith('FremenState'):
        return 'tab:blue'
    if name.startswith('Fremen'):
        return 'tab:cyan'
    if name.startswith('ours_move'):
        return 'tab:orange'
    if name.startswith('ours_time'):
        return 'tab:pink'
    if name.startswith('ours_all'):
        return 'tab:purple'
    if name.startswith('ourspt_') and name.endswith('_0epochs'):
        return 'tab:purple'
    if name.startswith('ourspt_'):
        return 'tab:orange'
    if name.startswith('ours'):
        return 'tab:red'
    raise RuntimeError(f"Method color not defined for {name}")

def method_marker(name):
    if name.startswith('Fremen'):
        return 's'
    if name.startswith('LastSeenAnd'):
        return 'o'
    if name.startswith('ourspt'):
        return 'o'
    if name.startswith('ours_all'):
        return 'o'
    if name.startswith('ours_time'):
        return 's'
    if name.startswith('ours'):
        return '^'
    else:
        return '^'


def get_method_labels(method, ablation = ''):
    if ablation.lower() == 'ablation_':
        name_dict = {}
    if ablation == '': 
        name_dict = {
                'FremenStateConditioned1':'FreMEn',
                'FremenStateConditioned2':'FreMEn2',
                'LastSeenAndStaticSemantic4':'Stat Sem4',
                'LastSeenAndStaticSemantic5':'Stat Sem',
                'ours_50epochs':'Ours',
                'ourspt_50epochs':'PreTr',
                'ours_allEdges_50epochs':'w/o Attn',
                'ours_timeLinear_50epochs':'w/o Time',
                'ours_move2_50epochs':'Ours2',
                'ours_move3_50epochs':'Ours3',
                'ours_move4_50epochs':'Ours4',
                'FremenStateConditioned':'FreMEn1',
                'LastSeenAndStaticSemantic':'Stat Sem1',
                'LastSeenAndStaticSemantic2':'Stat Sem2',
                'FremenStateConditioned3':'FreMEn3',
                'LastSeenAndStaticSemantic3':'Stat Sem3',
                'FremenStateConditioned4':'FreMEn4',
                'LastSeen':'Nothing',
                'Fremen':'FreMEn_prior',
                'StaticSemantic':'StatSem_prior'
                }
    return name_dict[method]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='logs/CoRL_eval_0819_2204', help='')
    args = parser.parse_args()

    out_dir = os.path.join(args.path,'visuals', TYPE)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_days_all = os.listdir(args.path)
    visual_dirs = [d for d in train_days_all if d.startswith('visual')]
    for v in visual_dirs:
        train_days_all.remove(v)
    datasets = os.listdir(os.path.join(args.path,train_days_all[0]))
    methods = os.listdir(os.path.join(args.path,train_days_all[0],datasets[0]))
    train_days_all = [int(t) for t in train_days_all]
    train_days_all.sort()

    if 0 in train_days_all:
        train_days_all.remove(0)
    if TYPE == 'ablation':
        train_days_all = [50]
        methods = ['ours_50epochs', 'ours_allEdges_50epochs', 'ours_timeLinear_50epochs']
    elif TYPE == 'baselines':
        methods = ['FremenStateConditioned1', 'LastSeenAndStaticSemantic5', 'ours_50epochs']
    elif TYPE == 'pretrain':
        methods = ['ours_50epochs', 'ourspt_50epochs']
    elif TYPE.startswith('without'):
        methods = ['FremenStateConditioned1', 'LastSeenAndStaticSemantic5', 'ours_50epochs', 'ourspt_50epochs']
        datasets.remove('persona' + TYPE[-1])

    print(train_days_all)
    print(datasets)
    print(methods)

    metrics_separate = {}

    for method in methods:
        metrics_separate[method] = {}
        for dataset in datasets:
            metrics_separate[method][dataset] = {'correct':[], 'wrong':[], 'unmoved':[], 'final_unnecessarily_moved':[]}
            for train_days in train_days_all:
                file = os.path.join(args.path,str(train_days),dataset,method,'new_evaluation.json')
                eval = json.load(open(file))
                for k,v in eval.items():
                    metrics_separate[method][dataset][k].append(v)
            for k in metrics_separate[method][dataset]:
                    metrics_separate[method][dataset][k] = np.array(metrics_separate[method][dataset][k])
            


    metrics_average = {}
    metrics_stdev = {}
    metrics_sum = {}
    measures = ['precision','recall','precision_easy','recall_easy','precision_without_final','f1score']
    for method in methods:
        for dataset in datasets:
            tp = metrics_separate[method][dataset]['correct']
            gt = metrics_separate[method][dataset]['correct'] + metrics_separate[method][dataset]['wrong'] + metrics_separate[method][dataset]['unmoved']
            out = metrics_separate[method][dataset]['correct'] + metrics_separate[method][dataset]['wrong'] + metrics_separate[method][dataset]['final_unnecessarily_moved']
            metrics_separate[method][dataset]['precision'] = tp/out
            metrics_separate[method][dataset]['recall'] = tp/gt
            tp_easy = tp + metrics_separate[method][dataset]['wrong']
            metrics_separate[method][dataset]['precision_easy'] = tp_easy/out
            metrics_separate[method][dataset]['recall_easy'] = tp_easy/gt
            out_wo_final = metrics_separate[method][dataset]['correct'] + metrics_separate[method][dataset]['wrong']
            metrics_separate[method][dataset]['precision_without_final'] = tp/out_wo_final
            p,r = metrics_separate[method][dataset]['precision'], metrics_separate[method][dataset]['recall']
            metrics_separate[method][dataset]['f1score'] = 2 * p * r / (p+r)
        metrics_average[method] = {m:np.mean(np.stack([metrics_separate[method][d][m] for d in datasets]),axis=0) for m in measures}
        metrics_stdev[method] = {m:np.std(np.stack([metrics_separate[method][d][m] for d in datasets]),axis=0) for m in measures}
        metrics_sum[method] = {m:np.sum(np.stack([metrics_separate[method][d][m] for d in datasets]),axis=0) for m in ['correct','wrong','unmoved','final_unnecessarily_moved']}


    f_agg, ax_agg = plt.subplots(1,3)
    max_train_days = max(train_days_all)
    for i,train_days in enumerate(train_days_all):
        f_sep, ax_sep = plt.subplots(1,3)
        for method in methods:
            label = get_method_labels(method)
            for dataset in datasets:
                ax_sep[0].plot(metrics_separate[method][dataset]['recall_easy'][i], metrics_separate[method][dataset]['precision_easy'][i], label=label, marker=method_marker(method), color=method_color(method), markersize = 20, linewidth=3)
                ax_sep[1].plot(metrics_separate[method][dataset]['recall'][i], metrics_separate[method][dataset]['precision'][i], label=label, marker=method_marker(method), color=method_color(method), markersize = 20, linewidth=3)
                ax_sep[2].plot(metrics_separate[method][dataset]['recall'][i], metrics_separate[method][dataset]['precision_without_final'][i], label=label, marker=method_marker(method), color=method_color(method), markersize = 20, linewidth=3)
                label = None
            label = get_method_labels(method) if train_days == max_train_days else None
            if train_days in [10,20,30,40,50] or train_days == max_train_days:
                ax_agg[0].errorbar(metrics_average[method]['recall_easy'][i], metrics_average[method]['precision_easy'][i], xerr=metrics_stdev[method]['recall_easy'][i], yerr=metrics_stdev[method]['precision_easy'][i], label=label, marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20, alpha=sqrt(train_days/max_train_days))
                ax_agg[1].errorbar(metrics_average[method]['recall'][i], metrics_average[method]['precision'][i], xerr=metrics_stdev[method]['recall'][i], yerr=metrics_stdev[method]['precision'][i], label=label, marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20, alpha=sqrt(train_days/max_train_days))
                ax_agg[2].errorbar(metrics_average[method]['recall'][i], metrics_average[method]['precision_without_final'][i], xerr=metrics_stdev[method]['recall'][i], yerr=metrics_stdev[method]['precision_without_final'][i], label=label, marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20, alpha=sqrt(train_days/max_train_days))
            
        for f in [0.2, 0.4, 0.6]:
            pinv = np.linspace(1,2/f-1, 100)
            rinv = 2/f - pinv
            p1, p2 = 1/(2/f-1/0.75), 1/(2/f-1/0.95)
            for ax in list(ax_agg)+list(ax_sep):
                ax.plot(1/pinv, 1/rinv, color='grey', linewidth=(1-f)*2)
                ax.text(0.68,p2,f'F-1 score = {f}', rotation=np.rad2deg(atan2(p2-p1, 0.2)), fontsize=30, backgroundcolor=[1,1,1,0.5])
                ax.set_xlabel('Recall', fontsize=45)
                ax.set_ylabel('Precision', fontsize=45)
                ax.tick_params(axis = 'x', labelsize=30)
                ax.tick_params(axis = 'y', labelsize=30)
                ax.legend(fontsize=40)
                ax.set_xlim([0,1])
                ax.set_ylim([0,1])

        ax_sep[0].set_title('Object only', fontsize=45)
        ax_sep[1].set_title('Object and Destination', fontsize=45)
        ax_sep[2].set_title('Both w/o final movements', fontsize=45)

        f_sep.set_size_inches(50,20)
        f_sep.tight_layout()
        f_sep.savefig(os.path.join(out_dir,f'ROC_sep_{train_days}.jpg'))
    
    ax_agg[0].set_title('Object only', fontsize=45)
    ax_agg[1].set_title('Object and Destination', fontsize=45)
    ax_agg[2].set_title('Both w/o final movements', fontsize=45)
    ax_agg[0].axis('equal')
    ax_agg[1].axis('equal')
    ax_agg[2].axis('equal')
    f_agg.set_size_inches(35,12)
    f_agg.tight_layout()
    f_agg.savefig(os.path.join(out_dir,f'ROC_agg.jpg'))

    f, ax = plt.subplots(2,2)
    for method in methods:
        ax[0,0].errorbar(train_days_all, metrics_average[method]['recall_easy'], yerr = metrics_stdev[method]['recall_easy'], label=get_method_labels(method), marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20)
        ax[1,0].errorbar(train_days_all, metrics_average[method]['precision_easy'], yerr = metrics_stdev[method]['precision_easy'], label=get_method_labels(method), marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20)
        ax[0,1].errorbar(train_days_all, metrics_average[method]['recall'], yerr = metrics_stdev[method]['recall'], label=get_method_labels(method), marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20)
        ax[1,1].errorbar(train_days_all, metrics_average[method]['precision'], yerr = metrics_stdev[method]['precision'], label=get_method_labels(method), marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20)
    
    ax[0,0].set_title('Object Only', fontsize=45)
    ax[0,1].set_title('Object and Destination', fontsize=45)
    ax[0,0].set_ylabel('Recall', fontsize=45)
    ax[1,0].set_ylabel('Precision', fontsize=45)
    ax[0,1].set_ylabel('Recall', fontsize=45)
    ax[1,1].set_ylabel('Precision', fontsize=45)

    for a in ax.reshape(-1):
        a.set_xlabel('Number of training days', fontsize=40)
        a.legend(fontsize=40)
        a.tick_params(axis = 'x', labelsize=30)
        a.tick_params(axis = 'y', labelsize=30)
        # a.set_ylim([0,1])
    
    f.set_size_inches(50,30)
    f.tight_layout()
    f.savefig(os.path.join(out_dir,f'PrecisionRecall_all.jpg'))

    f, ax = plt.subplots(2,2)
    markers = {'persona0':'o', 'persona1':'x', 'persona2':'^', 'persona3':'s', 'persona4':'*'}
    for method in methods:
        label = get_method_labels(method)
        for dataset in datasets:
            ax[0,0].plot(train_days_all, metrics_separate[method][dataset]['recall_easy'], label=label, marker=markers[dataset], color=method_color(method), linewidth=3, markersize = 20)
            ax[1,0].plot(train_days_all, metrics_separate[method][dataset]['precision_easy'], label=label, marker=markers[dataset], color=method_color(method), linewidth=3, markersize = 20)
            label=None
        ax[0,1].errorbar(train_days_all, metrics_average[method]['recall'], yerr = metrics_stdev[method]['recall'], label=get_method_labels(method), marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20)
        ax[1,1].errorbar(train_days_all, metrics_average[method]['precision'], yerr = metrics_stdev[method]['precision'], label=get_method_labels(method), marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20)
    
    ax[0,0].set_title('Each Dataset', fontsize=45)
    ax[0,1].set_title('Aggregate metrics', fontsize=45)
    ax[0,0].set_ylabel('Recall', fontsize=45)
    ax[1,0].set_ylabel('Precision', fontsize=45)
    ax[0,1].set_ylabel('Recall', fontsize=45)
    ax[1,1].set_ylabel('Precision', fontsize=45)

    for a in ax.reshape(-1):
        a.set_xlabel('Number of training days', fontsize=40)
        a.legend(fontsize=40)
        a.tick_params(axis = 'x', labelsize=30)
        a.tick_params(axis = 'y', labelsize=30)
        # a.set_ylim([0,1])
    
    f.set_size_inches(50,30)
    f.tight_layout()
    f.savefig(os.path.join(out_dir,f'PrecisionRecall.jpg'))

    f, ax = plt.subplots(1,2)
    for method in methods:
        ax[0].errorbar(train_days_all, metrics_average[method]['recall'], yerr = metrics_stdev[method]['recall'], label=get_method_labels(method), marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20)
        ax[1].errorbar(train_days_all, metrics_average[method]['precision_without_final'], yerr = metrics_stdev[method]['precision_without_final'], label=get_method_labels(method), marker=method_marker(method), color=method_color(method), capsize=6.0, linewidth=3, markersize = 20)
    
    ax[0].set_ylabel('Recall', fontsize=45)
    ax[1].set_ylabel('Precision', fontsize=45)

    for a in ax.reshape(-1):
        a.set_xlabel('Number of training days', fontsize=40)
        a.legend(fontsize=40)
        a.tick_params(axis = 'x', labelsize=30)
        a.tick_params(axis = 'y', labelsize=30)
        a.set_ylim([0,1])
    
    f.set_size_inches(50,30)
    f.tight_layout()
    f.savefig(os.path.join(out_dir,f'PrecisionRecall_woFinal.jpg'))



    f, ax = plt.subplots()
    for method in methods:
        ax.errorbar(train_days_all, metrics_average[method]['f1score'], yerr=metrics_stdev[method]['f1score'], label=get_method_labels(method), marker=method_marker(method), color=method_color(method), linewidth=3, markersize = 20, capsize=6)
    
    ax.set_ylabel('F1-Score', fontsize=45)

    ax.set_xlabel('Number of training days', fontsize=40)
    ax.legend(fontsize=40)
    ax.tick_params(axis = 'x', labelsize=30)
    ax.tick_params(axis = 'y', labelsize=30)
    # ax.set_ylim([0,1])
    
    f.set_size_inches(15,10)
    f.tight_layout()
    f.savefig(os.path.join(out_dir,f'F1score.jpg'))


    f, ax = plt.subplots()
    for i,method in enumerate(methods):
        ax.bar(i, metrics_sum[method]['correct'][-1], color=green)
        bottom = metrics_sum[method]['correct'][-1]
        ax.bar(i, metrics_sum[method]['unmoved'][-1], bottom=bottom, color=neutral)
        bottom += metrics_sum[method]['unmoved'][-1]
        ax.bar(i, metrics_sum[method]['wrong'][-1], bottom=bottom, color=red)
    
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([get_method_labels(m) for m in methods], fontsize=35, rotation=90)

    ax.tick_params(axis = 'x', labelsize=30)
    ax.tick_params(axis = 'y', labelsize=30)
    
    f.set_size_inches(10,10)
    f.tight_layout()
    f.savefig(os.path.join(out_dir,f'Bars_GT.jpg'))

    f, ax = plt.subplots()
    for i,method in enumerate(methods):
        ax.bar(i, metrics_sum[method]['final_unnecessarily_moved'][-1], color='tab:orange')
    
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([get_method_labels(m) for m in methods], fontsize=35, rotation=90)

    ax.tick_params(axis = 'x', labelsize=30)
    ax.tick_params(axis = 'y', labelsize=30)
    
    f.set_size_inches(10,10)
    f.tight_layout()
    f.savefig(os.path.join(out_dir,f'Bars_FP.jpg'))

