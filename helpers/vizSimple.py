from copy import deepcopy
from lib2to3.pgen2.literals import simple_escapes
from math import atan2, ceil
import os
import shutil
import argparse
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from breakdown_evaluations import activity_list

MAX_LOOKAHEAD = 1000
PLOT_CONFIDENCES = False

veryred = np.array([164,22,35])/256
verygreen = np.array([20,89,29])/256
neutral = np.array([245,226,200])/256*0.8

red = veryred * 0.7 + neutral * 0.3
green = verygreen * 0.7 + neutral * 0.3

num_routines = 10

change_colors = ['tab:blue', 'tab:orange', 'tab:purple']
change_names = ['taking out', 'other', 'putting away']

OUR_METHODS = ['Ours','Scratch']

def method_color(name):
    if name.startswith('Last'):
        return 'tab:green'
    if name.startswith('ours_activity25'):
        return 'tab:purple'
    if name.startswith('ours_activity50'):
        return 'tab:blue'
    if name.startswith('ours_activity75'):
        return 'tab:green'
    if name.startswith('ours_activity100'):
        return 'tab:olive'
    if name.startswith('noPreTr'):
        return 'tab:pink'
    if name.startswith('Fremen'):
        return 'tab:blue'
    if name.startswith('ourspt0_'):
        return 'tab:orange'
    if name.startswith('ourspt_') and name.endswith('_0epochs'):
        return 'tab:purple'
    if name.startswith('ourspt_'):
        return 'tab:orange'
    if name.startswith('ours_all'):
        return 'tab:pink'
    if name.startswith('ours_time'):
        return 'tab:purple'
    if name.startswith('ours'):
        return 'tab:red'
    if name.startswith('original'):
        return 'k'
    raise RuntimeError(f"Method color not defined for {name}")

def method_marker(name):
    if name.startswith('Last'):
        return 'o'
    if name.startswith('ours_activity25'):
        return 'o'
    if name.startswith('ours_activity50'):
        return 'x'
    if name.startswith('ours_activity75'):
        return '^'
    if name.startswith('ours_activity100'):
        return 's'
    if name.startswith('Fremen'):
        return 's'
    if name.startswith('ourspt0_'):
        return '.'
    if name.startswith('ourspt_') and name.endswith('_0epochs'):
        return '*'
    if name.startswith('ourspt_'):
        return 's'
    if name.startswith('ours_all'):
        return 's'
    if name.startswith('ours_time'):
        return 'o'
    if name.startswith('ours'):
        return '^'
    else:
        return '.'

def get_activity_difficulty(dataset_name, persona_name):
    stats_file = os.path.join('data',dataset_name, persona_name, 'activity_stats.json')
    probabilities = json.load(open(stats_file))["prob_days"]
    probabilities[None] = 0
    return probabilities

def method_label(name, ablation='default'):
    all_name_dict = {}
    all_name_dict['default'] = {}
    all_name_dict['baselines'] = {
            'ours_':'Ours',
            'ours':'Ours',
            'FremenStateConditioned':'FreMEn',
            'LastSeenAndStaticSemantic':'Stat Sem'
            }
    all_name_dict['pretrain'] = {
            'ourspt_':'Pre-Train',
            'ourspt0_':'PreTr0only',
            'ours_':'Scratch',
            }
    all_name_dict['ablations'] = {
            'ours_':'Ours',
            'ours':'Ours',
            'ours_allEdges_':'w/o Attn',
            'ours_allEdges':'w/o Attn',
            'ours_timeLinear_':'w/o Time',
            'ours_timeLinear':'w/o Time',
            }
    all_name_dict['ablations_activity'] = {
            'ours_':'Ours',
            'ours_activity25_':'Act-25',
            'ours_activity50_':'Act-50',
            'ours_activity75_':'Act-75',
            'ours_activity100_':'Act-100',
            'original_':'Old'
            }
    all_name_dict['custom'] = {
            'noPreTr_':'NoPT',
            'ours_':'Ours',
            'ours_':'Ours',
            'ours_overshoot_':'OverShoot',
            'ours_endgoal_pred_': 'GoalPred',
            'ours_latent_simil_': 'LatentSimil', 
            'ours_no_overshoot_': 'NoOversht',
            'ours_activity25_':'Act-25',
            'ours_activity50_':'Act-50',
            'ours_activity75_':'Act-75',
            'ours_activity100_':'Act-100',
            'ours_activity25_':'Act-25',
            'ours_activity50_':'Act-50',
            'ours_activity75_':'Act-75',
            'ours_activity100_':'Act-100',
            'original_':'Old'
            }
    name_dict = all_name_dict[ablation]
    return name_dict[name]

filenames = ['roc_inc_dest', 'roc_obj_only', 'num_changes', 'recall_by_type', 'recall_by_activity', 'UnmovedViolin', 'Moved', 'Moved_Line', 'Activity']


def visualize_eval_breakdowns(data, names, colors, markers, activity_difficulty=None):
    f1, ax_roc = plt.subplots()
    f2, ax_roc_onlyobj = plt.subplots()
    f3, ax_num_changes = plt.subplots()
    f4, ax_recl_by_changetype = plt.subplots()
    f5, ax_recl_by_activity = plt.subplots()
    f6, ax_succes_on_unmoved = plt.subplots(1,3)
    f7, ax_breakdown_on_moved = plt.subplots(1,2)
    f8, ax_breakdown_on_moved_line = plt.subplots(1,2)
    f9, ax_activity = plt.subplots()
    figs = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

    for ax in [ax_roc, ax_roc_onlyobj]:
        for f in [0.2, 0.4, 0.6]:
            pinv = np.linspace(1,2/f-1, 100)
            rinv = 2/f - pinv
            ax.plot(1/pinv, 1/rinv, color='grey', linewidth=(1-f)*2)
            p1, p2 = 1/(2/f-1/0.75), 1/(2/f-1/0.95)
            ax.text(0.68,p2,f'F-1 score = {f}', rotation=np.rad2deg(atan2(p2-p1, 0.2)), fontsize=30, backgroundcolor=[1,1,1,0.5])

    lookahead_steps = None

    info = {}

    print(names)
    

    all_correct_percs = []
    all_missed_percs = []
    all_wrong_percs = []
    all_unmoved_successes = []

    for sample_num, sample_data in enumerate(data):
        lookahead_steps = len(sample_data['moved']['correct'])
        offsets = np.linspace(-0.45,0.45,lookahead_steps+1)
        offsets = (offsets[1:]+offsets[:-1])/2
        width = offsets[1] - offsets[0] - 0.01

        success_on_unmoved = sample_data['unmoved']['tn']/sample_data['unmoved']['sum']
        all_unmoved_successes.append(success_on_unmoved)
        # ax_succes_on_unmoved[0].plot(sample_num, success_on_unmoved, markersize = 20, marker=(markers[sample_num]), label=(names[sample_num]), color=colors[sample_num], linewidth=3)
        ax_succes_on_unmoved[1].bar([sample_num + o for o in offsets], sample_data['unmoved']['tn'], color=green, width=width)
        ax_succes_on_unmoved[1].bar([sample_num + o for o in offsets], sample_data['unmoved']['fp'], bottom=sample_data['unmoved']['tn'], color=red, width=width)
        success_on_unmoved_rounded = int(round(sum(sample_data['unmoved']['tn'])/sum(sample_data['unmoved']['sum'])*1000))/1000
        ps = ax_succes_on_unmoved[2].bar(sample_num, 1-success_on_unmoved_rounded, bottom=success_on_unmoved_rounded, color=red)
        pf = ax_succes_on_unmoved[2].bar(sample_num, success_on_unmoved_rounded, color=green)
        # ax_succes_on_unmoved[2].bar_label(ps, label_type='center', fontsize=30)
        ax_succes_on_unmoved[2].bar_label(pf, label_type='center', fontsize=30)
        ax_succes_on_unmoved[2].errorbar(sample_num, np.mean(success_on_unmoved), yerr=[[np.mean(success_on_unmoved)-success_on_unmoved.min()], [success_on_unmoved.max()-np.mean(success_on_unmoved)]], ecolor='k', elinewidth=4, capsize=8)
        
        if sample_num == 0:
            labels_brkdwn = ['Correct','Missed','Wrong']
        
        all_correct_percs.append(sum(sample_data['moved']['correct'])/sum(sample_data['moved']['sum']))
        all_wrong_percs.append(sum(sample_data['moved']['wrong'])/sum(sample_data['moved']['sum']))
        all_missed_percs.append(sum(sample_data['moved']['missed'])/sum(sample_data['moved']['sum']))
        correct_rounded = int(round(sum(sample_data['moved']['correct'])/sum(sample_data['moved']['sum'])*1000))/1000
        wrong_rounded = int(round(sum(sample_data['moved']['wrong'])/sum(sample_data['moved']['sum'])*1000))/1000
        missed_rounded = int(round(sum(sample_data['moved']['missed'])/sum(sample_data['moved']['sum'])*1000))/1000
        pw = ax_breakdown_on_moved[1].bar(sample_num, wrong_rounded, bottom=missed_rounded+correct_rounded, color=red, label=labels_brkdwn[2])
        pc = ax_breakdown_on_moved[1].bar(sample_num, correct_rounded, color=green, label=labels_brkdwn[0])
        # pm = ax_breakdown_on_moved[1].bar(sample_num, missed_rounded, bottom=correct_rounded, color=neutral, label=labels_brkdwn[1])
        correct_perc = sample_data['moved']['correct']/sample_data['moved']['sum']
        wrong_perc = sample_data['moved']['wrong']/sample_data['moved']['sum']
        missed_perc = sample_data['moved']['missed']/sample_data['moved']['sum']
        ax_breakdown_on_moved[1].errorbar(sample_num, np.mean(correct_perc), yerr=[[np.mean(correct_perc)-correct_perc.min()], [correct_perc.max()-np.mean(correct_perc)]], ecolor='k', elinewidth=4, capsize=8)
        ax_breakdown_on_moved[1].errorbar(sample_num, np.mean(correct_perc + missed_perc), yerr=[[np.mean(wrong_perc)-wrong_perc.min()], [wrong_perc.max()-np.mean(wrong_perc)]], ecolor='k', elinewidth=4, capsize=8)
        # ax_breakdown_on_moved[1].errorbar(sample_num, np.mean(correct_perc + missed_perc), yerr=[[np.mean(wrong_perc).max()-np.mean(wrong_perc)], [np.mean(wrong_perc)-np.mean(wrong_perc).min()]], ecolor='k', elinewidth=4, capsize=8)
        
        ax_breakdown_on_moved_line[0].plot(sample_data['moved']['correct']/sample_data['moved']['sum'], color=colors[sample_num], marker=markers[sample_num], linewidth=3, markersize=20)
        ax_breakdown_on_moved_line[1].plot((sample_data['moved']['missed']+sample_data['moved']['correct'])/sample_data['moved']['sum'], color=colors[sample_num], marker=markers[sample_num], linewidth=3, markersize=20)

        # ax_breakdown_on_moved[1].bar_label(pc, label_type='center', fontsize=30)
        # # ax_breakdown_on_moved[1].bar_label(pm, label_type='center', fontsize=30)
        # ax_breakdown_on_moved[1].bar_label(pw, label_type='center', fontsize=30)
        ax_breakdown_on_moved[0].bar(sample_num + offsets, sample_data['moved']['correct'], color=green, width=width, label=labels_brkdwn[0])
        ax_breakdown_on_moved[0].bar(sample_num + offsets, sample_data['moved']['missed'], bottom=sample_data['moved']['correct'], color=neutral, width=width, label=labels_brkdwn[1])
        ax_breakdown_on_moved[0].bar(sample_num + offsets, sample_data['moved']['wrong'], bottom=sample_data['moved']['missed']+sample_data['moved']['correct'], color=red, width=width, label=labels_brkdwn[2])
        labels_brkdwn = [None, None, None]

        ax_num_changes.plot(np.arange(lookahead_steps)+1, sample_data['moved']['correct'] + sample_data['moved']['wrong'] + sample_data['unmoved']['fp'], markersize = 20, marker=(markers[sample_num]), label=(names[sample_num]), color=colors[sample_num], linewidth=3)
        precisions = sample_data['moved']['correct']/(sample_data['moved']['correct'] + sample_data['moved']['wrong'] + sample_data['unmoved']['fp'] + 1e-8)
        recalls = sample_data['moved']['correct']/(sample_data['moved']['sum'] + 1e-8)
        precisions_objonly = (sample_data['moved']['correct'] + sample_data['moved']['wrong'])/(sample_data['moved']['correct'] + sample_data['moved']['wrong'] + sample_data['unmoved']['fp'] + 1e-8)
        recalls_objonly = (sample_data['moved']['correct'] + sample_data['moved']['wrong'])/(sample_data['moved']['sum'] + 1e-8)

        alphas = np.linspace(1,0.2,lookahead_steps)
        ax_roc.scatter(recalls, precisions, marker=markers[sample_num], label=names[sample_num], color=colors[sample_num], alpha=alphas)
        ax_roc_onlyobj.scatter(recalls_objonly, precisions_objonly, marker=markers[sample_num], label=names[sample_num], color=colors[sample_num], alpha=alphas)

        ax_activity.bar(sample_num + offsets, sample_data['activity']['correct']/sample_data['activity']['sum'], color=green, width=width)

        # if 'by_change_type' in sample_data['completeness_breakdown'].keys():
        #     ax_recl_by_changetype.plot([1,2,3], [(sample_data['completeness_breakdown']['by_change_type'][ci][0]+sample_data['completeness_breakdown']['by_change_type'][ci][1])/sum(sample_data['completeness_breakdown']['by_change_type'][ci]) for ci in range(3)], linewidth=3, linestyle='dashed', marker=markers[sample_num], color=colors[sample_num], markersize = 20)
        #     ax_recl_by_changetype.plot([1,2,3], [(sample_data['completeness_breakdown']['by_change_type'][ci][0])/sum(sample_data['completeness_breakdown']['by_change_type'][ci]) for ci in range(3)], linewidth=3, label=(names[sample_num]), marker=markers[sample_num], color=colors[sample_num], markersize = 20)

        # if 'by_activity' in sample_data['completeness_breakdown'].keys():
        #     if activity_difficulty is None:
        #         ax_recl_by_activity.plot(np.arange(len(activity_list)), [(sample_data['completeness_breakdown']['by_activity'][ci][0]+sample_data['completeness_breakdown']['by_activity'][ci][1])/(sum(sample_data['completeness_breakdown']['by_activity'][ci])+1e-8) for ci in range(len(activity_list))], linewidth=3, label=(names[sample_num]), marker=markers[sample_num], color=colors[sample_num], markersize = 20)
        #         ax_recl_by_activity.plot(np.arange(len(activity_list)), [(sample_data['completeness_breakdown']['by_activity'][ci][0])/(sum(sample_data['completeness_breakdown']['by_activity'][ci])+1e-8) for ci in range(len(activity_list))], linewidth=3, marker=markers[sample_num], color=colors[sample_num], markersize = 20, linestyle='dashed')
        #         ax_recl_by_activity.set_xticks(np.arange(len(activity_list)))
        #     else:
        #         # ax_recl_by_activity.scatter([activity_difficulty[a] for a in activity_list], [(sample_data['completeness_breakdown']['by_activity'][ci][0]+sample_data['completeness_breakdown']['by_activity'][ci][1])/(sum(sample_data['completeness_breakdown']['by_activity'][ci])+1e-8) for ci in range(len(activity_list))], label=(names[sample_num]), marker=markers[sample_num], color=colors[sample_num], markersize = 20)
        #         ax_recl_by_activity.scatter([activity_difficulty[a] for a in activity_list], [(sample_data['completeness_breakdown']['by_activity'][ci][0])/(sum(sample_data['completeness_breakdown']['by_activity'][ci])+1e-8) for ci in range(len(activity_list))], marker=markers[sample_num], color=colors[sample_num]) #, markersize = 20)
        #         ax_recl_by_activity.set_xticks([activity_difficulty[a] for a in activity_list])

        info[names[sample_num]] = {}
        info[names[sample_num]]['ObjOnly_precision'] = list(precisions_objonly)
        info[names[sample_num]]['ObjOnly_recall'] = list(recalls_objonly)
        info[names[sample_num]]['ObjOnly_f1_score'] = list(2*precisions_objonly*recalls_objonly/(precisions_objonly+recalls_objonly))
        info[names[sample_num]]['IncDest_precision'] = list(precisions)
        info[names[sample_num]]['IncDest_recall'] = list(recalls)
        info[names[sample_num]]['IncDest_f1_score'] = list(2*precisions*recalls/(precisions+recalls))
        info[names[sample_num]]['Correct'] = list(sample_data['moved']['correct'].astype(float))
        info[names[sample_num]]['Wrong'] = list(sample_data['moved']['wrong'].astype(float))
        info[names[sample_num]]['Missed'] = list(sample_data['moved']['missed'].astype(float))
        info[names[sample_num]]['FP'] = list(sample_data['unmoved']['fp'].astype(float))
        info[names[sample_num]]['TN'] = list(sample_data['unmoved']['tn'].astype(float))

    ax_breakdown_on_moved[0].legend(fontsize=40)
    ax_breakdown_on_moved[0].set_xticks(np.arange(len(names)))
    ax_breakdown_on_moved[0].set_ylabel('Num. changes', fontsize=35)
    ax_breakdown_on_moved[0].set_xticklabels([(n) for n in names], fontsize=45)
    ax_breakdown_on_moved[0].tick_params(axis = 'y', labelsize=30)

    ax_breakdown_on_moved[1].legend(fontsize=40)
    ax_breakdown_on_moved[1].set_xticks(np.arange(len(names)))
    ax_breakdown_on_moved[1].set_ylabel("Used Objects", fontsize=35)
    ax_breakdown_on_moved[1].set_xticklabels([(n) for n in names], fontsize=45)
    ax_breakdown_on_moved[1].tick_params(axis = 'y', labelsize=30)

    ax_activity.legend(fontsize=40)
    ax_activity.set_xticks(np.arange(len(names)))
    ax_activity.set_ylabel("\% Activities", fontsize=35)
    ax_activity.set_xticklabels([(n) for n in names], fontsize=45)
    ax_activity.tick_params(axis = 'y', labelsize=30)

    ax_breakdown_on_moved_line[0].legend(fontsize=40)
    ax_breakdown_on_moved_line[0].set_xticks(np.arange(lookahead_steps))
    ax_breakdown_on_moved_line[0].set_ylabel("Correctly Moved Used Objects", fontsize=35)
    ax_breakdown_on_moved_line[0].set_ylim([0,1])
    ax_breakdown_on_moved_line[0].set_xticklabels(np.arange(lookahead_steps)+1, fontsize=45)
    ax_breakdown_on_moved_line[0].tick_params(axis = 'y', labelsize=30)

    ax_breakdown_on_moved_line[1].legend(fontsize=40)
    ax_breakdown_on_moved_line[1].set_xticks(np.arange(lookahead_steps))
    ax_breakdown_on_moved_line[1].set_ylabel("Non-misplaced Used Objects", fontsize=35)
    ax_breakdown_on_moved_line[1].set_ylim([0,1])
    ax_breakdown_on_moved_line[1].set_xticklabels(np.arange(lookahead_steps)+1, fontsize=45)
    ax_breakdown_on_moved_line[1].tick_params(axis = 'y', labelsize=30)

    # ax_recl_by_changetype.legend(fontsize=40)
    # ax_recl_by_changetype.set_xticks([1,2,3])
    # ax_recl_by_changetype.set_xticklabels(['Take out', 'Move', 'Put away'], fontsize=45)
    # ax_recl_by_changetype.tick_params(axis = 'y', labelsize=30)
    # ax_recl_by_changetype.set_ylim([ax_dest_acc_recl_norm.get_ylim()[0], ax_dest_acc_recl_norm.get_ylim()[1]+0.12])
    # ax_recl_by_changetype.set_title('recall and destination accuracy')

    # ax_recl_by_activity.legend(fontsize=10)
    # ax_recl_by_activity.set_xticklabels(activity_list, fontsize=10, rotation=90)
    # ax_recl_by_activity.tick_params(axis = 'y', labelsize=10)
    # ax_recl_by_activity.set_ylim([ax_dest_acc_recl_norm.get_ylim()[0], ax_dest_acc_recl_norm.get_ylim()[1]+0.12])
    # ax_recl_by_activity.set_title('recall & destination accuracy')


    ax_num_changes.plot(np.arange(lookahead_steps)+1, sample_data['moved']['sum'], 'o--', label='Actual\nChanges', color='black', linewidth=3)
    ax_num_changes.legend(fontsize=40)
    ax_num_changes.set_xticks(np.arange(lookahead_steps)+1)
    ax_num_changes.set_ylabel('Num. changes per step', fontsize=35)
    ax_num_changes.set_xlabel('Num. proactivity steps', fontsize=35)
    ax_num_changes.tick_params(axis = 'y', labelsize=40)
    ax_num_changes.tick_params(axis = 'x', labelsize=40)

    bplot = ax_succes_on_unmoved[0].violinplot(all_unmoved_successes) #, patch_artist=True)
    for patch, color in zip(bplot['bodies'], colors):
        patch.set_facecolor(color)
    ax_succes_on_unmoved[0].legend(fontsize=40)
    ax_succes_on_unmoved[2].set_xticks(np.arange(len(names)))
    ax_succes_on_unmoved[2].set_ylabel('Success Rate on Unused Objects', fontsize=35)
    ax_succes_on_unmoved[2].set_xticklabels([(n) for n in names], fontsize=45)
    ax_succes_on_unmoved[0].tick_params(axis = 'y', labelsize=40)
    ax_succes_on_unmoved[0].tick_params(axis = 'x', labelsize=40)
  
    # ax_succes_on_unmoved[1].legend(fontsize=40)
    ax_succes_on_unmoved[1].set_xticks(np.arange(len(names)))
    ax_succes_on_unmoved[1].set_ylabel('Success Rate on Unused Objects', fontsize=35)
    ax_succes_on_unmoved[1].set_xticklabels([(n) for n in names], fontsize=45)
    ax_succes_on_unmoved[1].tick_params(axis = 'y', labelsize=30)

    ax_succes_on_unmoved[2].legend(fontsize=40)
    ax_succes_on_unmoved[2].set_xticks(np.arange(len(names)))
    ax_succes_on_unmoved[2].set_ylabel('Success Rate on Unused Objects', fontsize=35)
    ax_succes_on_unmoved[2].set_xticklabels([(n) for n in names], fontsize=45)
    ax_succes_on_unmoved[2].tick_params(axis = 'y', labelsize=30)

    ax_roc.legend(fontsize=40)
    ax_roc.set_xlim([0,1])
    ax_roc.set_ylim([0,1])
    ax_roc.set_xlabel('Recall')
    ax_roc.set_ylabel('Precision')

    ax_roc_onlyobj.legend(fontsize=40)
    ax_roc_onlyobj.set_xlim([0,1])
    ax_roc_onlyobj.set_ylim([0,1])
    ax_roc_onlyobj.set_xlabel('Recall')
    ax_roc_onlyobj.set_ylabel('Precision')

    for fig in figs:
        fig.set_size_inches(45,23)
        fig.tight_layout()
    
    f1.set_size_inches(15,15)
    f1.tight_layout()
    f2.set_size_inches(15,15)
    f2.tight_layout()

    return figs, info



def average_stats(stats_list):
    aggregate = {}
    num_stats = len(stats_list)
    if num_stats == 0:
        return None
    # if num_stats == 1:
    #     return stats_list[0]
    for kk in stats_list[0].keys():
        if kk not in ['moved','unmoved','activity']:
            continue
        aggregate[kk] = {}
        for k in stats_list[0][kk].keys():
            aggregate[kk][k] = sum([np.array(s[kk][k]) for s in stats_list])
        aggregate[kk]['sum'] = sum([agg for agg in aggregate[kk].values()])
    return aggregate

def result_string_from_info(info):
    info_averages = {kk:{k:np.mean(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()}
    info_mins = {kk:{k:min(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()}
    info_maxs = {kk:{k:max(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()}
    info_stds = {kk:{k:np.std(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()}
    methods = list(info.keys())

    string = ''
    simple_results = [k for k in info_averages[methods[0]].keys() if '_' not in k]
    string += ('\n{} : {}'.format(' '*(40), ','.join(simple_results)))
    for m in methods:
        reses = ', '.join(['{:.4f}'.format(info_averages[m][r]) for r in simple_results])
        string += ('\n{} : {}'.format(m+' '*(40-len(m)), reses))
    
    string += '\n\n'

    string += ('\n{} : {}'.format(' '*(40), ','.join(simple_results)))
    for m in methods:
        denom1 = sum([info_averages[m][r] for r in simple_results[:3]])
        reses1 = ', '.join(['{:.2f}'.format(info_averages[m][r]/denom1*100) for r in simple_results[:3]])
        denom2 = sum([info_averages[m][r] for r in simple_results[3:]])
        reses2 = ', '.join(['{:.2f}'.format(info_averages[m][r]/denom2*100) for r in simple_results[3:]])
        string += ('\n{} : {}, {}'.format(m+' '*(40-len(m)), reses1, reses2))

    string += '\n\n\n\n'


    for result_type in ['IncDest', 'ObjOnly']:
        metrics = [result_type+'_'+x for x in ['f1_score', 'precision', 'recall']]
        string += ('\n\n\n\n---------- '+ result_type +' ----------')
        for res in metrics:
            string += ('\n----- '+ res +' -----')
            for m in methods:
                string += ('\n{} : {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(m+' '*(40-len(m)), info_mins[m][res], info_averages[m][res]-info_stds[m][res], info_averages[m][res], info_averages[m][res]+info_stds[m][res], info_maxs[m][res]))
        string += '\n\n\n\n'
        string += '\n f1_score  precision  recall'
        second_best = {'precision':0,  'recall':0,  'f1_score':0}
        our_method = [m for m in OUR_METHODS if m in methods]
        if not our_method: our_method = input(f'What out of {methods} is our method?  ')
        else: our_method = our_method[0]
        for m in methods:
            if m != our_method:
                for k in second_best.keys():
                    second_best[k] = max(second_best[k], info_averages[m][result_type+'_'+k])
            string += ('\n{} : {:.4f} & {:.4f} & {:.4f}  \\'.format(m+' '*(25-len(m)), info_averages[m][result_type+'_'+'f1_score'], info_averages[m][result_type+'_'+'precision'], info_averages[m][result_type+'_'+'recall']))
        m = 'second_best'
        string += ('\n{} : {:.4f} & {:.4f} & {:.4f} \\'.format(m+' '*(25-len(m)), second_best['f1_score'], second_best['precision'], second_best['recall']))
        perc_imp = [(info_averages[our_method][result_type+'_'+k] - second_best[k])/second_best[k] * 100 for k in ['f1_score', 'precision', 'recall']]
        m = 'perc_improvement'
        string += ('\n{} : {:2.2f} & {:2.2f} & {:2.2f} \\'.format(m+' '*(25-len(m)), perc_imp[0], perc_imp[1], perc_imp[2]))

        # string += '\n\nSignificance\n'
        # for res in metrics:
        #     string += ('\n----- '+ res +' -----')
        #     for m1 in methods:
        #         string += '\n{} : '.format(m1+' '*(40-len(m1)))
        #         for m2 in methods:
        #             value, p = ttest_ind(info[m1][res], info[m2][res], equal_var=False)
        #             string += '{:.6f} & '.format(p)
        #         string += '\\\\'

    return string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='logs/0916_1220_CoRL_final/', help='')
    
    args = parser.parse_args()


    all_training_days = [int(t) for t in os.listdir(args.path) if not t.startswith('visual')]
    all_training_days.sort(reverse=True)
    dirs = [os.path.join(args.path,str(p)) for p in all_training_days]

    datasets = os.listdir(dirs[0])

    if PLOT_CONFIDENCES:
        confidences = [0.0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.475,0.5,0.525,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
        data = [average_stats([json.load(open(os.path.join(dirs[0],dataset,'ours','evaluation.json')))[str(conf)] for dataset in datasets]) for conf in confidences]

        dir_out = os.path.join(args.path,'visuals','confidence')
        if not os.path.exists(dir_out): os.makedirs(dir_out)
        conf_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']*ceil(len(data)/10)
        conf_markers = ['.','o','x','^','p','*']*ceil(len(data)/6)
        figs, info = visualize_eval_breakdowns(data, confidences, conf_colors, conf_markers)
        for i,fig in enumerate(figs):
            fig.savefig(os.path.join(os.path.join(dir_out,filenames[i]+'.jpg')))
        with open(os.path.join(dir_out,'info.json'), 'w') as f:
            json.dump(info, f)
        raise(RuntimeError)

    all_methods = {
        'baselines' : ['FremenStateConditioned', 'LastSeenAndStaticSemantic', 'ours_'],
        'pretrain' : ['ourspt_', 'ours_','FremenStateConditioned', 'LastSeenAndStaticSemantic'],
        'ablations' : ['ours_','ours_allEdges_','ours_timeLinear_'],
        'ablations_activity' : ['original_','ours_','ours_activity25_','ours_activity50_' ,'ours_activity75_' ,'ours_activity100_'],
        'custom' : ['original_','ours_', 'ours_latent_simil_', 'ours_no_overshoot_']
    }


    # all_methods = {
    #     'baselines' : ['FremenStateConditioned', 'LastSeenAndStaticSemantic', 'ours'],
    #     'pretrain' : ['ourspt', 'ours','FremenStateConditioned', 'LastSeenAndStaticSemantic'],
    #     'ablations' : ['ours','ours_allEdges','ours_timeLinear'],
    # }


    for typ in [
        'ablations_activity', 
        # 'baselines', 
        # 'ablations', 
        # 'pretrain',
        'custom'
        ]:


        if typ.startswith('ablations'):
            training_days = [max(all_training_days)]
        else:
            training_days = all_training_days


        dirs = [os.path.join(args.path,str(p)) for p in training_days]

        methods = all_methods[typ]
        method_dirs = [[candidate for candidate in os.listdir(os.path.join(dirs[0],datasets[0])) if (candidate.startswith(method) and (len(candidate) < len(method) + 12))][0] for method in methods]
        print(method_dirs)

        combined_dir_out = os.path.join(args.path,'visuals',typ,'combined')

        master_combined_data = {m:{} for m in methods}
        master_combined_errs = {m:{} for m in methods}

        for dir in dirs:
            dir_out = os.path.join(args.path,'visuals',typ,os.path.basename(dir))
            if not os.path.exists(dir_out): os.makedirs(dir_out)
            data = []

            # if os.path.basename(dir) == '50':
            #     for dataset in datasets:
            #         for method in methods:
            #             data.append(json.load(open(os.path.join(dir,dataset,method,'evaluation.json'))))
                    
            #         dataset_dir_out = os.path.join(dir_out,dataset)
            #         if not os.path.exists(dataset_dir_out): os.makedirs(dataset_dir_out)

            #         names = [method_label(m,ablation=typ) for m in methods]
            #         markers = [method_marker(m) for m in methods]
            #         colors = [method_color(m) for m in methods]
            #         figs, info = visualize_eval_breakdowns(data, names, colors, markers, activity_difficulty=get_activity_difficulty(dataset_name='personaWithoutClothesAllObj', persona_name=dataset))
            #         for i,fig in enumerate(figs):
            #             fig.savefig(os.path.join(os.path.join(dataset_dir_out,filenames[i]+'.jpg')))
            #         with open(os.path.join(dataset_dir_out,'info.json'), 'w') as f:
            #             json.dump(info, f)
            #         with open(os.path.join(dataset_dir_out,'result.txt'), 'w') as f:
            #             f.write(result_string_from_info(info))
            #         data = []

            for method in method_dirs:
                data.append(average_stats([json.load(open(os.path.join(dir,dataset,method,'evaluation.json'))) for dataset in datasets]))

            names = [method_label(m,ablation=typ) for m in methods]
            markers = [method_marker(m) for m in methods]
            colors = [method_color(m) for m in methods]
            figs, info = visualize_eval_breakdowns(data, names, colors, markers)
            for i,fig in enumerate(figs):
                fig.savefig(os.path.join(os.path.join(dir_out,filenames[i]+'.jpg')))
            with open(os.path.join(dir_out,'info.json'), 'w') as f:
                json.dump(info, f)
            with open(os.path.join(dir_out,'result.txt'), 'w') as f:
                f.write(result_string_from_info(info))
            

            info_averages = deepcopy({kk:{k:np.mean(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()})
            info_errs = deepcopy({kk:{k:np.std(v) for k,v in vv.items() if k != 'time_only_accuracy'} for kk,vv in info.items()})

            for m in methods:
                for res in info_averages[method_label(m,ablation=typ)].keys():
                    if res not in master_combined_data[m]: master_combined_data[m][res] = []
                    if res not in master_combined_errs[m]: master_combined_errs[m][res] = []
                    master_combined_data[m][res].append(info_averages[method_label(m,ablation=typ)][res])
                    master_combined_errs[m][res].append(info_errs[method_label(m,ablation=typ)][res])

        f_f1, ax_f1 = plt.subplots()
        f_f1h, ax_f1h = plt.subplots()
        f_pr, ax_pr = plt.subplots()
        f_prh, ax_prh = plt.subplots()
        f_rc, ax_rc = plt.subplots()
        f_da, ax_da = plt.subplots()

        plt.xticks(fontsize=30)

        print(methods)

        for m in methods:
            # print(master_combined_data[m])
            ax_f1.errorbar(training_days, master_combined_data[m]['ObjOnly_f1_score'], yerr=master_combined_errs[m]['ObjOnly_f1_score'], markersize = 20, marker=method_marker(m), color=method_color(m), label=method_label(m,ablation=typ), capsize=6.0, linewidth=3)
            ax_f1h.errorbar(training_days, master_combined_data[m]['IncDest_f1_score'], yerr=master_combined_errs[m]['IncDest_f1_score'], markersize = 20, marker=method_marker(m), color=method_color(m), label=method_label(m,ablation=typ), capsize=6.0, linewidth=3)
            ax_pr.errorbar(training_days, master_combined_data[m]['ObjOnly_precision'], yerr=master_combined_errs[m]['ObjOnly_precision'], markersize = 20, marker=method_marker(m), color=method_color(m), label=method_label(m,ablation=typ), capsize=6.0, linewidth=3)
            ax_prh.errorbar(training_days, master_combined_data[m]['IncDest_precision'], yerr=master_combined_errs[m]['IncDest_precision'], markersize = 20, marker=method_marker(m), color=method_color(m), label=method_label(m,ablation=typ), capsize=6.0, linewidth=3)
            ax_rc.errorbar(training_days, master_combined_data[m]['ObjOnly_recall'], yerr=master_combined_errs[m]['ObjOnly_recall'], markersize = 20, marker=method_marker(m), color=method_color(m), label=method_label(m,ablation=typ), capsize=6.0, linewidth=3)
            ax_da.errorbar(training_days, master_combined_data[m]['IncDest_recall'], yerr=master_combined_errs[m]['IncDest_recall'], markersize = 20, marker=method_marker(m), color=method_color(m), label=method_label(m,ablation=typ), capsize=6.0, linewidth=3)

        for ax in [ax_f1, ax_f1h, ax_pr, ax_prh, ax_rc, ax_da]:
            # ax.set_xticks(training_days)
            # ax.set_xticklabels(master_combined_name)
            plt.setp(ax.get_xticklabels(), fontsize=45)
            plt.setp(ax.get_yticklabels(), fontsize=45)
            ax.set_xlabel('Num. Days of Observations', fontsize=35)
            # ax.set_ylim([0,1])
            ax.legend(fontsize=40)

        ax_f1.set_ylabel('F-1 Score', fontsize=35)
        ax_f1h.set_ylabel('F-1 Score', fontsize=35)
        ax_pr.set_ylabel('Precision', fontsize=35)
        ax_prh.set_ylabel('Precision', fontsize=35)
        ax_rc.set_ylabel('Recall', fontsize=35)
        ax_da.set_ylabel('Recall', fontsize=35)

        if os.path.exists(combined_dir_out):
            cont = input(f'Directory {combined_dir_out} exists! Do you want to overwrite it? (y/n)')
            if cont != 'y' : raise RuntimeError()
            shutil.rmtree(combined_dir_out)
        os.makedirs(combined_dir_out)
        f_f1.set_size_inches(15,10)
        f_f1.tight_layout()
        f_f1h.set_size_inches(15,10)
        f_f1h.tight_layout()
        f_pr.set_size_inches(15,10)
        f_pr.tight_layout()
        f_prh.set_size_inches(15,10)
        f_prh.tight_layout()
        f_rc.set_size_inches(15,10)
        f_rc.tight_layout()
        f_da.set_size_inches(15,10)
        f_da.tight_layout()
        f_f1.savefig(os.path.join(combined_dir_out,'ObjOnly_f1-score.jpg'))
        f_f1h.savefig(os.path.join(combined_dir_out,'IncDest_f1-score.jpg'))
        f_pr.savefig(os.path.join(combined_dir_out,'ObjOnly_precision.jpg'))
        f_prh.savefig(os.path.join(combined_dir_out,'IncDest_precision.jpg'))
        f_rc.savefig(os.path.join(combined_dir_out,'ObjOnly_recall.jpg'))
        f_da.savefig(os.path.join(combined_dir_out,'IncDest_recall.jpg'))
