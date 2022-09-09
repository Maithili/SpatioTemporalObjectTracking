from copy import deepcopy
from lib2to3.pgen2.literals import simple_escapes
from math import atan2, ceil
import os
import shutil
import argparse
import random
import json
from sqlite3 import SQLITE_CREATE_TEMP_TABLE
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from breakdown_evaluations import activity_list

MAX_LOOKAHEAD = 1000
PLOT_CONFIDENCES = False

red = np.array([164,22,35])/256
green = np.array([20,89,29])/256
neutral = np.array([245,226,200])/256*0.7

redder = red-neutral
greener = green-neutral

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
            'ours_50epochs':'Ours',
            'FremenStateConditioned':'FreMEn',
            'LastSeenAndStaticSemantic':'Stat Sem'
            }
    all_name_dict['pretrain'] = {
            'ourspt_50epochs':'Pre-Train',
            'ourspt0_50epochs':'PreTr0only',
            'ours_50epochs':'Scratch',
            }
    all_name_dict['ablations'] = {
            'ours_50epochs':'Ours',
            'ours_activity25_50epochs':'Activity 25',
            'ours_activity50_50epochs':'Activity 50',
            'ours_activity75_50epochs':'Activity 75',
            'ours_activity100_50epochs':'Activity 100',
            'ours_allEdges_50epochs':'w/o Attn',
            'ours_timeLinear_50epochs':'w/o Time'
            }
    name_dict = all_name_dict[ablation]
    return name_dict[name]

filenames = ['recall_accuracy','precision','ObjOnly_F1','IncDest_PrecRec', 'ObjOnly_PrecRec', 'recall_accuracy_norm', 'precision_norm', 'time_only_prediction', 'destination_accuracy', 'num_changes','destination_accuracy_line', 'optimistic', 'recall_by_type', 'recall_by_activity', 'recall_line', 'TrendOnlyObjectss', 'TrendWithDestination', 'ObjOnly_PrecRecAgg', 'IncDest_PrecRecAgg', 'IncDest_F1', 'Split', 'Error_Analysis', 'Error_Analysis_Too', 'Oracle_Analysis', 'Unmoved', 'Moved']


def visualize_eval_breakdowns(data, names, colors, markers, activity_difficulty=None):
    # fig, ax = plt.subplots(2,3)
    f1, ax_comp_t_tl = plt.subplots()
    f2, ax_prec = plt.subplots()
    f3, ax_f1 = plt.subplots()
    f4, ax_comp_tl_prec = plt.subplots()
    f5, ax_comp_t_prec = plt.subplots()
    f6, ax_dest_acc_recl_norm = plt.subplots()
    f7, ax_prec_norm = plt.subplots()
    f8, ax_time_only = plt.subplots()
    f9, ax_dest_acc_norm = plt.subplots()
    f10, ax_num_changes = plt.subplots()
    f11, ax_dest_acc_norm2 = plt.subplots()
    f12, ax_optimistic_da_recl = plt.subplots()
    f13, ax_recl_by_changetype = plt.subplots()
    f14, ax_recl_by_activity = plt.subplots()
    f15, ax_recl_norm2 = plt.subplots()
    f16, ax_prec_norm2 = plt.subplots(1,2)
    f17, ax_da_norm2 = plt.subplots(1,2)
    f18, ax_avg_prec_rec = plt.subplots()
    f19, ax_avg_prec_rec_hard = plt.subplots()
    f20, ax_f1_harder = plt.subplots()
    f21, ax_split = plt.subplots()
    f22, ax_error_analysis = plt.subplots()
    f23, ax_error_analysis2 = plt.subplots()
    f24, ax_oracle_analysis = plt.subplots()
    f25, ax_succes_on_unmoved = plt.subplots(1,3)
    f26, ax_breakdown_on_moved = plt.subplots(1,2)
    figs =[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26]

    for ax in [ax_comp_tl_prec, ax_comp_t_prec, ax_avg_prec_rec, ax_avg_prec_rec_hard]:
        for f in [0.2, 0.4, 0.6]:
            pinv = np.linspace(1,2/f-1, 100)
            rinv = 2/f - pinv
            ax.plot(1/pinv, 1/rinv, color='grey', linewidth=(1-f)*2)
            p1, p2 = 1/(2/f-1/0.75), 1/(2/f-1/0.95)
            ax.text(0.68,p2,f'F-1 score = {f}', rotation=np.rad2deg(atan2(p2-p1, 0.2)), fontsize=30, backgroundcolor=[1,1,1,0.5])

    lookahead_steps = None

    info = {}

    print(names)
    offsets = np.linspace(-0.45,0.45,len(data[0]['precision_breakdown'])+1)
    offsets = (offsets[1:]+offsets[:-1])/2
    width = offsets[1] - offsets[0] - 0.01

    all_precisions = []
    all_precisions_harder = []
    all_recalls = []
    all_destaccs = []

    for sample_num, sample_data in enumerate(data):
        lookahead_steps = len(sample_data['precision_breakdown'])
        quality_steps = len(sample_data['precision_breakdown'])

        # breakdown_steps = len(sample_data['breakdown']['correct'])
        # success_on_unmoved = np.array(sample_data['breakdown']['tn'])/(np.array(sample_data['breakdown']['tn']) + np.array(sample_data['breakdown']['fp']))
        # ax_succes_on_unmoved.plot(np.arange(breakdown_steps), success_on_unmoved, markersize = 20, marker=(markers[sample_num]), label=(names[sample_num]), color=colors[sample_num], linewidth=3)
        # for step in range(breakdown_steps):
        #     if sample_num == 0 and step == 0:
        #         labels_brkdwn = ['Correct','Missed','Wrong']
        #     ax_breakdown_on_moved.bar(sample_num + offsets[step], sample_data['breakdown']['correct'][step], color=green, width=width, label=labels_brkdwn[0])
        #     ax_breakdown_on_moved.bar(sample_num + offsets[step], sample_data['breakdown']['missed'][step], bottom=sample_data['breakdown']['correct'][step], color=neutral, width=width, label=labels_brkdwn[1])
        #     ax_breakdown_on_moved.bar(sample_num + offsets[step], sample_data['breakdown']['wrong'][step], bottom=sample_data['breakdown']['missed'][step]+sample_data['breakdown']['correct'][step], color=red, width=width, label=labels_brkdwn[2])
        #     labels_brkdwn = [None, None, None]

        sample_data['tn'] = [87*num_routines - sample_data['precision_breakdown'][step][0] - sample_data['precision_breakdown'][step][1] - sample_data['completeness_breakdown']['by_lookahead'][step][2] for step in range(lookahead_steps)]
        success_on_unmoved = [sample_data['tn'][step]/ (sample_data['tn'][step] + sample_data['precision_breakdown'][step][1]) for step in range(lookahead_steps)]
        ax_succes_on_unmoved[0].plot(np.arange(lookahead_steps), success_on_unmoved, markersize = 20, marker=(markers[sample_num]), label=(names[sample_num]), color=colors[sample_num], linewidth=3)
        ax_succes_on_unmoved[1].bar([sample_num + o for o in offsets], sample_data['tn'], color=green, width=width)
        ax_succes_on_unmoved[1].bar([sample_num + o for o in offsets], [sample_data['precision_breakdown'][step][1] for step in range(lookahead_steps)], bottom=sample_data['tn'], color=red, width=width)
        overall_success = sum(sample_data['tn'])/ (sum(sample_data['tn']) + sum([sample_data['precision_breakdown'][step][1] for step in range(lookahead_steps)]))
        overall_success = int(round(overall_success*1000))/1000
        ps = ax_succes_on_unmoved[2].bar(sample_num, 1-overall_success, bottom=overall_success, color=red)
        pf = ax_succes_on_unmoved[2].bar(sample_num, overall_success, color=green)
        ax_succes_on_unmoved[2].bar_label(ps, label_type='center', fontsize=30)
        ax_succes_on_unmoved[2].bar_label(pf, label_type='center', fontsize=30)
        if sample_num == 0:
            labels_brkdwn = ['Correct','Missed','Wrong']
        correct = sum([a for a in [sample_data['completeness_breakdown']['by_lookahead'][step][0] for step in range(lookahead_steps)]])/num_routines
        wrong = sum([a for a in [sample_data['completeness_breakdown']['by_lookahead'][step][1] for step in range(lookahead_steps)]])/num_routines
        missed = sum([a for a in [sample_data['completeness_breakdown']['by_lookahead'][step][2] for step in range(lookahead_steps)]])/num_routines
        total = correct+wrong+missed
        correct = int(round(correct/total*1000))/1000
        wrong = int(round(wrong/total*1000))/1000
        missed = int(round(missed/total*1000))/1000
        pc = ax_breakdown_on_moved[1].bar(sample_num, correct, color=green, label=labels_brkdwn[0])
        pm = ax_breakdown_on_moved[1].bar(sample_num, missed, bottom=correct, color=neutral, label=labels_brkdwn[1])
        pw = ax_breakdown_on_moved[1].bar(sample_num, wrong, bottom=missed+correct, color=red, label=labels_brkdwn[2])
        ax_breakdown_on_moved[1].bar_label(pc, label_type='center', fontsize=30)
        ax_breakdown_on_moved[1].bar_label(pm, label_type='center', fontsize=30)
        ax_breakdown_on_moved[1].bar_label(pw, label_type='center', fontsize=30)
        for step in range(lookahead_steps):
            correct, wrong, missed = [a/num_routines for a in sample_data['completeness_breakdown']['by_lookahead'][step]]
            ax_breakdown_on_moved[0].bar(sample_num + offsets[step], correct, color=green, width=width, label=labels_brkdwn[0])
            ax_breakdown_on_moved[0].bar(sample_num + offsets[step], missed, bottom=correct, color=neutral, width=width, label=labels_brkdwn[1])
            ax_breakdown_on_moved[0].bar(sample_num + offsets[step], wrong, bottom=missed+correct, color=red, width=width, label=labels_brkdwn[2])
            labels_brkdwn = [None, None, None]

        for step in range(quality_steps-1, -1, -1):
            if sample_num == 0 and step == 0:
                ax_prec.bar(sample_num + offsets[step], sum(sample_data['precision_breakdown'][step])/num_routines, color=red-redder*0.3, width=width, label='False Positives')
                ax_prec.bar(sample_num + offsets[step], sample_data['precision_breakdown'][step][0]/num_routines, color=green, width=width, label='Correct Time')
                ax_prec_norm.bar(sample_num + offsets[step], sample_data['precision_breakdown'][step][0]/(sum(sample_data['precision_breakdown'][step])+1e-8), color=green-greener*0.2, width=width, label='Precision')
            else:
                ax_prec.bar(sample_num + offsets[step], sum(sample_data['precision_breakdown'][step])/num_routines, color=red-redder*0.3, width=width)
                ax_prec.bar(sample_num + offsets[step], sample_data['precision_breakdown'][step][0]/num_routines, color=green, width=width)
                ax_prec_norm.bar(sample_num + offsets[step], sample_data['precision_breakdown'][step][0]/(sum(sample_data['precision_breakdown'][step])+1e-8), color=green-greener*0.2, width=width)
        ax_num_changes.plot(np.arange(lookahead_steps)+1, [sum(qb)/10 for qb in sample_data['precision_breakdown']], markersize = 20, marker=(markers[sample_num]), label=(names[sample_num]), color=colors[sample_num], linewidth=3)
        precisions = [qb[0]/(sum(qb)+1e-8) for qb in sample_data['precision_breakdown']]
        all_precisions.append(precisions)
        comp_steps = len(sample_data['completeness_breakdown']['by_lookahead'])
        assert quality_steps==comp_steps

        # if without_types:
        for step in range(comp_steps-1, -1, -1):
            if sample_num == 0 and step == 0:
                ax_comp_t_tl.bar(sample_num + offsets[step], sum(sample_data['completeness_breakdown']['by_lookahead'][step])/num_routines, color=red-redder*0.3, width=width, label='Wrong Time')
                ax_comp_t_tl.bar(sample_num + offsets[step], (sample_data['completeness_breakdown']['by_lookahead'][step][0]+sample_data['completeness_breakdown']['by_lookahead'][step][1])/num_routines, color=green-greener*0.3, width=width, label='Correct Time')
                ax_comp_t_tl.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/num_routines, color=green, width=width, label='Correct Time \n+ Destination')
                ax_dest_acc_recl_norm.bar(sample_num + offsets[step], (sample_data['completeness_breakdown']['by_lookahead'][step][0]+sample_data['completeness_breakdown']['by_lookahead'][step][1])/sum(sample_data['completeness_breakdown']['by_lookahead'][step])/num_routines, color=green-greener*0.5, width=width, label='Recall')
                ax_dest_acc_recl_norm.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/sum(sample_data['completeness_breakdown']['by_lookahead'][step])/num_routines, color=green-greener*0.0, width=width, label='Destination\nAccuracy')
                ax_dest_acc_norm.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/sum(sample_data['completeness_breakdown']['by_lookahead'][step])/num_routines, color=green-greener*0.2, width=width, label='Destination\nAccuracy')
        
            else:
                ax_comp_t_tl.bar(sample_num + offsets[step], sum(sample_data['completeness_breakdown']['by_lookahead'][step])/num_routines, color=red-redder*0.3, width=width)
                ax_comp_t_tl.bar(sample_num + offsets[step], (sample_data['completeness_breakdown']['by_lookahead'][step][0]+sample_data['completeness_breakdown']['by_lookahead'][step][1])/num_routines, color=green-greener*0.3, width=width)
                ax_comp_t_tl.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/num_routines, color=green, width=width)
                ax_dest_acc_recl_norm.bar(sample_num + offsets[step], (sample_data['completeness_breakdown']['by_lookahead'][step][0]+sample_data['completeness_breakdown']['by_lookahead'][step][1])/sum(sample_data['completeness_breakdown']['by_lookahead'][step])/num_routines, color=green-greener*0.5, width=width)
                ax_dest_acc_recl_norm.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/sum(sample_data['completeness_breakdown']['by_lookahead'][step])/num_routines, color=green-greener*0.0, width=width)
                ax_dest_acc_norm.bar(sample_num + offsets[step], sample_data['completeness_breakdown']['by_lookahead'][step][0]/sum(sample_data['completeness_breakdown']['by_lookahead'][step])/num_routines, color=green-greener*0.2, width=width)
        
        if 'by_change_type' in sample_data['completeness_breakdown'].keys():
            ax_recl_by_changetype.plot([1,2,3], [(sample_data['completeness_breakdown']['by_change_type'][ci][0]+sample_data['completeness_breakdown']['by_change_type'][ci][1])/sum(sample_data['completeness_breakdown']['by_change_type'][ci]) for ci in range(3)], linewidth=3, linestyle='dashed', marker=markers[sample_num], color=colors[sample_num], markersize = 20)
            ax_recl_by_changetype.plot([1,2,3], [(sample_data['completeness_breakdown']['by_change_type'][ci][0])/sum(sample_data['completeness_breakdown']['by_change_type'][ci]) for ci in range(3)], linewidth=3, label=(names[sample_num]), marker=markers[sample_num], color=colors[sample_num], markersize = 20)

        if 'by_activity' in sample_data['completeness_breakdown'].keys():
            if activity_difficulty is None:
                ax_recl_by_activity.plot(np.arange(len(activity_list)), [(sample_data['completeness_breakdown']['by_activity'][ci][0]+sample_data['completeness_breakdown']['by_activity'][ci][1])/(sum(sample_data['completeness_breakdown']['by_activity'][ci])+1e-8) for ci in range(len(activity_list))], linewidth=3, label=(names[sample_num]), marker=markers[sample_num], color=colors[sample_num], markersize = 20)
                ax_recl_by_activity.plot(np.arange(len(activity_list)), [(sample_data['completeness_breakdown']['by_activity'][ci][0])/(sum(sample_data['completeness_breakdown']['by_activity'][ci])+1e-8) for ci in range(len(activity_list))], linewidth=3, marker=markers[sample_num], color=colors[sample_num], markersize = 20, linestyle='dashed')
                ax_recl_by_activity.set_xticks(np.arange(len(activity_list)))
            else:
                # ax_recl_by_activity.scatter([activity_difficulty[a] for a in activity_list], [(sample_data['completeness_breakdown']['by_activity'][ci][0]+sample_data['completeness_breakdown']['by_activity'][ci][1])/(sum(sample_data['completeness_breakdown']['by_activity'][ci])+1e-8) for ci in range(len(activity_list))], label=(names[sample_num]), marker=markers[sample_num], color=colors[sample_num], markersize = 20)
                ax_recl_by_activity.scatter([activity_difficulty[a] for a in activity_list], [(sample_data['completeness_breakdown']['by_activity'][ci][0])/(sum(sample_data['completeness_breakdown']['by_activity'][ci])+1e-8) for ci in range(len(activity_list))], marker=markers[sample_num], color=colors[sample_num]) #, markersize = 20)
                ax_recl_by_activity.set_xticks([activity_difficulty[a] for a in activity_list])


        ax_dest_acc_norm2.plot(np.arange(lookahead_steps)+1, [cb[0]/(sum(cb)+1e-8) for cb in sample_data['completeness_breakdown']['by_lookahead']], markersize = 20, marker=markers[sample_num], color=colors[sample_num], linewidth=3, label=(names[sample_num]))
        completeness_tl = [cb[0]/(sum(cb)+1e-8) for cb in sample_data['completeness_breakdown']['by_lookahead']]
        completeness_t = [(cb[0]+cb[1])/(sum(cb)+1e-8) for cb in sample_data['completeness_breakdown']['by_lookahead']]
        all_destaccs.append(completeness_tl)
        all_recalls.append(completeness_t)

        precision_harder = [cb[0]/(sum(qb)+1e-8) for cb,qb in zip(sample_data['completeness_breakdown']['by_lookahead'],sample_data['precision_breakdown'])]
        all_precisions_harder.append(precision_harder)

        if 'optimistic_completeness_breakdown' in sample_data.keys():
            ax_optimistic_da_recl.plot(np.arange(lookahead_steps)+1, [cb[0]/(sum(cb)+1e-8) for cb in sample_data['optimistic_completeness_breakdown']['by_lookahead']], markersize = 20, marker=markers[sample_num], color=colors[sample_num], linewidth=3, label=(names[sample_num]), linestyle='dashed')
            ax_optimistic_da_recl.plot(np.arange(lookahead_steps)+1, [(cb[0]+cb[1])/(sum(cb)+1e-8) for cb in sample_data['optimistic_completeness_breakdown']['by_lookahead']], markersize = 20, marker=markers[sample_num], color=colors[sample_num], linewidth=3)

        if sample_num == len(data)-1 :
            ax_num_changes.plot(np.arange(lookahead_steps)+1, [sum(cb)/10 for cb in sample_data['completeness_breakdown']['by_lookahead']], 'o--', label='Actual\nChanges', color='black', linewidth=3)

        f1 = [2*p*r/(p+r+1e-8) for p,r in zip(precisions, completeness_t)]
        ax_f1.plot(np.arange(lookahead_steps)+1, f1, markersize = 20, marker=markers[sample_num], color=colors[sample_num], linewidth=3, label=(names[sample_num]))

        f1_harder = [2*p*r/(p+r+1e-8) for p,r in zip(precision_harder, completeness_tl)]
        ax_f1_harder.plot(np.arange(lookahead_steps)+1, f1_harder, markersize = 20, marker=markers[sample_num], color=colors[sample_num], linewidth=3, label=(names[sample_num]))

        left = 0
        p1 = ax_split.barh(sample_num, sample_data['completeness_breakdown']['by_lookahead'][-1][0], left=left, label='correct', color='tab:green')
        left += sample_data['completeness_breakdown']['by_lookahead'][-1][0]
        p2 = ax_split.barh(sample_num, sample_data['completeness_breakdown']['by_lookahead'][-1][1], left=left, label='wrong', color='tab:red')
        left += sample_data['completeness_breakdown']['by_lookahead'][-1][1]
        p4 = ax_split.barh(sample_num, sample_data['completeness_breakdown']['by_lookahead'][-1][2], left=left, label='missed', color='tab:grey')
        left += sample_data['completeness_breakdown']['by_lookahead'][-1][2]
        p3 = ax_split.barh(sample_num, -sample_data['precision_breakdown'][-1][1], left=0, label='extra', color='tab:orange')
        # left += sample_data['precision_breakdown'][-1][1]
        
        ax_split.bar_label(p1, label_type='center', fontsize=40, fmt="%.2f")
        ax_split.bar_label(p2, label_type='center', fontsize=40, fmt="%.2f")
        ax_split.bar_label(p3, label_type='center', fontsize=40, fmt="%.2f")
        ax_split.bar_label(p4, label_type='center', fontsize=40, fmt="%.2f")

        precision_harder_without_fp = [cb[0]/((qb[0])+1e-8) for cb,qb in zip(sample_data['completeness_breakdown']['by_lookahead'],sample_data['precision_breakdown'])]
        f1_harder_without_fp = [2*p*r/(p+r+1e-8) for p,r in zip(precision_harder_without_fp, completeness_tl)]
        f1_without_fp = [2*r/(1+r+1e-8) for r in completeness_t]
        f1_without_missed = [2*p/(1+p+1e-8) for p in precisions]


        if sample_num==0:
            labels = ['Missed Oracle', 'False Object Oracle', 'Destination Oracle', 'Original']
        ax_error_analysis.bar(sample_num, 1, color='tab:grey', label=labels[0])
        ax_error_analysis.bar(sample_num, np.mean(f1_without_fp), color='tab:cyan', label=labels[2])
        ax_error_analysis.bar(sample_num, np.mean(f1_harder_without_fp), color='tab:blue', label=labels[1])
        ax_error_analysis.bar(sample_num, np.mean(f1_harder), color='tab:green', label=labels[3])
        
        ax_error_analysis2.bar(sample_num, 1, color='tab:grey', label=labels[0])
        # f1_without_missed = [2*p/(1+p+1) for p in precisions]
        # ax_error_analysis.bar(sample_nu, np.mean(f1_without_missed),  colors='tab:blue')        
        ax_error_analysis2.bar(sample_num, np.mean(f1_without_fp), color='tab:blue', label=labels[1])
        ax_error_analysis2.bar(sample_num, np.mean(f1), color='tab:cyan', label=labels[2])
        ax_error_analysis2.bar(sample_num, np.mean(f1_harder), color='tab:green', label=labels[3])
        labels = [None, None, None, None]

        if 'with_oracle' in sample_data.keys():
            if sample_num==0:
                labels = ['With Oracle', 'Without Oracle']
            ax_oracle_analysis.bar(sample_num, sample_data['with_oracle']['correct']/(sample_data['with_oracle']['correct']+sample_data['with_oracle']['wrong']), color='tab:green', label=label[0])
            ax_oracle_analysis.bar(sample_num, sample_data['without_oracle']['correct']/(sample_data['without_oracle']['correct']+sample_data['without_oracle']['wrong']), color='tab:blue', label=label[1])

        alphas = np.linspace(1,0.2,quality_steps)
        for i in range(quality_steps):
            label = (names[sample_num]) if i==0 else None
            ax_comp_tl_prec.plot(completeness_tl[i], precision_harder[i], markersize = 20, marker=markers[sample_num],markeredgewidth = 5, label=label, color=colors[sample_num], alpha=alphas[i])
            # ax_comp_tl_prec.plot(completeness_t[i], precisions[i], markersize = 20, marker='.', markeredgewidth = 5, label=label, color=colors[sample_num], alpha=alphas[i])
            ax_comp_t_prec.plot(completeness_t[i], precisions[i], markersize = 20, marker=markers[sample_num], markeredgewidth = 5, label=label, color=colors[sample_num], alpha=alphas[i])
        ax_avg_prec_rec.errorbar(np.mean(completeness_t), np.mean(precisions), xerr=np.std(completeness_t), yerr=np.std(precisions), markersize = 20, marker=markers[sample_num], markeredgewidth = 5, label = (names[sample_num]), color=colors[sample_num])
        ax_avg_prec_rec_hard.errorbar(np.mean(completeness_tl), np.mean(precision_harder), xerr=np.std(completeness_tl), yerr=np.std(precision_harder), markersize = 20, marker=markers[sample_num], markeredgewidth = 5, label = (names[sample_num]), color=colors[sample_num])

        info[names[sample_num]] = {}
        info[names[sample_num]]['ObjOnly_precision'] = precisions
        info[names[sample_num]]['ObjOnly_recall'] = completeness_t
        info[names[sample_num]]['ObjOnly_f1_score'] = f1
        info[names[sample_num]]['IncDest_precision'] = precision_harder
        info[names[sample_num]]['IncDest_recall'] = completeness_tl
        info[names[sample_num]]['IncDest_f1_score'] = f1_harder
        info[names[sample_num]]['Correct'] = [sample_data['completeness_breakdown']['by_lookahead'][s][0]/num_routines for s in range(lookahead_steps)]
        info[names[sample_num]]['Wrong'] = [sample_data['completeness_breakdown']['by_lookahead'][s][1]/num_routines for s in range(lookahead_steps)]
        info[names[sample_num]]['Missed'] = [sample_data['completeness_breakdown']['by_lookahead'][s][2]/num_routines for s in range(lookahead_steps)]
        info[names[sample_num]]['FP'] = [sample_data['precision_breakdown'][s][1]/num_routines for s in range(lookahead_steps)]
        info[names[sample_num]]['TN'] = [sample_data['tn'][s]/num_routines for s in range(lookahead_steps)]

 

    for ls in range(lookahead_steps):
        print(ls+1, ' steps...')
        print([precs[ls] for precs in all_precisions])
        ax_prec_norm2[0].plot(names, [recs[ls] for recs in all_recalls], label=f'{ls+1}-steps', markersize = 20, marker='.', markeredgewidth = 5)
        ax_prec_norm2[1].plot(names, [precs[ls] for precs in all_precisions], label=f'{ls+1}-steps', markersize = 20, marker='.', markeredgewidth = 5)
        ax_recl_norm2.plot(names, [recs[ls] for recs in all_recalls], label=f'{ls+1}-steps')
        ax_da_norm2[0].plot(names, [das[ls] for das in all_destaccs], label=f'{ls+1}-steps', markersize = 20, marker='.', markeredgewidth = 5)
        ax_da_norm2[1].plot(names, [precs[ls] for precs in all_precisions_harder], label=f'{ls+1}-steps', markersize = 20, marker='.', markeredgewidth = 5)

    ax_prec_norm2[0].set_xlabel('Confidence Threshold', fontsize=45)
    ax_prec_norm2[1].set_xlabel('Confidence Threshold', fontsize=45)
    plt.setp(ax_prec_norm2[0].get_xticklabels(), fontsize=45)
    plt.setp(ax_prec_norm2[1].get_xticklabels(), fontsize=45)
    plt.setp(ax_prec_norm2[0].get_yticklabels(), fontsize=45)
    plt.setp(ax_prec_norm2[1].get_yticklabels(), fontsize=45)
    ax_prec_norm2[0].set_ylabel('Recall', fontsize=45)
    ax_prec_norm2[1].set_ylabel('Precision', fontsize=45)

    ax_da_norm2[0].set_xlabel('Confidence Threshold', fontsize=45)
    ax_da_norm2[1].set_xlabel('Confidence Threshold', fontsize=45)
    plt.setp(ax_da_norm2[0].get_xticklabels(), fontsize=45)
    plt.setp(ax_da_norm2[1].get_xticklabels(), fontsize=45)
    plt.setp(ax_da_norm2[0].get_yticklabels(), fontsize=45)
    plt.setp(ax_da_norm2[1].get_yticklabels(), fontsize=45)
    ax_da_norm2[0].set_ylabel('Recall', fontsize=45)
    ax_da_norm2[1].set_ylabel('Precision', fontsize=45)

    ax_prec_norm2[0].legend(fontsize=40, loc='upper right')
    ax_prec_norm2[1].legend(fontsize=40)
    ax_recl_norm2.legend(fontsize=40, loc='upper right')
    ax_da_norm2[0].legend(fontsize=40, loc='upper right')
    ax_da_norm2[1].legend(fontsize=40)

    ax_comp_t_prec.legend(fontsize=40, loc='upper right')
    ax_comp_t_prec.set_xlabel('Recall', fontsize=45)
    ax_comp_t_prec.set_ylabel('Precision', fontsize=45)
    ax_comp_t_prec.tick_params(axis = 'y', labelsize=30)
    ax_comp_t_prec.tick_params(axis = 'x', labelsize=30)
    ax_comp_t_prec.set_ylim([0,1])

    ax_comp_tl_prec.legend(fontsize=40, loc='upper right')
    ax_comp_tl_prec.set_xlabel('Recall', fontsize=45)
    ax_comp_tl_prec.set_ylabel('Precision', fontsize=45)
    ax_comp_tl_prec.tick_params(axis = 'y', labelsize=30)
    ax_comp_tl_prec.tick_params(axis = 'x', labelsize=30)
    ax_comp_tl_prec.set_ylim([0,1])

    ax_avg_prec_rec.legend(fontsize=40, loc='upper right')
    ax_avg_prec_rec.set_xlabel('Recall', fontsize=45)
    ax_avg_prec_rec.set_ylabel('Precision', fontsize=45)
    ax_avg_prec_rec.tick_params(axis = 'y', labelsize=30)
    ax_avg_prec_rec.tick_params(axis = 'x', labelsize=30)
    ax_avg_prec_rec.set_ylim([0,1])


    ax_avg_prec_rec_hard.legend(fontsize=40, loc='upper right')
    ax_avg_prec_rec_hard.set_xlabel('Recall', fontsize=45)
    ax_avg_prec_rec_hard.set_ylabel('Precision', fontsize=45)
    ax_avg_prec_rec_hard.tick_params(axis = 'y', labelsize=30)
    ax_avg_prec_rec_hard.tick_params(axis = 'x', labelsize=30)
    ax_avg_prec_rec_hard.set_ylim([0,1])
    
    ax_comp_t_tl.legend(fontsize=40)
    ax_comp_t_tl.set_xticks(np.arange(len(names)))
    ax_comp_t_tl.set_xticklabels([(n) for n in names], fontsize=45)
    ax_comp_t_tl.set_ylabel('Num. changes per step', fontsize=35)
    ax_comp_t_tl.tick_params(axis = 'y', labelsize=30)
    # ax_comp_t_tl.set_title('Fraction of changes correctly predicted', fontsize=30)
    
    ax_prec.legend(fontsize=40)
    ax_prec.set_xticks(np.arange(len(names)))
    ax_prec.set_ylabel('Num. changes per step', fontsize=35)
    ax_prec.set_xticklabels([(n) for n in names], fontsize=45)
    ax_prec.tick_params(axis = 'y', labelsize=30)
    # ax_prec.set_title('Correct fraction of predictions', fontsize=30)
    # ax_prec.set_ylim([0,10])

    ax_breakdown_on_moved[0].legend(fontsize=40)
    ax_breakdown_on_moved[0].set_xticks(np.arange(len(names)))
    ax_breakdown_on_moved[0].set_ylabel('Num. changes', fontsize=35)
    ax_breakdown_on_moved[0].set_xticklabels([(n) for n in names], fontsize=45)
    ax_breakdown_on_moved[0].tick_params(axis = 'y', labelsize=30)

    # ax_breakdown_on_moved[1].legend(fontsize=40)
    ax_breakdown_on_moved[1].set_xticks(np.arange(len(names)))
    ax_breakdown_on_moved[1].set_ylabel("Used Objects", fontsize=35)
    ax_breakdown_on_moved[1].set_xticklabels([(n) for n in names], fontsize=45)
    ax_breakdown_on_moved[1].tick_params(axis = 'y', labelsize=30)

    ax_prec_norm.legend(fontsize=40)
    ax_prec_norm.set_xticks(np.arange(len(names)))
    ax_prec_norm.set_xticklabels([(n) for n in names], fontsize=45)
    ax_prec_norm.tick_params(axis = 'y', labelsize=30)
    # ax_prec_norm.set_title('Precision', fontsize=30)
    ax_prec_norm.set_ylim([0,1])

    ax_dest_acc_recl_norm.legend(fontsize=40)
    ax_dest_acc_recl_norm.set_xticks(np.arange(len(names)))
    ax_dest_acc_recl_norm.set_xticklabels([(n) for n in names], fontsize=45)
    ax_dest_acc_recl_norm.tick_params(axis = 'y', labelsize=30)
    ax_dest_acc_recl_norm.set_ylim([ax_dest_acc_recl_norm.get_ylim()[0], ax_dest_acc_recl_norm.get_ylim()[1]+0.12])
    # ax_dest_acc_recl_norm.set_title('Recall & Destination Accuracy', fontsize=30)
    ax_dest_acc_recl_norm.set_ylim([0,1])

    ax_dest_acc_norm.legend(fontsize=40)
    ax_dest_acc_norm.set_xticks(np.arange(len(names)))
    ax_dest_acc_norm.set_xticklabels([(n) for n in names], fontsize=45)
    ax_dest_acc_norm.tick_params(axis = 'y', labelsize=30)
    # ax_dest_acc_norm.set_ylim([ax_dest_acc_norm.get_ylim()[0], ax_dest_acc_norm.get_ylim()[1]+0.12])

    ax_dest_acc_recl_norm.legend(fontsize=40)
    ax_dest_acc_recl_norm.set_xticks(np.arange(len(names)))
    ax_dest_acc_recl_norm.set_xticklabels([(n) for n in names], fontsize=45)
    ax_dest_acc_recl_norm.tick_params(axis = 'y', labelsize=30)
    ax_dest_acc_recl_norm.set_ylim([ax_dest_acc_recl_norm.get_ylim()[0], ax_dest_acc_recl_norm.get_ylim()[1]+0.12])
    # ax_dest_acc_recl_norm.set_title('Recall & Destination Accuracy', fontsize=30)


    ax_recl_by_changetype.legend(fontsize=40)
    ax_recl_by_changetype.set_xticks([1,2,3])
    ax_recl_by_changetype.set_xticklabels(['Take out', 'Move', 'Put away'], fontsize=45)
    ax_recl_by_changetype.tick_params(axis = 'y', labelsize=30)
    ax_recl_by_changetype.set_ylim([ax_dest_acc_recl_norm.get_ylim()[0], ax_dest_acc_recl_norm.get_ylim()[1]+0.12])
    ax_recl_by_changetype.set_title('recall and destination accuracy')

    ax_recl_by_activity.legend(fontsize=10)
    ax_recl_by_activity.set_xticklabels(activity_list, fontsize=10, rotation=90)
    ax_recl_by_activity.tick_params(axis = 'y', labelsize=10)
    ax_recl_by_activity.set_ylim([ax_dest_acc_recl_norm.get_ylim()[0], ax_dest_acc_recl_norm.get_ylim()[1]+0.12])
    ax_recl_by_activity.set_title('recall & destination accuracy')

    ax_time_only.legend(fontsize=40)
    ax_time_only.set_xticks(np.arange(len(names)))
    ax_time_only.set_ylabel('Num. changes per step', fontsize=35)
    ax_time_only.set_xticklabels([(n) for n in names], fontsize=45)
    ax_time_only.tick_params(axis = 'y', labelsize=30)
    # ax_time_only.set_title('Time-based predictions', fontsize=30)

    ax_num_changes.legend(fontsize=40)
    ax_num_changes.set_xticks(np.arange(lookahead_steps)+1)
    ax_num_changes.set_ylabel('Num. changes per step', fontsize=35)
    ax_num_changes.set_xlabel('Num. proactivity steps', fontsize=35)
    ax_num_changes.tick_params(axis = 'y', labelsize=40)
    ax_num_changes.tick_params(axis = 'x', labelsize=40)

    ax_succes_on_unmoved[0].legend(fontsize=40)
    ax_succes_on_unmoved[0].set_xticks(np.arange(lookahead_steps)+1)
    ax_succes_on_unmoved[0].set_ylabel('Success Rate on Unused Objects', fontsize=35)
    ax_succes_on_unmoved[0].set_xlabel('Num. proactivity steps', fontsize=35)
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


    ax_dest_acc_norm2.legend(fontsize=40)
    ax_dest_acc_norm2.set_xticks(np.arange(lookahead_steps)+1)
    ax_dest_acc_norm2.set_ylabel('Destination Accuracy', fontsize=35)
    ax_dest_acc_norm2.set_xlabel('Num. proactivity steps', fontsize=35)
    ax_dest_acc_norm2.tick_params(axis = 'y', labelsize=40)
    ax_dest_acc_norm2.tick_params(axis = 'x', labelsize=40)

    ax_f1_harder.legend(fontsize=40)
    ax_f1_harder.set_xticks(np.arange(lookahead_steps)+1)
    ax_f1_harder.set_ylabel('F-1 Score', fontsize=35)
    ax_f1_harder.set_xlabel('Num. proactivity steps', fontsize=35)
    ax_f1_harder.tick_params(axis = 'y', labelsize=40)
    ax_f1_harder.tick_params(axis = 'x', labelsize=40)

    ax_f1.legend(fontsize=40)
    ax_f1.set_xticks(np.arange(lookahead_steps)+1)
    ax_f1.set_ylabel('F-1 Score', fontsize=35)
    ax_f1.set_xlabel('Num. proactivity steps', fontsize=35)
    ax_f1.tick_params(axis = 'y', labelsize=40)
    ax_f1.tick_params(axis = 'x', labelsize=40)

    ax_optimistic_da_recl.legend(fontsize=40)
    ax_optimistic_da_recl.set_xticks(np.arange(lookahead_steps)+1)
    ax_optimistic_da_recl.set_ylabel('Optimistic Recall', fontsize=35)
    ax_optimistic_da_recl.set_ylim([0,1])
    ax_optimistic_da_recl.set_xlabel('Num. proactivity steps', fontsize=35)
    ax_optimistic_da_recl.tick_params(axis = 'y', labelsize=40)
    ax_optimistic_da_recl.tick_params(axis = 'x', labelsize=40)

    ax_error_analysis.legend(fontsize=40)
    ax_error_analysis.set_xticks(np.arange(len(names)))
    ax_error_analysis.set_xticklabels([(n) for n in names], fontsize=45)
    ax_error_analysis.tick_params(axis = 'y', labelsize=30)
    ax_error_analysis.set_ylabel('F-1 Score', fontsize=35)

    ax_error_analysis2.legend(fontsize=40)
    ax_error_analysis2.set_xticks(np.arange(len(names)))
    ax_error_analysis2.set_xticklabels([(n) for n in names], fontsize=45)
    ax_error_analysis2.tick_params(axis = 'y', labelsize=30)
    ax_error_analysis2.set_ylabel('F-1 Score', fontsize=35)

    ax_oracle_analysis.legend(fontsize=40)
    ax_oracle_analysis.set_xticks(np.arange(len(names)))
    ax_oracle_analysis.set_xticklabels([(n) for n in names], fontsize=45)
    ax_oracle_analysis.tick_params(axis = 'y', labelsize=30)
    

    for fig in figs:
        fig.set_size_inches(30,15)
        fig.tight_layout()

    f3.set_size_inches(20,10)
    f3.tight_layout()
    f20.set_size_inches(20,10)
    f20.tight_layout()

    f4.set_size_inches(15,12)
    f4.tight_layout()
    f5.set_size_inches(15,12)
    f5.tight_layout()
    f18.set_size_inches(15,12)
    f18.tight_layout()
    f19.set_size_inches(15,12)
    f19.tight_layout()
    f10.set_size_inches(15,10)
    f10.tight_layout()
    f11.set_size_inches(15,10)
    f11.tight_layout()
    f13.set_size_inches(15,10)
    f13.tight_layout()
    f14.set_size_inches(10,10)
    f14.tight_layout()
    f22.set_size_inches(15,10)
    f22.tight_layout()
    f23.set_size_inches(15,10)
    f23.tight_layout()
    
        # for fig in figs:
        #     fig.set_size_inches(8,10)
        #     fig.tight_layout()

        # f3.set_size_inches(8,5)
        # f3.tight_layout()

        # ax_comp_t_prec.legend(fontsize=40, loc='lower right')
        # ax_comp_tl_prec.legend(fontsize=40, loc='lower right')

        # f4.set_size_inches(10,10)
        # f4.tight_layout()
        # f5.set_size_inches(10,10)
        # f5.tight_layout()
        # f11.set_size_inches(10,10)
        # f11.tight_layout()


    return figs, info



def average_stats(stats_list):
    avg = {}
    num_stats = len(stats_list)
    if num_stats == 0:
        return None
    if num_stats == 1:
        return stats_list[0]
    lookahead_steps = min(MAX_LOOKAHEAD, len(stats_list[0]['precision_breakdown']))
    if 'breakdown' in stats_list[0].keys():
        avg['breakdown'] = {k:np.sum([sl['breakdown'][k] for sl in stats_list]) for k in stats_list[0]['breakdown'].keys()}
    avg['precision_breakdown'] = [[ sum([sl['precision_breakdown'][s][c] for sl in stats_list])/num_stats for c in range(2)]for s in range(lookahead_steps)]
    avg['completeness_breakdown'] = {}
    if 'by_lookahead' in stats_list[0]['completeness_breakdown']:
        avg['completeness_breakdown']['by_lookahead'] = [[ sum([sl['completeness_breakdown']['by_lookahead'][s][c] for sl in stats_list])/num_stats for c in range(3)]for s in range(lookahead_steps)]
        if 'by_activity' in stats_list[0]['completeness_breakdown'].keys():
            avg['completeness_breakdown']['by_change_type'] = [[ sum([sl['completeness_breakdown']['by_change_type'][t][c] for sl in stats_list])/num_stats for c in range(3)]for t in range(3)]
        if 'by_activity' in stats_list[0]['completeness_breakdown'].keys():
            avg['completeness_breakdown']['by_activity'] = [[ sum([sl['completeness_breakdown']['by_activity'][t][c] for sl in stats_list])/num_stats for c in range(3)]for t in range(len(activity_list))]
    else:
        avg['completeness_breakdown']['by_lookahead'] = [[ sum([sl['completeness_breakdown'][s][c] for sl in stats_list])/num_stats for c in range(3)]for s in range(lookahead_steps)]
    if 'optimistic_completeness_breakdown' in stats_list[0].keys():
        avg['optimistic_completeness_breakdown'] = {
            'by_lookahead' : [[ sum([sl['optimistic_completeness_breakdown']['by_lookahead'][s][c] for sl in stats_list])/num_stats for c in range(3)]for s in range(lookahead_steps)]
        }        

    return avg

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
    parser.add_argument('--path', type=str, default='logs/CoRL_eval_0822_1607_original/', help='')
    
    args = parser.parse_args()


    all_training_days = [int(t) for t in os.listdir(args.path) if not t.startswith('visual')]
    all_training_days.sort(reverse=True)
    dirs = [os.path.join(args.path,str(p)) for p in all_training_days]

    datasets = os.listdir(dirs[0])


    if PLOT_CONFIDENCES:
        confidences = [float(k) for k in json.load(open(os.path.join(dirs[0],datasets[0],'oursconf6_50epochs','evaluation.json')))["with_confidence"].keys()]
        confidences.sort()
        confidences = confidences[:-1]
        confidences = [c for c in confidences if c > 0.4]
        data = [average_stats([json.load(open(os.path.join(dirs[0],dataset,'oursconf6_50epochs','evaluation.json')))["with_confidence"][str(conf)] for dataset in datasets]) for conf in confidences]


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
        'baselines' : ['FremenStateConditioned', 'LastSeenAndStaticSemantic', 'ours_50epochs'],
        'pretrain' : ['ourspt_50epochs', 'ours_50epochs'],
        'ablations' : ['ours_50epochs','ours_allEdges_50epochs','ours_timeLinear_50epochs']
    }


    for typ in [
        'baselines', 
        'ablations', 
        'pretrain',
        ]:


        if typ == 'ablations':
            training_days = [max(all_training_days)]
        else:
            training_days = all_training_days


        dirs = [os.path.join(args.path,str(p)) for p in training_days]

        methods = all_methods[typ]

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

            for method in methods:
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
