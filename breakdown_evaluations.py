import os
import json
import numpy as np
from math import isnan
import torch
from torch.nn.functional import one_hot
import wandb
from copy import deepcopy
from GraphTranslatorModule import _erase_edges
from encoders import human_readable_from_external
from evaluation import _get_masks

def evaluate_all_breakdowns(model, test_routines, additional_steps=5, deterministic_input_loop=False):
    metrics = {'recall_tp':{0:0}, 'precision_tp':{0:0}, 'typewise':{0:0}}
    metrics['recall_tp'].update({i+1:0 for i in range(additional_steps+1)})
    metrics['recall_tp'].update({-i-1:0 for i in range(additional_steps+1)})
    metrics['precision_tp'].update({i+1:0 for i in range(additional_steps+1)})
    metrics['precision_tp'].update({-i-1:0 for i in range(additional_steps+1)})
    metrics['typewise'].update({'take_out':{'correct':0,'wrong':0}, 'put_away':{'correct':0,'wrong':0}, 'other':{'correct':0,'wrong':0}})

    num_routines = len(test_routines)
    for (routine, additional_info) in test_routines:
        while len(routine) > 0:
            first_routine = test_routines.collate_fn([routine.pop()])
            eval, one_step_details = model.step(first_routine)

            change_type_masks = {}
            change_type_masks['take_out'] = ((first_routine['change_type'] == 1).to(bool))[one_step_details['evaluate_node']]
            change_type_masks['other'] = ((first_routine['change_type'] == 2).to(bool))[one_step_details['evaluate_node']]
            change_type_masks['put_away'] = ((first_routine['change_type'] == 3).to(bool))[one_step_details['evaluate_node']]

            gt_tensor = one_step_details['gt']['location'][one_step_details['evaluate_node']]
            output_tensor = one_step_details['output']['location'][one_step_details['evaluate_node']]
            input_tensor = one_step_details['input']['location'][one_step_details['evaluate_node']]

            predicted_static_mask = deepcopy((output_tensor == input_tensor).to(bool))
            predicted_changes = deepcopy(output_tensor)
            recall_breakdown = torch.zeros_like(input_tensor).to(int)
            predicted_changes[predicted_static_mask] = -1
            recall_breakdown[predicted_static_mask] = -1000

            ground_truth_static_mask = deepcopy((gt_tensor == input_tensor).to(bool))
            ground_truth_changes = deepcopy(output_tensor)
            precision_breakdown = torch.zeros_like(input_tensor).to(int)
            ground_truth_changes[ground_truth_static_mask] = -1
            precision_breakdown[ground_truth_static_mask] = -1000


            def update_breakdown(breakdown, wrong, new_changes, step_num):
                breakdown[(np.bitwise_and(new_changes, breakdown==0))] = step_num
                breakdown[(np.bitwise_and(breakdown==step_num, wrong))] = -step_num
                return breakdown

            wrong = deepcopy((gt_tensor != output_tensor).to(bool))
            predicted_moved = deepcopy((output_tensor != input_tensor).to(bool))
            ground_truth_moved = deepcopy((gt_tensor != input_tensor).to(bool))
            precision_breakdown = update_breakdown(precision_breakdown, wrong, predicted_moved, 1)
            recall_breakdown = update_breakdown(recall_breakdown, wrong, ground_truth_moved, 1)

            data_list = [test_routines.collate_fn([routine[j]]) for j in range(min(additional_steps, len(routine)))]
            
            for i,data in enumerate(data_list):
                if i>0:
                    data['edges'] = prev_edges
                _, details = model.step(data)
                gt_tensor, output_tensor = details['gt']['location'][details['evaluate_node']], details['output']['location'][details['evaluate_node']]
                wrong = (gt_tensor != output_tensor).to(bool)

                precision_breakdown = update_breakdown(precision_breakdown, 
                                                       wrong = deepcopy((output_tensor != ground_truth_changes).to(bool)), 
                                                       new_changes = deepcopy((output_tensor != input_tensor).to(bool)), 
                                                       step_num = i+2)
                recall_breakdown = update_breakdown(recall_breakdown, 
                                                    wrong = deepcopy((gt_tensor != predicted_changes).to(bool)), 
                                                    new_changes =  deepcopy((gt_tensor != input_tensor).to(bool)), 
                                                    step_num = i+2)

                if deterministic_input_loop:
                    prev_edges = one_hot(details['output']['location'], num_classes = details['output']['location'].size()[-1])
                else:
                    prev_edges = details['output_probs']['location']

            for val in metrics['recall_tp']:
                metrics['recall_tp'][val] += float((recall_breakdown == val).sum())/num_routines                
            for val in metrics['precision_tp']:
                metrics['precision_tp'][val] += float((precision_breakdown == val).sum())/num_routines
            for type in metrics['typewise']:
                if type == 0:
                    metrics['typewise'][type] = metrics['precision_tp'][0]
                else:
                    metrics['typewise'][type]['correct'] += float(np.bitwise_and(change_type_masks[type], precision_breakdown > 0).sum())/num_routines
                    metrics['typewise'][type]['wrong'] += float(np.bitwise_and(change_type_masks[type], precision_breakdown < 0).sum())/num_routines

    return metrics


def evaluate(model, data, cfg, output_dir, logger=None):
    
    info = evaluate_all_breakdowns(model, data.test_routines)
    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
        json.dump(info,f)

    return info