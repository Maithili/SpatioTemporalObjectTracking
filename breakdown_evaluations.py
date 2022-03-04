import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from math import isnan
import torch
from torch.nn.functional import one_hot
import wandb
from copy import deepcopy
from GraphTranslatorModule import _erase_edges
from encoders import human_readable_from_external
from evaluation import _get_masks

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
newcolors[:25, :] = white
newcmp = ListedColormap(['white', 'tab:blue', 'tab:orange', 'tab:purple'])

def evaluate_all_breakdowns(model, test_routines, lookahead_steps=6, deterministic_input_loop=False, node_names=[]):
    
    num_change_types = 3

    raw_data = {'inputs':[], 'outputs':[], 'ground_truths':[], 'futures':[], 'change_types':[]}
    results = {'recall_breakdown':[[0,0] for _ in range(lookahead_steps)],
               'precision_breakdown': {
                    'missed_changes' : 0,
                    'by_lookahead' : [[0,0] for _ in range(lookahead_steps)],
                    'by_change_type' : [[0,0,0] for _ in range(num_change_types)]
                }
              }
    figures = []

    results['all_moves'] = []

    num_routines = len(test_routines)
    for (routine, additional_info) in test_routines:
        
        routine_length = len(routine)
        num_nodes = additional_info['active_nodes'].sum()
        routine_inputs = torch.empty(routine_length, num_nodes)
        routine_outputs = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_output_step = torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)
        routine_ground_truths = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_futures = torch.ones(routine_length, num_nodes) * -1
        routine_change_types = torch.zeros(routine_length, num_nodes).to(int)

        changes_output_all = torch.zeros(routine_length, num_nodes).to(bool)
        changes_gt = torch.zeros(routine_length, num_nodes).to(bool)
        for step in range(routine_length):
            data_list = [test_routines.collate_fn([routine[j]]) for j in range(step, min(step+lookahead_steps, routine_length))]
            for i,data in enumerate(data_list):
                if i>0:
                    data['edges'] = prev_edges
                
                _, details = model.step(data)
                
                gt_tensor = details['gt']['location'][details['evaluate_node']]
                output_tensor = details['output']['location'][details['evaluate_node']]
                input_tensor = details['input']['location'][details['evaluate_node']]

                if i == 0:
                    routine_inputs[step,:] = deepcopy(input_tensor)
                    changes_gt[step, :] = deepcopy((gt_tensor != input_tensor).to(bool))
                    routine_ground_truths[step, changes_gt[step, :]] = deepcopy(gt_tensor[changes_gt[step, :]])
                    routine_change_types[step,:] = deepcopy(data['change_type'].to(int)[details['evaluate_node']])
                new_changes_out = deepcopy(np.bitwise_and(output_tensor != input_tensor , np.bitwise_not(changes_output_all[step,:]))).to(bool)
                routine_outputs[step, :][new_changes_out] = deepcopy(output_tensor[new_changes_out])
                routine_output_step[step, :][new_changes_out] = i
                changes_output_all[step,:] = deepcopy(np.bitwise_or(changes_output_all[step,:], new_changes_out)).to(bool)

                if deterministic_input_loop:
                    prev_edges = one_hot(details['output']['location'], num_classes = details['output']['location'].size()[-1]).to(torch.float32)
                else:
                    prev_edges = (details['output_probs']['location']).to(torch.float32)

        routine_future_steps = torch.zeros_like(routine_ground_truths)
        routine_futures = deepcopy(routine_ground_truths)
        for i in range(routine_length-2, -1, -1):
            mask_copy_next = routine_futures[i,:] == -1
            routine_futures[i,:][mask_copy_next] = routine_futures[i+1,:][mask_copy_next]
            routine_future_steps[i,:][mask_copy_next] = routine_future_steps[i+1,:][mask_copy_next] - 1

        routine_future_steps[routine_futures < 0] = routine_future_steps.min().min()-1

        correct = deepcopy(routine_outputs == routine_ground_truths).to(int)
        wrong = deepcopy(routine_outputs != routine_ground_truths).to(int)

        # assert np.equal((routine_ground_truths >= 0), changes_gt)
        for ls in range(lookahead_steps):
            changes_output_for_step = deepcopy(routine_output_step  == ls)
            results['recall_breakdown'][ls][0] += int((correct[changes_output_for_step]).sum())
            results['recall_breakdown'][ls][1] += int((wrong[changes_output_for_step]).sum())
            changes_output_and_gt = deepcopy(np.bitwise_and(changes_output_for_step, changes_gt))
            results['precision_breakdown']['by_lookahead'][ls][0] += int((correct[changes_output_and_gt]).sum())
            results['precision_breakdown']['by_lookahead'][ls][1] += int((wrong[changes_output_and_gt]).sum())

        new_missed = int(np.bitwise_and(np.bitwise_not(changes_output_all), changes_gt).sum())
        results['precision_breakdown']['missed_changes'] += new_missed

        routine_change_types = routine_change_types.to(int)
        assert torch.equal(routine_change_types > 0, changes_gt)
        assert torch.equal(routine_output_step<lookahead_steps, changes_output_all)
        # changes_output_and_gt_all = deepcopy(np.bitwise_and(changes_output_all, changes_gt))
        for ct in range(num_change_types):
            ct_mask = deepcopy(np.bitwise_and(changes_output_all, routine_change_types == (ct+1)))
            results['precision_breakdown']['by_change_type'][ct][0] += int((correct[ct_mask]).sum())
            results['precision_breakdown']['by_change_type'][ct][1] += int((wrong[ct_mask]).sum())
            ct_mask_missed = deepcopy(np.bitwise_and(np.bitwise_not(changes_output_all), routine_change_types == (ct+1)))
            results['precision_breakdown']['by_change_type'][ct][2] += int(ct_mask_missed.sum())
        # print (routine_change_types.unique())
        assert abs(sum([results['precision_breakdown']['by_change_type'][ct][2] for ct in range(num_change_types)]) - results['precision_breakdown']['missed_changes']) == 0 , "missed changes don't add up!"
        assert abs(sum([results['precision_breakdown']['by_change_type'][ct][0] for ct in range(num_change_types)]) - sum([results['precision_breakdown']['by_lookahead'][s][0] for s in range(lookahead_steps)])) == 0, "correct changes don't add up"
        assert abs(sum([results['precision_breakdown']['by_change_type'][ct][1] for ct in range(num_change_types)]) - sum([results['precision_breakdown']['by_lookahead'][s][1] for s in range(lookahead_steps)])) == 0, "wrong changes don't add up"

        fig, axs = plt.subplots(1,5)
        fig.set_size_inches(30,20)

        labels = []
        if len(node_names) > 0:
            labels = [name for name,active in zip(node_names, additional_info['active_nodes']) if active]

        ax = axs[0]
        img_output = lookahead_steps - routine_output_step
        ax.imshow(img_output, cmap='Blues', vmin=0, vmax=lookahead_steps, aspect='auto')
        ax.set_title('Stepwise Output')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        ax = axs[1]
        img_output = changes_output_all.to(int)
        img_output[routine_outputs != routine_futures] *= (-1)
        ax.imshow(img_output, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Output Location Correctness')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        ax = axs[2]
        ax.imshow(routine_future_steps.to(int), cmap='Blues', aspect='auto')
        ax.set_title('Future Ground Truths')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        ax = axs[3]
        img_gt = (routine_ground_truths >= 0).to(float)
        img_gt[routine_ground_truths != routine_outputs] *= (-1)
        img_gt[routine_outputs == -1] *= 0.5
        ax.imshow(img_gt, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Ground Truths w/ Correctness')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        ax = axs[4]
        img_ct = routine_change_types
        ax.imshow(img_ct, cmap=newcmp, vmin=0, vmax=3, aspect='auto')
        ax.set_title('Ground Truths w/ Change Types')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        fig.tight_layout()
        figures.append(fig)

        moves = {}
        for i,act_node in enumerate(labels):
            changes = routine_ground_truths[:,i][routine_ground_truths[:,i]>0]
            changes_predicted = routine_outputs[:,i][routine_outputs[:,i]>0]
            node_changes = [node_names[c] for c in changes]
            node_changes_predicted = [node_names[c] for c in changes_predicted]
            moves[act_node] = {'actual':node_changes, 'predicted':node_changes_predicted}

        results['all_moves'].append(moves)

        raw_data['inputs'].append(routine_inputs)
        raw_data['outputs'].append(routine_outputs)
        raw_data['ground_truths'].append(routine_ground_truths)
        raw_data['futures'].append(routine_futures)
        raw_data['change_types'].append(routine_change_types)

    return results, raw_data, figures


def evaluate(model, data, cfg, output_dir, logger=None):
    
    info, raw_data, figures = evaluate_all_breakdowns(model, data.test_routines, node_names=data.node_classes)
    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
        json.dump(info,f)

    torch.save(raw_data, os.path.join(output_dir, 'raw_data.pt'))

    os.makedirs(os.path.join(output_dir,'figures'))
    for i,fig  in enumerate(figures):
        fig.savefig(os.path.join(output_dir,'figures',str(i)+'.jpg'))

    return info