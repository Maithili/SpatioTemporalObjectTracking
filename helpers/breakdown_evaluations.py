import os
import shutil
import json
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from math import isnan, floor, ceil, sqrt
import torch
from torch.nn.functional import one_hot
import wandb
from copy import deepcopy
from encoders import human_readable_from_external

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
newcolors[:25, :] = white
newcmp = ListedColormap(['white', 'tab:blue', 'tab:orange', 'tab:purple'])

activity_list = [
"brushing_teeth",
"showering",
"breakfast",
"dinner",
"computer_work",
"lunch",
"cleaning",
"laundry",
"leave_home",
"come_home",
"socializing",
"taking_medication",
"watching_tv",
"vaccuum_cleaning",
"reading",
"going_to_the_bathroom",
"getting_dressed",
"kitchen_cleaning",
"take_out_trash",
"wash_dishes",
"playing_music",
"listening_to_music",
None
]
def _erase_edges(edges, dyn_mask):
    empty_edges = torch.ones_like(edges)/edges.size()[-1]
    empty_edges[dyn_mask == 0] = edges[dyn_mask == 0]
    empty_edges = empty_edges/(empty_edges.sum(dim=-1).unsqueeze(-1).repeat(1,1,edges.size()[-1])+1e-8)
    return empty_edges

def evaluate_all_breakdowns(model, test_routines, activity_list, lookahead_steps=12, deterministic_input_loop=False, node_names=[], print_importance=False, confidences=[], learned_model=False):
    
    use_cuda = torch.cuda.is_available() and learned_model
    if use_cuda: model.to('cuda')
    else: print(f'Learned Model {learned_model} NOT USING CUDA. THIS WILL TAKE AGESSSSS!!!!!!!!!!!!')

    num_change_types = 3

    raw_data = {'inputs':[], 'outputs':[], 'ground_truths':[], 'futures':[], 'change_types':[]}
    results = {'breakdown':{'correct':[0 for _ in range(lookahead_steps)], 
                            'wrong':[0 for _ in range(lookahead_steps)], 
                            'missed':[0 for _ in range(lookahead_steps)], 
                            'fp':[0 for _ in range(lookahead_steps)], 
                            'tn':[0 for _ in range(lookahead_steps)]},
               'precision_breakdown':[[0,0] for _ in range(lookahead_steps)],
               'completeness_breakdown': {
                    'by_lookahead' : [[0,0,0] for _ in range(lookahead_steps)],
                    'by_change_type' : [[0,0,0] for _ in range(num_change_types)],
                    'by_activity' : [[0,0,0] for _ in activity_list]
                },
                'optimistic_completeness_breakdown': {
                    'by_lookahead' : [[0,0,0] for _ in range(lookahead_steps)]
                },
                'timeonly_breakdown_direct':{
                    'correct': 0,
                    'wrong': 0
                },
                'timeonly_breakdown_playahead':{
                    'correct': 0,
                    'wrong': 0
                }
              }
    results_conf = {conf: {'precision_breakdown':[[0,0] for _ in range(lookahead_steps)],
                           'completeness_breakdown': [[0,0,0] for _ in range(lookahead_steps)]} for conf in confidences}
    results_oracle = {'with_oracle': {'correct': 0, 'wrong': 0}, 'without_oracle': {'correct': 0, 'wrong': 0}}
    results_by_obj = [{'correct':[], 'wrong':[], 'missed':[], 'fp':[]} for _ in range(lookahead_steps)]
    object_stats = []
    figures = []
    figures_imp = []

    results['all_moves'] = []
    results['num_changes'] = [[] for _ in range(lookahead_steps)]
    total_num_steps = 0

    for (routine, additional_info) in test_routines:

        routine_length = len(routine)
        total_num_steps += routine_length
        num_nodes = additional_info['active_nodes'].sum()

        if 'activity' in additional_info:
            activities = additional_info['activity']
        else:
            activities = [None] * routine_length
        routine_inputs = torch.empty(routine_length, num_nodes)
        routine_outputs = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_output_step = torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)
        routine_ground_truths = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_ground_truth_step = torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)
        routine_futures = torch.ones(routine_length, num_nodes) * -1
        routine_change_types = torch.zeros(routine_length, num_nodes).to(int)
        routine_activities = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_outputs_conf = {c:{'output':torch.ones(routine_length, num_nodes).to(int) * -1, 'step':torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)} for c in confidences}

        changes_output_all = torch.zeros(routine_length, num_nodes).to(bool)
        changes_gt_all = torch.zeros(routine_length, num_nodes).to(bool)

        # play_forward_edges = test_routines.collate_fn([routine[0]])['edges']
        if print_importance: 
            fig_imp, axs_imp = plt.subplots(int(floor(sqrt(routine_length))), int(ceil(routine_length/floor(sqrt(routine_length)))))
            axs_imp = axs_imp.reshape(-1)

        for step in range(routine_length):
            data_list = [test_routines.collate_fn([routine[j]]) for j in range(step, min(step+lookahead_steps, routine_length))]
            activities_in_lookahead = [activities[j] for j in range(step, min(step+lookahead_steps, routine_length))]

            first_data = deepcopy(data_list[0])

            # def evaluate_timeonly(edges, results_dict):
            #     first_data['edges'] = edges
            #     if use_cuda: 
            #         for k in first_data.keys():
            #             first_data[k] = first_data[k].to('cuda')
            #     _, details = model.step(first_data)
            #     correct = deepcopy(details['gt']['location'][details['evaluate_node']] == details['output']['location'][details['evaluate_node']]).to(int)
            #     wrong = deepcopy(details['gt']['location'][details['evaluate_node']] != details['output']['location'][details['evaluate_node']]).to(int)
            #     results_dict['correct'] += float(correct.sum())/routine_length
            #     results_dict['wrong'] += float(wrong.sum())/routine_length
            #     return (details['output_probs']['location']).to(torch.float32)

            # _ = evaluate_timeonly(_erase_edges(first_data['edges'], first_data['dynamic_edges_mask']), results['timeonly_breakdown_direct'])
            # play_forward_edges = evaluate_timeonly(play_forward_edges, results['timeonly_breakdown_playahead'])
            
            context_pred = None
            for i,(data, act) in enumerate(zip(data_list,activities_in_lookahead)):
                assert i<lookahead_steps
                if i>0:
                    data['edges'] = prev_edges
                    # data['processed_context'] = context_pred
                
                if use_cuda: 
                    for k in data.keys():
                        data[k] = data[k].to('cuda')
                        
                _, details, context_pred = model.step(data, prev_context=context_pred)
                
                if i==0 and print_importance:
                    axs_imp[step].imshow(details['importance_weights'].squeeze(0))

                gt_tensor = details['gt']['location'][details['evaluate_node']].cpu()
                output_tensor = details['output']['location'][details['evaluate_node']].cpu()
                output_probs = details['output_probs']['location'][details['evaluate_node']].cpu()
                input_tensor = details['input']['location'][details['evaluate_node']].cpu()
                
                if i == 0:
                    routine_inputs[step,:] = deepcopy(input_tensor)
                new_changes_out = deepcopy(np.bitwise_and(output_tensor != input_tensor , np.bitwise_not(changes_output_all[step,:]))).to(bool)
                new_changes_gt = deepcopy(np.bitwise_and(gt_tensor != routine_inputs[step,:] , np.bitwise_not(changes_gt_all[step,:]))).to(bool)
                
                if new_changes_out.max() > 0:
                    print('No!!')

                if i == 0:
                    moved_output_probs = details['output_probs']['location'].cpu().detach().numpy()
                    moved_output_probs -= np.eye(moved_output_probs.shape[-1])
                    moved_output_tensor = moved_output_probs.argmax(-1)[details['evaluate_node'].cpu()]
                    correct_without_oracle = deepcopy(np.bitwise_and((output_tensor == gt_tensor), new_changes_gt))
                    wrong_without_oracle = deepcopy(np.bitwise_and((output_tensor != gt_tensor), new_changes_gt))
                    correct_with_oracle = deepcopy(np.bitwise_and((moved_output_tensor == gt_tensor), new_changes_gt))
                    wrong_with_oracle = deepcopy(np.bitwise_and((moved_output_tensor != gt_tensor), new_changes_gt))
                    results_oracle['with_oracle']['correct'] += correct_with_oracle.sum()
                    results_oracle['with_oracle']['wrong'] += wrong_with_oracle.sum()
                    results_oracle['without_oracle']['correct'] += correct_without_oracle.sum()
                    results_oracle['without_oracle']['wrong'] += wrong_without_oracle.sum()
                
                if i == 0:
                    origins_one = np.arange(num_nodes)
                    object_stats += ([((o,int(d)),step) for o,d in zip(origins_one[(new_changes_gt).to(bool)], gt_tensor[(new_changes_gt).to(bool)])])
                # if i == 0 and sum(new_changes_gt)>0:
                #     for obj in np.argwhere(new_changes_gt)[0]:
                #         object_stats['timesteps'][obj].append(step)
                #     object_stats['num_times_moved'] += new_changes_gt
                # assert new_changes_out.sum().sum() == 0
                routine_outputs[step, :][new_changes_out] = deepcopy(output_tensor[new_changes_out])
                routine_output_step[step, :][new_changes_out] = i
                routine_ground_truths[step, :][new_changes_gt] = deepcopy(gt_tensor[new_changes_gt])
                routine_ground_truth_step[step, :][new_changes_gt] = i
                # assert all(data['change_type'].to(int)[details['evaluate_node']][new_changes_gt] > 0)
                routine_change_types[step,:][new_changes_gt] = deepcopy(data['change_type'].cpu().to(int)[details['evaluate_node']])[new_changes_gt]
                routine_activities[step,:][new_changes_gt] = act
                changes_output_all[step,:] = deepcopy(np.bitwise_or(changes_output_all[step,:], new_changes_out)).to(bool)
                changes_gt_all[step,:] = deepcopy(np.bitwise_or(changes_gt_all[step,:], new_changes_gt)).to(bool)

                # output_conf = (output_probs*(torch.nn.functional.one_hot(output_tensor, num_classes=output_probs.size()[-1]))).sum(-1) #- output_probs*(torch.nn.functional.one_hot(input_tensor, num_classes=output_probs.size()[-1]))).sum(-1)
                output_conf = (output_probs*(torch.nn.functional.one_hot(output_tensor, num_classes=output_probs.size()[-1]))).sum(-1)*(routine_inputs[step,:]!=output_tensor).to(float)
                
                for conf in confidences:
                    new_changes_conf = deepcopy(np.bitwise_and((output_conf > conf), np.bitwise_not(routine_outputs_conf[conf]['step'][step, :]<lookahead_steps))).to(bool)
                    routine_outputs_conf[conf]['output'][step, :][new_changes_conf] = deepcopy(output_tensor[new_changes_conf])
                    routine_outputs_conf[conf]['step'][step, :][new_changes_conf] = i
                

                if deterministic_input_loop:
                    prev_edges = one_hot(details['output']['location'], num_classes = details['output']['location'].size()[-1]).to(torch.float32)
                else:
                    prev_edges = (details['output_probs']['location']).to(torch.float32)

        gt_changes_list = np.argwhere((routine_ground_truth_step == 0))  # step, object
        for gt_step, gt_obj in zip(gt_changes_list[0,:], gt_changes_list[1,:]):
            gt_dest = routine_ground_truths[gt_step, gt_obj]
            outputs_to_consider = np.array([routine_outputs[lb, gt_obj] for lb in range(gt_step, max(-1,gt_step-lookahead_steps), -1)])
            outputs_steps_to_consider = np.array([routine_output_step[lb, gt_obj] for lb in range(gt_step, max(-1,gt_step-lookahead_steps), -1)])
            assert gt_step < lookahead_steps or len(outputs_to_consider) == lookahead_steps
            
            for s in range(len(outputs_to_consider)):
                outputs_for_lookahead = [out_dest for out_dest,pred_st in zip(outputs_to_consider[:s+1], outputs_steps_to_consider[:s+1]) if pred_st <= s]
                if len(outputs_for_lookahead) > 0:
                    assert max(outputs_for_lookahead) >= 0, f"Something smells fishy! {s} : {outputs_for_lookahead}\n{outputs_to_consider}\n{outputs_steps_to_consider}"
                    if gt_dest in outputs_for_lookahead:
                        results['optimistic_completeness_breakdown']['by_lookahead'][s][0] += 1
                    else:
                        results['optimistic_completeness_breakdown']['by_lookahead'][s][1] += 1
                else:
                    results['optimistic_completeness_breakdown']['by_lookahead'][s][2] += 1
            

        if print_importance:
            fig_imp.set_size_inches(25,22)
            fig_imp.tight_layout()
            figures_imp.append(fig_imp)
        
        correct = {conf:deepcopy(routine_outputs_conf[conf]['output'] == routine_ground_truths).to(int) for conf in confidences}
        wrong = {conf:deepcopy(routine_outputs_conf[conf]['output'] != routine_ground_truths).to(int) for conf in confidences}
        
        for conf in confidences:
            for ls in range(lookahead_steps):
                changes_output_for_step = deepcopy(routine_outputs_conf[conf]['step']  <= ls)
                changes_gt_for_step = deepcopy(routine_ground_truth_step  <= ls)
                changes_output_and_gt = deepcopy(np.bitwise_and(changes_output_for_step, changes_gt_for_step))
                changes_output_and_not_gt = deepcopy(np.bitwise_and(changes_output_for_step, np.bitwise_not(changes_gt_for_step)))
                changes_not_output_and_gt = deepcopy(np.bitwise_and(np.bitwise_not(changes_output_for_step), changes_gt_for_step))
                results_conf[conf]['precision_breakdown'][ls][0] += int(changes_output_and_gt.sum())/routine_length
                results_conf[conf]['precision_breakdown'][ls][1] += int(changes_output_and_not_gt.sum())/routine_length
                results_conf[conf]['completeness_breakdown'][ls][0] += int((correct[conf][changes_output_and_gt]).sum())/routine_length
                results_conf[conf]['completeness_breakdown'][ls][1] += int((wrong[conf][changes_output_and_gt]).sum())/routine_length
                results_conf[conf]['completeness_breakdown'][ls][2] += int((changes_not_output_and_gt).sum())/routine_length
                

        correct = deepcopy(routine_outputs == routine_ground_truths).to(int)
        wrong = deepcopy(routine_outputs != routine_ground_truths).to(int)

        # assert np.equal((routine_ground_truths >= 0), changes_gt)
        for ls in range(lookahead_steps):
            changes_output_for_step = deepcopy(routine_output_step  <= ls)
            changes_gt_for_step = deepcopy(routine_ground_truth_step  <= ls)
            # results['num_changes'][ls].append(list(changes_gt_for_step.sum(-1).reshape(-1)))
            changes_output_and_gt = deepcopy(np.bitwise_and(changes_output_for_step, changes_gt_for_step))
            changes_output_and_not_gt = deepcopy(np.bitwise_and(changes_output_for_step, np.bitwise_not(changes_gt_for_step)))
            changes_not_output_and_gt = deepcopy(np.bitwise_and(np.bitwise_not(changes_output_for_step), changes_gt_for_step))
            changes_tn = deepcopy(np.bitwise_and(np.bitwise_not(changes_output_for_step), np.bitwise_not(changes_gt_for_step)))
            results['breakdown']['correct'][ls] += int((correct[changes_output_and_gt]).sum())
            results['breakdown']['wrong'][ls] += int((wrong[changes_output_and_gt]).sum())
            results['breakdown']['missed'][ls] += int(changes_not_output_and_gt.sum())
            results['breakdown']['fp'][ls] += int(changes_output_and_not_gt.sum())
            results['breakdown']['tn'][ls] += int(changes_tn.sum())

            results['precision_breakdown'][ls][0] += int(changes_output_and_gt.sum())/routine_length
            results['precision_breakdown'][ls][1] += int(changes_output_and_not_gt.sum())/routine_length
            results['completeness_breakdown']['by_lookahead'][ls][0] += int((correct[changes_output_and_gt]).sum())/routine_length
            results['completeness_breakdown']['by_lookahead'][ls][1] += int((wrong[changes_output_and_gt]).sum())/routine_length
            results['completeness_breakdown']['by_lookahead'][ls][2] += int(changes_not_output_and_gt.sum())/routine_length
            origins = np.ones_like(routine_ground_truths) * np.arange(num_nodes)
            results_by_obj[ls]['correct'] += ([(o,int(d)) for o,d in zip(origins[(correct * changes_output_and_gt).to(bool)], routine_ground_truths[(correct * changes_output_and_gt).to(bool)])])
            results_by_obj[ls]['wrong'] += ([(o,int(d)) for o,d in zip(origins[(wrong * changes_output_and_gt).to(bool)], routine_ground_truths[(wrong * changes_output_and_gt).to(bool)])])
            results_by_obj[ls]['missed'] += ([(o,int(d)) for o,d in zip(origins[(changes_not_output_and_gt).to(bool)], routine_ground_truths[(changes_not_output_and_gt).to(bool)])])
            results_by_obj[ls]['fp'] += ([(o,int(d)) for o,d in zip(origins[(changes_output_and_not_gt).to(bool)], routine_ground_truths[(changes_output_and_not_gt).to(bool)])])
            
        routine_change_types = routine_change_types.to(int)
        # assert torch.equal(routine_change_types > 0, changes_gt_all)
        assert torch.equal(routine_output_step<lookahead_steps, changes_output_all)
        # changes_output_and_gt_all = deepcopy(np.bitwise_and(changes_output_all, changes_gt))
        for ct in range(num_change_types):
            ct_mask = deepcopy(np.bitwise_and(changes_output_all, routine_change_types == (ct+1)))
            results['completeness_breakdown']['by_change_type'][ct][0] += int((correct[ct_mask]).sum())/routine_length
            results['completeness_breakdown']['by_change_type'][ct][1] += int((wrong[ct_mask]).sum())/routine_length
            ct_mask_missed = deepcopy(np.bitwise_and(np.bitwise_not(changes_output_all), routine_change_types == (ct+1)))
            results['completeness_breakdown']['by_change_type'][ct][2] += int(ct_mask_missed.sum())/routine_length
        # print (routine_change_types.unique())
        # assert abs(sum([results['completeness_breakdown']['by_change_type'][ct][2] for ct in range(num_change_types)]) - results['completeness_breakdown']['by_lookahead'][-1][2]) == 0 , "missed changes don't add up!"
        # assert abs(sum([results['completeness_breakdown']['by_change_type'][ct][0] for ct in range(num_change_types)]) - results['completeness_breakdown']['by_lookahead'][-1][0]) == 0, "correct changes don't add up"
        # assert abs(sum([results['completeness_breakdown']['by_change_type'][ct][1] for ct in range(num_change_types)]) - results['completeness_breakdown']['by_lookahead'][-1][1]) == 0, "wrong changes don't add up"

        for aidx in range(len(activity_list)):
            a_mask = deepcopy(np.bitwise_and(changes_output_all, routine_activities == aidx))
            results['completeness_breakdown']['by_activity'][aidx][0] += int((correct[a_mask]).sum())/routine_length
            results['completeness_breakdown']['by_activity'][aidx][1] += int((wrong[a_mask]).sum())/routine_length
            a_mask_missed = deepcopy(np.bitwise_and(np.bitwise_not(changes_output_all), routine_activities == aidx))
            results['completeness_breakdown']['by_activity'][aidx][2] += int(a_mask_missed.sum())/routine_length

        
        fig, axs = plt.subplots(1,5)
        fig.set_size_inches(30,20)

        labels = []
        if len(node_names) > 0:
            labels = [name for name,active in zip(node_names, additional_info['active_nodes']) if active]

        ax = axs[0]
        img_output = lookahead_steps - routine_output_step
        img_output[routine_ground_truths == -1] *= (-1)
        ax.imshow(img_output, cmap='RdBu', vmin=-lookahead_steps, vmax=lookahead_steps, aspect='auto')
        ax.set_title('Stepwise Output')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        ax = axs[1]
        img_output = lookahead_steps - routine_ground_truth_step
        ax.imshow(img_output, cmap='Blues', vmin=0, vmax=lookahead_steps, aspect='auto')
        ax.set_title('Stepwise Ground Truth')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        ax = axs[2]
        img_output = changes_output_all.to(float)
        img_output[routine_outputs != routine_ground_truths] *= (-1)
        img_output[np.bitwise_and(changes_gt_all, routine_outputs != routine_ground_truths)] *= (-0.5)
        ax.imshow(img_output, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Output Correctness')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        ax = axs[3]
        img_gt = (changes_gt_all).to(float)
        img_gt[routine_ground_truths != routine_outputs] *= (-1)
        img_gt[np.bitwise_and(changes_output_all, routine_ground_truths != routine_outputs)] *= (-0.5)
        ax.imshow(img_gt, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Ground Truths\n w/ Correctness')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        ax = axs[4]
        img_ct = routine_change_types
        ax.imshow(img_ct, cmap=newcmp, vmin=0, vmax=3, aspect='auto')
        ax.set_title('Ground Truths\n w/ Change Types')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        fig.tight_layout()
        figures.append(fig)

        moves = {}
        for i,act_node in enumerate(labels):
            # changes = torch.stack([routine_ground_truths[:,i][routine_ground_truths[:,i]>0], routine_outputs[:,i][routine_ground_truths[:,i]>0]], dim=-1)
            # changes_predicted = torch.stack([routine_outputs[:,i][routine_outputs[:,i]>0], routine_ground_truths[:,i][routine_outputs[:,i]>0]], dim=-1)
            # node_changes = [node_names[c] if c>=0 else None for c in changes]
            # node_changes_predicted = [node_names[c] if c>=0 else None for c in changes_predicted]
            any_changes = (routine_ground_truths[:,i]+routine_outputs[:,i])>0
            def get_name(idx):
                if idx>=0: return node_names[idx]
                else: return 'No Change'
            changes = [get_name(o)+'({})'.format(o_st)+ '  ' + get_name(gt)+'({})'.format(gt_st) for (o, o_st, gt, gt_st) in zip(routine_outputs[:,i][any_changes], routine_output_step[:,i][any_changes], routine_ground_truths[:,i][any_changes], routine_ground_truth_step[:,i][any_changes])]
            moves[act_node] = {'pred/actual':changes}

        results['all_moves'].append(moves)

        raw_data['inputs'].append(routine_inputs)
        raw_data['outputs'].append(routine_outputs)
        raw_data['ground_truths'].append(routine_ground_truths)
        raw_data['futures'].append(routine_futures)
        # raw_data['change_types'].append(routine_change_types)

    transition_diff = {}
    transitions = []
    for tr,step in object_stats:
        if tr in transition_diff:
            transition_diff[tr].append(step)
        else:
            transitions.append(tr)
            transition_diff[tr] = [step]
    


    obj_eval_figs = []
    for ls in range(lookahead_steps):
        def prec(obj_dest_pair):
            return results_by_obj[ls]['correct'].count(obj_dest_pair)/(results_by_obj[ls]['correct'].count(obj_dest_pair)+results_by_obj[ls]['wrong'].count(obj_dest_pair)+results_by_obj[ls]['fp'].count(obj_dest_pair)+1e-8)
        def recl(obj_dest_pair):
            return results_by_obj[ls]['correct'].count(obj_dest_pair)/(results_by_obj[ls]['correct'].count(obj_dest_pair)+results_by_obj[ls]['wrong'].count(obj_dest_pair)+results_by_obj[ls]['missed'].count(obj_dest_pair)+1e-8)
        fig, axs = plt.subplots(1,2)
        precisions = [prec(tr) for tr in transitions]
        recalls = [recl(tr) for tr in transitions]
        axs[0].scatter([len(transition_diff[tr]) for tr in transitions], [np.std(transition_diff[tr]) for tr in transitions], color=[[1-p,0,p]for p in precisions])
        axs[1].scatter([len(transition_diff[tr]) for tr in transitions], [np.std(transition_diff[tr]) for tr in transitions], color=[[1-r,0,r]for r in recalls])
        # axs[0].scatter(prob, results_by_obj[ls][typ].reshape(-1)[by_prob], marker='x', color=colors_obj_eval[typ], label=typ)
        # axs[1].plot(var, results_by_obj[ls][typ].reshape(-1)[by_var], color=colors_obj_eval[typ], label=typ)
        axs[0].set_ylabel('Time Variability')
        axs[1].set_ylabel('Time Variability')
        axs[0].set_xlabel('Num. times object moved')
        axs[1].set_xlabel('Num. times object moved')
        axs[0].set_title(f'Precision-{ls}')
        axs[1].set_title(f'Recall-{ls}')
        obj_eval_figs.append(fig)

    results['with_confidence'] = results_conf

    return results, raw_data, figures + figures_imp + obj_eval_figs


def evaluate(model, data, output_dir, print_importance=False, lookahead_steps=12, confidences=[], learned_model=False):
    
    info, raw_data, figures = evaluate_all_breakdowns(model, data.test_routines, node_names=data.common_data['node_classes'], print_importance=print_importance, lookahead_steps=lookahead_steps, confidences=confidences, learned_model=learned_model, activity_list=data.common_data['activities'])
    json.dump(info, open(os.path.join(output_dir, 'evaluation.json'), 'w'), indent=4)

    torch.save(raw_data, os.path.join(output_dir, 'raw_data.pt'))

    if os.path.exists(os.path.join(output_dir,'figures')):
        shutil.rmtree(os.path.join(output_dir,'figures'))
    os.makedirs(os.path.join(output_dir,'figures'))
    for i,fig  in enumerate(figures):
        fig.savefig(os.path.join(output_dir,'figures',str(i)+'.jpg'))

    plt.close('all')

    return info