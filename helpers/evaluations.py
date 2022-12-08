import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from math import isnan, floor, ceil, sqrt
import torch
from torch.nn.functional import one_hot
import wandb
from copy import deepcopy

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

def evaluate_all_breakdowns(model, test_routines, lookahead_steps=12, deterministic_input_loop=False, node_names=[], print_importance=False, confidences=[], learned_model=False):
    
    use_cuda = torch.cuda.is_available() and learned_model
    if use_cuda: model.to('cuda')
    else: print(f'Learned Model {learned_model} NOT USING CUDA. THIS WILL TAKE AGESSSSS!!!!!!!!!!!!')

    num_change_types = 3

    raw_data = {'inputs':[], 'outputs':[], 'ground_truths':[], 'futures':[], 'change_types':[]}
    results = {conf: 
                {'moved':{'correct':[0 for _ in range(lookahead_steps)], 
                            'wrong':[0 for _ in range(lookahead_steps)], 
                            'missed':[0 for _ in range(lookahead_steps)]},
                 'unmoved':{'fp':[0 for _ in range(lookahead_steps)], 
                            'tn':[0 for _ in range(lookahead_steps)]}
                } for conf in confidences}
    results_other = {'moved_change_type' : {'correct':[0 for _ in range(num_change_types)], 
                                            'wrong':[0 for _ in range(num_change_types)], 
                                            'missed':[0 for _ in range(num_change_types)]}}
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

        routine_inputs = torch.empty(routine_length, num_nodes)
        routine_outputs = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_output_step = torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)
        routine_ground_truths = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_ground_truth_step = torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)
        routine_futures = torch.ones(routine_length, num_nodes) * -1
        routine_change_types = torch.zeros(routine_length, num_nodes).to(int)
        routine_outputs_conf = {c:{'output':torch.ones(routine_length, num_nodes).to(int) * -1, 'step':torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)} for c in confidences}

        changes_output_all = torch.zeros(routine_length, num_nodes).to(bool)
        changes_gt_all = torch.zeros(routine_length, num_nodes).to(bool)

        if print_importance: 
            fig_imp, axs_imp = plt.subplots(int(floor(sqrt(routine_length))), int(ceil(routine_length/floor(sqrt(routine_length)))))
            axs_imp = axs_imp.reshape(-1)

        for step in range(routine_length):
            data_list = [test_routines.collate_fn([routine[j]]) for j in range(step, min(step+lookahead_steps, routine_length))]


            for i,data in enumerate(data_list):
                assert i<lookahead_steps
                if i>0:
                    data['edges'] = prev_edges
                
                if use_cuda: 
                    for k in data.keys():
                        data[k] = data[k].to('cuda')
                        
                _, details, _ = model.step(data)
                
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
                
                if i == 0:
                    origins_one = np.arange(num_nodes)
                    object_stats += ([((o,int(d)),step) for o,d in zip(origins_one[(new_changes_gt).to(bool)], gt_tensor[(new_changes_gt).to(bool)])])
                routine_outputs[step, :][new_changes_out] = deepcopy(output_tensor[new_changes_out])
                routine_output_step[step, :][new_changes_out] = i
                routine_ground_truths[step, :][new_changes_gt] = deepcopy(gt_tensor[new_changes_gt])
                routine_ground_truth_step[step, :][new_changes_gt] = i
                routine_change_types[step,:][new_changes_gt] = deepcopy(data['change_type'].cpu().to(int)[details['evaluate_node']])[new_changes_gt]
                changes_output_all[step,:] = deepcopy(np.bitwise_or(changes_output_all[step,:], new_changes_out)).to(bool)
                changes_gt_all[step,:] = deepcopy(np.bitwise_or(changes_gt_all[step,:], new_changes_gt)).to(bool)

                output_conf = (output_probs*(torch.nn.functional.one_hot(output_tensor, num_classes=output_probs.size()[-1]))).sum(-1)*(routine_inputs[step,:]!=output_tensor).to(float)
                
                for conf in confidences:
                    new_changes_conf = deepcopy(np.bitwise_and((output_conf > conf), np.bitwise_not(routine_outputs_conf[conf]['step'][step, :]<lookahead_steps))).to(bool)
                    routine_outputs_conf[conf]['output'][step, :][new_changes_conf] = deepcopy(output_tensor[new_changes_conf])
                    routine_outputs_conf[conf]['step'][step, :][new_changes_conf] = i
                

                if deterministic_input_loop:
                    prev_edges = one_hot(details['output']['location'], num_classes = details['output']['location'].size()[-1]).to(torch.float32)
                else:
                    prev_edges = (details['output_probs']['location']).to(torch.float32)
        

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
                results[conf]['moved']['correct'][ls] += int((correct[conf][changes_output_and_gt]).sum())
                results[conf]['moved']['wrong'][ls] += int((wrong[conf][changes_output_and_gt]).sum())
                results[conf]['moved']['missed'][ls] += int((changes_not_output_and_gt).sum())
                results[conf]['unmoved']['fp'][ls] += int(changes_output_and_not_gt.sum())
                results[conf]['unmoved']['tn'][ls] += int(deepcopy(np.bitwise_and(np.bitwise_not(changes_output_for_step), np.bitwise_not(changes_gt_for_step))).sum())
                

        correct = deepcopy(routine_outputs == routine_ground_truths).to(int)
        wrong = deepcopy(routine_outputs != routine_ground_truths).to(int)

        for ls in range(lookahead_steps):
            changes_output_for_step = deepcopy(routine_output_step  <= ls)
            changes_gt_for_step = deepcopy(routine_ground_truth_step  <= ls)
            changes_output_and_gt = deepcopy(np.bitwise_and(changes_output_for_step, changes_gt_for_step))
            changes_output_and_not_gt = deepcopy(np.bitwise_and(changes_output_for_step, np.bitwise_not(changes_gt_for_step)))
            changes_not_output_and_gt = deepcopy(np.bitwise_and(np.bitwise_not(changes_output_for_step), changes_gt_for_step))
            changes_tn = deepcopy(np.bitwise_and(np.bitwise_not(changes_output_for_step), np.bitwise_not(changes_gt_for_step)))
            origins = np.ones_like(routine_ground_truths) * np.arange(num_nodes)
            results_by_obj[ls]['correct'] += ([(o,int(d)) for o,d in zip(origins[(correct * changes_output_and_gt).to(bool)], routine_ground_truths[(correct * changes_output_and_gt).to(bool)])])
            results_by_obj[ls]['wrong'] += ([(o,int(d)) for o,d in zip(origins[(wrong * changes_output_and_gt).to(bool)], routine_ground_truths[(wrong * changes_output_and_gt).to(bool)])])
            results_by_obj[ls]['missed'] += ([(o,int(d)) for o,d in zip(origins[(changes_not_output_and_gt).to(bool)], routine_ground_truths[(changes_not_output_and_gt).to(bool)])])
            results_by_obj[ls]['fp'] += ([(o,int(d)) for o,d in zip(origins[(changes_output_and_not_gt).to(bool)], routine_ground_truths[(changes_output_and_not_gt).to(bool)])])
            
        routine_change_types = routine_change_types.to(int)
        assert torch.equal(routine_output_step<lookahead_steps, changes_output_all)
        for ct in range(num_change_types):
            ct_mask = deepcopy(np.bitwise_and(changes_output_all, routine_change_types == (ct+1)))
            results_other['moved_change_type']['correct'][ct] += int((correct[ct_mask]).sum())
            results_other['moved_change_type']['wrong'][ct] += int((wrong[ct_mask]).sum())
            ct_mask_missed = deepcopy(np.bitwise_and(np.bitwise_not(changes_output_all), routine_change_types == (ct+1)))
            results_other['moved_change_type']['missed'][ct] += int(ct_mask_missed.sum())
        # print (routine_change_types.unique())
        # assert abs(sum([results['completeness_breakdown']['by_change_type'][ct][2] for ct in range(num_change_types)]) - results['completeness_breakdown']['by_lookahead'][-1][2]) == 0 , "missed changes don't add up!"
        # assert abs(sum([results['completeness_breakdown']['by_change_type'][ct][0] for ct in range(num_change_types)]) - results['completeness_breakdown']['by_lookahead'][-1][0]) == 0, "correct changes don't add up"
        # assert abs(sum([results['completeness_breakdown']['by_change_type'][ct][1] for ct in range(num_change_types)]) - results['completeness_breakdown']['by_lookahead'][-1][1]) == 0, "wrong changes don't add up"

        
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

    results['num_steps'] = total_num_steps

    return results, raw_data, figures + figures_imp + obj_eval_figs


def evaluate(model, data, output_dir, print_importance=False, lookahead_steps=12, confidences=[], learned_model=False):
    
    info, raw_data, figures = evaluate_all_breakdowns(model, data.test_routines, node_names=data.common_data['node_classes'], print_importance=print_importance, lookahead_steps=lookahead_steps, confidences=confidences, learned_model=learned_model)
    json.dump(info, open(os.path.join(output_dir, 'evaluation.json'), 'w'), indent=4)

    torch.save(raw_data, os.path.join(output_dir, 'raw_data.pt'))

    if os.path.exists(os.path.join(output_dir,'figures')):
        shutil.rmtree(os.path.join(output_dir,'figures'))
    os.makedirs(os.path.join(output_dir,'figures'))
    for i,fig  in enumerate(figures):
        fig.savefig(os.path.join(output_dir,'figures',str(i)+'.jpg'))

    plt.close('all')

    return info