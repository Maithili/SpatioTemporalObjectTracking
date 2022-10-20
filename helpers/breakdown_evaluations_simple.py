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
import torch.nn.functional as F
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


def evaluate_all_breakdowns(model, test_routines, lookahead_steps=6, deterministic_input_loop=False, node_names=[], print_importance=False, confidences=[], learned_model=False):
    
    use_cuda = torch.cuda.is_available() and learned_model
    if use_cuda: model.to('cuda')
    elif learned_model: print(f'Learned Model NOT USING CUDA. THIS WILL TAKE AGESSSSS!!!!!!!!!!!!')

    raw_data = {'inputs':[], 'outputs':[], 'ground_truths':[], 'futures':[]}
    results = {'moved':{'correct':[0 for _ in range(lookahead_steps)], 
                            'wrong':[0 for _ in range(lookahead_steps)], 
                            'missed':[0 for _ in range(lookahead_steps)]},
                 'unmoved':{'fp':[0 for _ in range(lookahead_steps)], 
                            'tn':[0 for _ in range(lookahead_steps)]}
                }

    figures = []

    results['all_moves'] = []
    results['num_changes'] = [[] for _ in range(lookahead_steps)]

    for routine in test_routines:

        routine_length = routine[0]-3-lookahead_steps
        num_nodes = routine[1].size()[-2]

        routine_inputs = torch.empty(routine_length, num_nodes)
        routine_outputs = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_output_step = torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)
        routine_ground_truths = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_ground_truth_step = torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)

        changes_output_all = torch.zeros(routine_length, num_nodes).to(bool)
        changes_gt_all = torch.zeros(routine_length, num_nodes).to(bool)

        # play_forward_edges = test_routines.collate_fn([routine[0]])['edges']

        data = test_routines.collate_fn([routine])
            
        if use_cuda: 
            for k in data.keys():
                data[k] = data[k].to('cuda')
                    
        model_results = model.evaluate_prediction(data, num_steps=lookahead_steps)

        routine_inputs = data['edges'][:,2:-1-lookahead_steps,:,:].argmax(-1).squeeze(0).cpu()
        for step, result in enumerate(model_results['object']):
            gt_tensor = data['edges'][:,3+step:-lookahead_steps+step,:,:].argmax(-1).squeeze(0).cpu()
            output_probs = result[:,:routine_inputs.size()[0],:,:].squeeze(0).cpu()
            output_tensor = result[:,:routine_inputs.size()[0],:,:].argmax(-1).squeeze(0).cpu()
            
            new_changes_out = deepcopy(np.bitwise_and(output_tensor != routine_inputs, np.bitwise_not(changes_output_all))).to(bool)
            new_changes_gt = deepcopy(np.bitwise_and(gt_tensor != routine_inputs, np.bitwise_not(changes_gt_all[step,:]))).to(bool)
            
            routine_outputs[new_changes_out] = output_tensor[new_changes_out]
            routine_output_step[new_changes_out] = step
            routine_ground_truths[new_changes_gt] = gt_tensor[new_changes_gt]
            routine_ground_truth_step[new_changes_gt] = step

        correct = deepcopy(routine_outputs == routine_ground_truths).to(int)
        wrong = deepcopy(routine_outputs != routine_ground_truths).to(int)
        
        for ls in range(lookahead_steps):
            changes_output_for_step = deepcopy(routine_output_step  <= ls)
            changes_gt_for_step = deepcopy(routine_ground_truth_step  <= ls)
            changes_output_and_gt = deepcopy(np.bitwise_and(changes_output_for_step, changes_gt_for_step))
            changes_output_and_not_gt = deepcopy(np.bitwise_and(changes_output_for_step, np.bitwise_not(changes_gt_for_step)))
            changes_not_output_and_gt = deepcopy(np.bitwise_and(np.bitwise_not(changes_output_for_step), changes_gt_for_step))
            results['moved']['correct'][ls] += int((correct[changes_output_and_gt]).sum())
            results['moved']['wrong'][ls] += int((wrong[changes_output_and_gt]).sum())
            results['moved']['missed'][ls] += int((changes_not_output_and_gt).sum())
            results['unmoved']['fp'][ls] += int(changes_output_and_not_gt.sum())
            results['unmoved']['tn'][ls] += int(deepcopy(np.bitwise_and(np.bitwise_not(changes_output_for_step), np.bitwise_not(changes_gt_for_step))).sum())
                    

        fig, axs = plt.subplots(1,4)
        fig.set_size_inches(30,20)

        labels = []
        if len(node_names) > 0:
            labels = node_names # [name for name,active in zip(node_names, additional_info['active_nodes']) if active]

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

        # ax = axs[4]
        # img_ct = routine_change_types
        # ax.imshow(img_ct, cmap=newcmp, vmin=0, vmax=3, aspect='auto')
        # ax.set_title('Ground Truths\n w/ Change Types')
        # ax.set_xticks(np.arange(num_nodes))
        # ax.set_xticklabels(labels, rotation=90)

        fig.tight_layout()
        figures.append(fig)

        raw_data['inputs'].append(routine_inputs)
        raw_data['outputs'].append(routine_outputs)
        raw_data['ground_truths'].append(routine_ground_truths)

        del data
    
    return results, raw_data, figures


def evaluate(model, data, output_dir, print_importance=False, lookahead_steps=6, confidences=[], learned_model=False):
    
    model.evaluate = True

    info, raw_data, figures = evaluate_all_breakdowns(model, data.get_test_split(), node_names=data.common_data['node_classes'], print_importance=print_importance, lookahead_steps=lookahead_steps, confidences=confidences, learned_model=learned_model)
    json.dump(info, open(os.path.join(output_dir, 'evaluation.json'), 'w'), indent=4)

    torch.save(raw_data, os.path.join(output_dir, 'raw_data.pt'))

    if os.path.exists(os.path.join(output_dir,'figures')):
        shutil.rmtree(os.path.join(output_dir,'figures'))
    os.makedirs(os.path.join(output_dir,'figures'))
    for i,fig  in enumerate(figures):
        fig.savefig(os.path.join(output_dir,'figures',str(i)+'.jpg'))

    plt.close('all')

    return info