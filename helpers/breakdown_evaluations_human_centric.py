import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from torch.nn.functional import one_hot


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

def do_stuff(model, test_routines, node_names=[]):
    
    result_counts = {'correct':0, 'wrong':0, 'unmoved':0, 'final_unnecessarily_moved':0}
    figures = []

    for (routine, additional_info) in test_routines:

        routine_length = len(routine)
        num_nodes = additional_info['active_nodes'].sum()
        image_stepwise = np.zeros((routine_length, num_nodes))
        image_gt = np.zeros((routine_length, num_nodes))

        prev_edges = test_routines.collate_fn([routine[0]])['edges']
        for step, data_raw in enumerate(routine):
            data = test_routines.collate_fn([data_raw])

            dataset_input_full = data['edges'].squeeze(-1).argmax(-1)
            data['edges'] = prev_edges

            _, details = model.step(data)

            gt_tensor = details['gt']['location'][details['evaluate_node']]
            input_tensor = details['input']['location'][details['evaluate_node']]
            output_tensor = details['output']['location'][details['evaluate_node']]
            dataset_input = dataset_input_full[details['evaluate_node']]
            next_state = details['output']['location']
            next_state[details['gt']['location'] != dataset_input_full] = details['gt']['location'][details['gt']['location'] != dataset_input_full]

            gt_changes = (gt_tensor != dataset_input)
            correct = (output_tensor == gt_tensor)
            unmoved = (output_tensor == dataset_input)
            wrong = np.bitwise_and((output_tensor != gt_tensor), np.bitwise_not(unmoved))

            step_results = {}
            step_results['correct'] = (np.bitwise_and(correct, gt_changes))
            step_results['unmoved'] = (np.bitwise_and(unmoved, gt_changes))
            step_results['wrong'] = (np.bitwise_and(wrong, gt_changes))

            assert (np.bitwise_or(np.bitwise_or(step_results['correct'], step_results['unmoved']), step_results['wrong']) == gt_changes).all(), f"Did NOT classify GT changes properly from {gt_changes} to \n {step_results['correct']},\n {step_results['unmoved']},\n {step_results['wrong']}"

            image_stepwise[step,:] = (output_tensor != input_tensor).to(int)
            image_gt[step,:] = step_results['correct'].to(int) * 1 + step_results['unmoved'].to(int) * (-0.5) + step_results['wrong'].to(int) * (-1)

            for m in step_results:
                result_counts[m] += int(step_results[m].sum())

            prev_edges = one_hot(next_state, num_classes = details['output']['location'].size()[-1]).to(torch.float32)

        result_counts['final_unnecessarily_moved'] += int((next_state[details['evaluate_node']] != gt_tensor).sum())

        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(20,20)

        labels = []
        if len(node_names) > 0:
            labels = [name for name,active in zip(node_names, additional_info['active_nodes']) if active]

        ax = axs[0]
        ax.imshow(image_stepwise, cmap='Blues', vmin=0, vmax=2, aspect='auto')
        ax.set_title('Stepwise Output')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        ax = axs[1]
        ax.imshow(image_gt, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Ground Truth')
        ax.set_xticks(np.arange(num_nodes))
        ax.set_xticklabels(labels, rotation=90)

        fig.tight_layout()
        figures.append(fig)


    
    return result_counts, figures


def evaluate(model, data, output_dir):
    
    info, figures = do_stuff(model, data.test_routines, node_names=data.common_data['node_classes'])
    json.dump(info, open(os.path.join(output_dir, 'new_evaluation.json'), 'w'), indent=4)
    if os.path.exists(os.path.join(output_dir,'new_figures')):
        shutil.rmtree(os.path.join(output_dir,'new_figures'))
    os.makedirs(os.path.join(output_dir,'new_figures'))
    for i,fig  in enumerate(figures):
        fig.savefig(os.path.join(output_dir,'new_figures',str(i)+'.jpg'))

    plt.close('all')

    return info