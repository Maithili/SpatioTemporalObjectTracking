import os
import numpy as np
import torch
from torch.nn.functional import one_hot
from copy import deepcopy
from GraphTranslatorModule import _erase_edges
from encoders import human_readable_from_external

def multiple_steps(model, test_routines, unconditional=False):
    accuracies = []
    for (routine, additional_info) in list(test_routines):
        n_step = 0
        while len(routine) > 0:
            data = test_routines.collate_fn([routine.pop()])
            if n_step == 0 and unconditional:
                data['edges'] = _erase_edges(data['edges'])
            if n_step > 0:
                data['edges'] = details['output_probs']['location']
            eval, details = model.step(data)
            if len(accuracies)>n_step:
                accuracies[n_step].append(eval['accuracy'])
            else:
                accuracies.append([eval['accuracy']])
            n_step += 1
    avg_accuracy_stepwise = [float(np.mean(a)) for a in accuracies]
    return avg_accuracy_stepwise

def object_search(model, test_routines, object_ids_to_search, dict_node_idx_from_id, lookahead_steps, deterministic_input_loop):
    total_guesses = 0
    hits = [[0,0,0] for _ in range(len(object_ids_to_search))]
    num_hit_counts = len(hits[0])
    for (routine, additional_info) in list(test_routines):
        for i in range(len(routine) - lookahead_steps):
            prev_edges = test_routines.collate_fn([routine[i]])['edges']
            for j in range(lookahead_steps):
                data = test_routines.collate_fn([routine[i+j]])
                data['edges'] = prev_edges
                eval, details = model.step(data)
                if deterministic_input_loop:
                    prev_edges = one_hot(details['output']['location'], num_classes = details['output']['location'].size()[-1])
                else:
                    prev_edges = details['output_probs']['location']
            for i,obj_id in enumerate(object_ids_to_search):
                obj_idx = dict_node_idx_from_id[obj_id]
                actual = int(details['gt']['location'][0, obj_idx])
                guess_rank = sum(details['output_probs']['location'][0,obj_idx,:] > details['output_probs']['location'][0,obj_idx,actual])
                if guess_rank < num_hit_counts:
                    hits[i][guess_rank] += 1
            total_guesses += 1
    def accumulate_hits(hit_count_in):
        hit_count_in.append(0)
        return [float(sum(hit_count_in[:i+1])/total_guesses) for i in range(num_hit_counts)]
    cumulative_hit_ratio = [accumulate_hits(h) for h in hits]
    return cumulative_hit_ratio, total_guesses

class ChangePlanner():
    def __init__(self, initial_details, threshold=0.1):
        self.initial_locations = deepcopy(initial_details['input']['location'])
        self.threshold = threshold
    def __call__(self, target_details):
        target_loc = target_details['output']['location']
        current_loc_probs = torch.gather(target_details['output_probs']['location'][0,:,:], 1, self.initial_locations.transpose(1,0)).transpose(0,1)
        target_loc_probs = torch.gather(target_details['output_probs']['location'][0,:,:], 1, target_loc.transpose(1,0)).transpose(0,1)
        # mask = np.bitwise_and(target_loc != self.initial_locations, target_details['evaluate_node'])
        mask = np.bitwise_and(target_loc_probs - current_loc_probs > self.threshold, target_details['evaluate_node'])
        return np.argwhere(mask)[1,:], (target_loc[mask]).view(-1), (self.initial_locations[mask]).view(-1)

def get_actions(model, test_routines, node_classes, action_dir, dict_node_idx_from_id, lookahead_steps, action_probability_thresh, deterministic_input_loop):
    os.makedirs(action_dir)
    actions_with_eval = [{} for _  in test_routines]
    action_types = ['proactive', 'restorative']
    summary_eval = {act_typ:{'good':0,'bad':0,'total':0.0001} for act_typ in action_types}
    for routine_num,(routine, additional_info) in enumerate(test_routines):
        for i in range(len(routine)):
            initial_data = test_routines.collate_fn([routine[i]])
            eval, details_initial = model.step(initial_data)
            proactive_change_planner = ChangePlanner(details_initial, threshold=action_probability_thresh[0])
            reactive_change_planner = ChangePlanner(details_initial, threshold=action_probability_thresh[1])

            for prev_step_action_lists in actions_with_eval[routine_num].values():
                for act_typ in action_types:
                    for prev_action in prev_step_action_lists[act_typ]:
                        if prev_action['eval'] is None:
                            actual_location = details_initial['input']['location'][0,prev_action['object']]
                            if prev_action['to'] == actual_location:
                                prev_action['eval'] = 1
                                summary_eval[act_typ]['good'] += 1
                            elif prev_action['from'] != actual_location:
                                prev_action['eval'] = 0
            
            obj_idx_in_use = [dict_node_idx_from_id[o['id']] for o in additional_info[i]['obj_in_use']]

            prev_edges = initial_data['edges']
            for j in range(min(lookahead_steps, len(routine)-i)):
                data = test_routines.collate_fn([routine[i+j]])
                data['edges'] = prev_edges
                eval, details = model.step(data)
                if deterministic_input_loop:
                    prev_edges = one_hot(details['output']['location'], num_classes = details['output']['location'].size()[-1])
                else:
                    prev_edges = details['output_probs']['location']
            changes_obj, changes_to, changes_from = proactive_change_planner(details)
            proactive_actions = []
            for obj, to, fr in zip(changes_obj, changes_to, changes_from):
                string_action = node_classes[obj]+' from '+node_classes[fr]+' to '+node_classes[to]
                eval = -1 if obj in obj_idx_in_use else None
                proactive_actions.append({'object':int(obj), 'from':int(fr), 'to':int(to), 'string':string_action, 'eval':eval})
                if eval == -1 : summary_eval['proactive']['bad'] += 1
                summary_eval['proactive']['total'] += 1
            
            data_in = initial_data
            data_in['edges'] = _erase_edges(data_in['edges'])
            eval, details_unconditional = model.step(initial_data)
            changes_obj, changes_to, changes_from = reactive_change_planner(details_unconditional)
            restorative_actions = []
            for obj, to, fr in zip(changes_obj, changes_to, changes_from):
                string_action = node_classes[obj]+' from '+node_classes[fr]+' to '+node_classes[to]
                eval = -1 if obj in obj_idx_in_use else None
                restorative_actions.append({'object':int(obj), 'from':int(fr), 'to':int(to), 'string':string_action, 'eval':eval})
                if eval == -1 : summary_eval['restorative']['bad'] += 1
                summary_eval['restorative']['total'] += 1
        
            actions_with_eval[routine_num][human_readable_from_external(additional_info[i]['timestamp'])] = {'proactive':proactive_actions, 'restorative':restorative_actions, 'obj_in_use':additional_info[i]['obj_in_use']}

    for i,routine_actions in enumerate(actions_with_eval):
        with open(os.path.join(action_dir,'{:03d}.txt'.format(i)), 'w') as f:
            for timestamp, data in routine_actions.items():
                f.write('\n## {}\n'.format(timestamp))
                f.write('## {}\n'.format(data['obj_in_use']))
                f.write('Proactive\n')
                for act_pro in data['proactive']:
                    f.write(' - ('+str(act_pro['eval'])+') '+act_pro['string']+'\n')
                f.write('Restorative\n')
                for act_res in data['restorative']:
                    f.write(' - ('+str(act_res['eval'])+') '+act_res['string']+'\n')
                        
    return summary_eval

def something(model, test_routines):
    for (routine, additional_info) in test_routines:
        while len(routine) > 0:
            data = test_routines.collate_fn([routine.pop()])
            eval, details = model.step(data)
    return 