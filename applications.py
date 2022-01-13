import os
import numpy as np
from copy import deepcopy
from GraphTranslatorModule import _erase_edges
from encoders import human_readable_from_external

def multiple_steps(model, test_routines, unconditional=False):
    accuracies = []
    for routine in list(test_routines):
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
    avg_accuracy_stepwise = [np.mean(a) for a in accuracies]
    return avg_accuracy_stepwise

def object_search(model, test_routines, object_ids_to_search, dict_node_idx_from_id):
    total_guesses = 0
    hits = [[0,0,0] for _ in range(len(object_ids_to_search))]
    num_hit_counts = len(hits[0])
    for routine in list(test_routines):
        while len(routine) > 0:
            data = routine.pop()
            eval, details = model.step(test_routines.collate_fn([data]))
            for i,obj_id in enumerate(object_ids_to_search):
                obj_idx = dict_node_idx_from_id[obj_id]
                actual = int(details['gt']['location'][0, obj_idx])
                guess_rank = sum(details['output_probs']['location'][0,obj_idx,:] > details['output_probs']['location'][0,obj_idx,actual])
                if guess_rank < num_hit_counts:
                    hits[i][guess_rank] += 1
            total_guesses += 1
    def accumulate_hits(hit_count_in):
        hit_count_in.append(0)
        return [sum(hit_count_in[:i+1])/total_guesses for i in range(num_hit_counts)]
    cumulative_hit_ratio = [accumulate_hits(h) for h in hits]
    return cumulative_hit_ratio, total_guesses

class ChangePlanner():
    def __init__(self, initial_details):
        self.initial_locations = deepcopy(initial_details['input']['location'])
    def __call__(self, target_details):
        target_loc = target_details['output']['location']
        mask = np.bitwise_and(target_loc != self.initial_locations, target_details['evaluate_node'])
        return np.argwhere(mask)[1,:], (target_loc[mask]).view(-1)

def get_actions(model, test_routines, node_classes, action_dir, lookahead_steps = 5):
    os.makedirs(action_dir)
    for routine_num,routine in enumerate(test_routines):
        with open(os.path.join(action_dir,'{:03d}.txt'.format(routine_num)), 'w') as f:
            for i in range(len(routine) - lookahead_steps):
                initial_data = test_routines.collate_fn([routine[i]])
                eval, details_initial = model.step(initial_data)
                change_planner = ChangePlanner(details_initial)
                
                f.write('\n## {}\n'.format(human_readable_from_external(initial_data['timestamp'])))

                if i+lookahead_steps < len(routine):
                    prev_edges = initial_data['edges']
                    for j in range(lookahead_steps):
                        data = test_routines.collate_fn([routine[i+j]])
                        data['edges'] = prev_edges
                        eval, details = model.step(data)
                        prev_edges = details['output_probs']['location']
                    changes_obj, changes_loc = change_planner(details)
                    for co, cl in zip(changes_obj, changes_loc):
                        f.write('(Proactive)   '+ node_classes[co]+' to '+node_classes[cl]+'\n')
                
                data_in = initial_data
                data_in['edges'] = _erase_edges(data_in['edges'])
                eval, details_unconditional = model.step(initial_data)
                changes_obj, changes_loc = change_planner(details_unconditional)
                for co, cl in zip(changes_obj, changes_loc):
                    f.write('(Restorative) '+ node_classes[co]+' to '+node_classes[cl]+'\n')
    return 

def something(model, test_routines):
    for routine in test_routines:
        while len(routine) > 0:
            data = test_routines.collate_fn([routine.pop()])
            eval, details = model.step(data)
    return 