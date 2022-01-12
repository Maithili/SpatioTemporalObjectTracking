import numpy as np
from GraphTranslatorModule import _erase_edges

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

def something(model, test_routines):
    for routine in test_routines:
        while len(routine) > 0:
            data = test_routines.collate_fn([routine.pop()])
            eval, details = model.step(data)
    return 