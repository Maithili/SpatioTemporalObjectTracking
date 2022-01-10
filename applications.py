import numpy as np
from reader import CollateToDict
from GraphTranslatorModule import _erase_edges

def multiple_steps(model, test_routines, unconditional=False):
    accuracies = []
    for routine in list(test_routines):
        n_step = 0
        while len(routine) > 0:
            data = routine.pop()
            if n_step == 0 and unconditional:
                data['edges'] = _erase_edges(data['edges'])
            if n_step > 0:
                data['edges'] = details['output_probs']['location']
            eval, details = model.step(test_routines.collate_fn([data]))
            if len(accuracies)>=n_step:
                accuracies[n_step].append(eval['accuracy'])
            else:
                accuracies.append([eval['accuracy']])
            n_step += 1
    return accuracies

def object_search(model, test_routines, object_ids_to_search, dict_node_idx_from_id):
    total_guesses = [0]*len(object_ids_to_search)
    hits = [[0,0,0]]*len(object_ids_to_search)
    nodes = np.arange(len(dict_node_idx_from_id))
    for routine in list(test_routines):
        while len(routine) > 0:
            data = routine.pop()
            eval, details = model.step(test_routines.collate_fn([data]))
            for i,obj_id in enumerate(object_ids_to_search):
                obj_idx = dict_node_idx_from_id[obj_id]
                guesses = sorted(zip(details['output_probs']['location'][0,obj_idx,:], nodes), reverse=True)[:3]
                guesses = [g[1] for g in guesses]
                guess_hit = details['gt']['location'][0, obj_idx] == guesses
                hits[i][guess_hit] += 1
                total_guesses[i] += 1
    return hits, total_guesses

def something(model, test_routines):
    for routine in test_routines:
        while len(routine) > 0:
            data = routine.pop()
            eval, details = model.step(data)
    return 