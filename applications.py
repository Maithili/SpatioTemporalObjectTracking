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

def multiple_steps(model, test_routines, unconditional=False):
    accuracies = []
    for (routine, additional_info) in test_routines:
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

def unconditional_accuracy(model, test_routines):
    accuracies = []
    for (routine, additional_info) in test_routines:
        while len(routine) > 0:
            data = test_routines.collate_fn([routine.pop()])
            data['edges'] = _erase_edges(data['edges'])
            eval, details = model.step(data)
            accuracies.append(eval['accuracy'])
    avg_accuracy = float(sum(accuracies)/len(accuracies))
    return avg_accuracy

def object_search(model, test_routines, object_ids_to_search, dict_node_idx_from_id, lookahead_steps, deterministic_input_loop):
    total_guesses = 0.000001
    hits = [[0,0,0] for _ in range(len(object_ids_to_search))]
    num_hit_counts = len(hits[0])
    for (routine, additional_info) in test_routines:
        for i in range(len(routine) - lookahead_steps):
            prev_edges = test_routines.collate_fn([routine[i]])['edges']
            for j in range(lookahead_steps):
                data = test_routines.collate_fn([routine[i+j]])
                data['edges'] = prev_edges
                _, details = model.step(data)
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
    if not os.path.exists(action_dir):
        os.makedirs(action_dir)
    actions_with_eval = [{} for _  in test_routines]
    action_types = ['proactive', 'restorative']
    summary_eval = {act_typ:{'good':0,'bad':0,'total':0.0001} for act_typ in action_types}
    for routine_num,(routine, additional_info) in enumerate(test_routines):
        for i in range(len(routine)):
            initial_data = test_routines.collate_fn([routine[i]])
            _, details_initial = model.step(initial_data)
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
            
            # obj_idx_in_use = [dict_node_idx_from_id[o[0]] for o in additional_info[i]['obj_in_use']]
            obj_idx_in_use = []

            prev_edges = initial_data['edges']
            for j in range(min(lookahead_steps, len(routine)-i)):
                data = test_routines.collate_fn([routine[i+j]])
                data['edges'] = prev_edges
                _, details = model.step(data)
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
            _, details_unconditional = model.step(initial_data)
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



def complete_metrics(result, num_examples):

    result['precision'] = float(result['true_positive']/result['num_actual_changes'])
    result['accuracy_over_true_positive'] = float(result['true_positive_correctly_predicted']/result['true_positive'])
    result['accuracy_over_actual_changes'] = float(result['true_positive_correctly_predicted']/result['num_actual_changes'])

    if 'num_predicted_changes' in result.keys():
        result['recall'] = float(result['true_positive']/result['num_predicted_changes'])
        result['F1-score'] = 2*result['precision']*result['recall']/(result['precision']+result['recall'])
        result['accuracy_over_predicted_changes'] = float(result['true_positive_correctly_predicted']/result['num_predicted_changes'])
        result['num_predicted_changes'] = float(result['num_predicted_changes'] / num_examples)

    result['true_positive'] = float(result['true_positive'] / num_examples)
    result['num_actual_changes'] = float(result['num_actual_changes'] / num_examples)
    result['true_positive_correctly_predicted'] = float(result['true_positive_correctly_predicted'] / num_examples)

    return result


def complete_multistep_metrics(results):

    out_results = {}

    out_results['precision'] = float(results['prec_true_positive']/results['num_actual_changes'])
    out_results['accuracy_over_actual_changes'] = float(results['prec_true_positive_correctly_predicted']/results['num_actual_changes'])
    out_results['accuracy_over_true_positive'] = float(results['prec_true_positive_correctly_predicted']/results['prec_true_positive'])

    if 'recl_true_positive' in results.keys():
        out_results['recall'] = float(results['recl_true_positive']/results['num_predicted_changes'])
        out_results['F1-score'] = 2*out_results['precision']*out_results['recall']/(out_results['precision']+out_results['recall'])
        out_results['accuracy_over_predicted_changes'] = float(results['recl_true_positive_correctly_predicted']/results['num_predicted_changes'])

    # results['prec_true_positive'] = float(results['prec_true_positive']/num_examples)
    # results['prec_true_positive_correctly_predicted'] = float(results['prec_true_positive_correctly_predicted']/num_examples)
    # results['num_actual_changes'] = float(results['num_actual_changes']/num_examples)
    # results['recl_true_positive'] = float(results['recl_true_positive']/num_examples)
    # results['recl_true_positive_correctly_predicted'] = float(results['recl_true_positive_correctly_predicted']/num_examples)
    # results['num_predicted_changes'] = float(results['num_predicted_changes']/num_examples)

    return out_results


def collect_n_steps(model, data_list, n, deterministic_input_loop):
    collected = {}
    prev_edges = data_list[0]['edges']
    input_nodes = (data_list[0]['nodes']).argmax(-1)
    input_edges = (data_list[0]['edges']).squeeze(-1).argmax(-1)
    gt_changed = (input_edges*0).to(bool)
    out_changed = (input_edges*0).to(bool)
    collected['input'] = {'class':deepcopy(input_nodes),'location':deepcopy(input_edges)}
    collected['output'] = {'class':deepcopy(input_nodes),'location':deepcopy(input_edges)}
    collected['gt'] = {'class':deepcopy(input_nodes),'location':deepcopy(input_edges)}
    collected['change_type'] = (input_edges*0).to(int)
    for data in data_list:
        data['edges'] = prev_edges
        _, details = model.step(data)
        gt_tensor, output_tensor, input_tensor = details['gt']['location'], details['output']['location'], details['input']['location']
        masks = _get_masks(gt_tensor, output_tensor, input_tensor)

        gt_mask = np.bitwise_and(masks['gt_positives'], np.bitwise_not(gt_changed).to(bool)).to(bool)
        collected['gt']['location'][gt_mask] = deepcopy(gt_tensor[gt_mask])
        collected['change_type'][gt_mask] = deepcopy(data['change_type'][gt_mask].to(int))
        gt_changed = np.bitwise_or(gt_changed, masks['gt_positives']).to(bool)
        out_mask = np.bitwise_and(masks['out_positives'], np.bitwise_not(out_changed).to(bool)).to(bool)
        collected['output']['location'][out_mask] = deepcopy(output_tensor[out_mask])
        out_changed = np.bitwise_or(out_changed, masks['out_positives']).to(bool)

        if deterministic_input_loop:
            prev_edges = one_hot(details['output']['location'], num_classes = details['output']['location'].size()[-1])
        else:
            prev_edges = details['output_probs']['location']
    
    collected['evaluate_node'] = details['evaluate_node']
    return collected

def evaluate_precision_recall_loosely(model, test_routines, lookahead_steps=5):
    metrics = {}

    metrics['all'] = {}
    metrics['all']['deterministic'] = {'prec_true_positive' : 0.0, 'prec_true_positive_correctly_predicted': 0.0}
    metrics['all']['stochastic'] = {'prec_true_positive' : 0.0, 'prec_true_positive_correctly_predicted': 0.0}
    metrics['all']['common'] = {'num_actual_changes' : 0.0, 'num_predicted_changes' : 0.0, 'recl_true_positive' : 0.0, 'recl_true_positive_correctly_predicted': 0.0}

    metrics['take_out'] = {}
    metrics['take_out']['deterministic'] = {'prec_true_positive' : 0.0, 'prec_true_positive_correctly_predicted': 0.0}
    metrics['take_out']['stochastic'] = {'prec_true_positive' : 0.0, 'prec_true_positive_correctly_predicted': 0.0}
    metrics['take_out']['common'] = {'num_actual_changes' : 0.0}
    
    metrics['put_away'] = deepcopy(metrics['take_out'])
    metrics['other'] = deepcopy(metrics['take_out'])

    num_examples = 0
    for (routine, additional_info) in test_routines:
        while len(routine) > 0:
            num_examples += 1
            data_list = [test_routines.collate_fn([routine[j]]) for j in range(min(lookahead_steps, len(routine)))]
            eval, one_step_details = model.step(test_routines.collate_fn([routine.pop()]))
            details_deterministic = collect_n_steps(model, data_list, lookahead_steps, deterministic_input_loop=True)
            details_stochastic = collect_n_steps(model, data_list, lookahead_steps, deterministic_input_loop=False)
            
            input_tensor = one_step_details['input']['location'][one_step_details['evaluate_node']]
            gt_tensor = details_deterministic['gt']['location'][details_deterministic['evaluate_node']]
            output_tensor_deterministic = details_deterministic['output']['location'][details_deterministic['evaluate_node']]
            output_tensor_stochastic = details_stochastic['output']['location'][details_stochastic['evaluate_node']]
            one_step_gt_tensor = one_step_details['gt']['location'][one_step_details['evaluate_node']]
            one_step_output_tensor = one_step_details['output']['location'][one_step_details['evaluate_node']]
            change_types = details_deterministic['change_type']
            change_type_masks = {}
            change_type_masks['take_out'] = ((change_types == 1).to(bool))[one_step_details['evaluate_node']]
            change_type_masks['other'] = ((change_types == 2).to(bool))[one_step_details['evaluate_node']]
            change_type_masks['put_away'] = ((change_types == 3).to(bool))[one_step_details['evaluate_node']]

            prec_det_masks = _get_masks(one_step_gt_tensor, output_tensor_deterministic, input_tensor)
            prec_stoch_masks = _get_masks(one_step_gt_tensor, output_tensor_stochastic, input_tensor)
            rec_masks = _get_masks(gt_tensor, one_step_output_tensor, input_tensor)
            metrics['all']['deterministic']['prec_true_positive'] += prec_det_masks['tp'].sum()
            metrics['all']['deterministic']['prec_true_positive_correctly_predicted'] += (np.bitwise_and(prec_det_masks['tp'], prec_det_masks['correct'])).sum()
            metrics['all']['stochastic']['prec_true_positive'] += prec_stoch_masks['tp'].sum()
            metrics['all']['stochastic']['prec_true_positive_correctly_predicted'] += (np.bitwise_and(prec_stoch_masks['tp'], prec_stoch_masks['correct'])).sum()
            metrics['all']['common']['num_actual_changes'] += prec_det_masks['gt_positives'].sum()
            metrics['all']['common']['recl_true_positive'] += rec_masks['tp'].sum()
            metrics['all']['common']['recl_true_positive_correctly_predicted'] += (np.bitwise_and(rec_masks['tp'], rec_masks['correct'])).sum()
            metrics['all']['common']['num_predicted_changes'] += rec_masks['out_positives'].sum()

            for type in change_type_masks:
                metrics[type]['deterministic']['prec_true_positive'] += prec_det_masks['tp'][change_type_masks[type]].sum()
                metrics[type]['deterministic']['prec_true_positive_correctly_predicted'] += (np.bitwise_and(prec_det_masks['tp'], prec_det_masks['correct'])[change_type_masks[type]]).sum()
                metrics[type]['stochastic']['prec_true_positive'] += prec_stoch_masks['tp'][change_type_masks[type]].sum()
                metrics[type]['stochastic']['prec_true_positive_correctly_predicted'] += (np.bitwise_and(prec_stoch_masks['tp'], prec_stoch_masks['correct'])[change_type_masks[type]]).sum()
                metrics[type]['common']['num_actual_changes'] += prec_det_masks['gt_positives'][change_type_masks[type]].sum()

    result = {}

    for type, type_metrics in metrics.items():
        type_metrics['deterministic'].update(type_metrics['common'])
        type_metrics['stochastic'].update(type_metrics['common'])    
        result[type] = {}
        result[type]['deterministic'] = complete_multistep_metrics(type_metrics['deterministic'])
        result[type]['stochastic'] = complete_multistep_metrics(type_metrics['stochastic'])

    return result


def evaluate_precision_recall(model, test_routines):
    result = {}
    result['all'] = {'true_positive' : 0.0, 'num_actual_changes' : 0.0, 'num_predicted_changes' : 0.0, 'true_positive_correctly_predicted': 0.0}
    result['accuracy'] = 0.0
    result['take_out'] = {'true_positive' : 0.0, 'num_actual_changes' : 0.0, 'true_positive_correctly_predicted': 0.0}
    result['put_away'] = {'true_positive' : 0.0, 'num_actual_changes' : 0.0, 'true_positive_correctly_predicted': 0.0}
    result['other'] = {'true_positive' : 0.0, 'num_actual_changes' : 0.0, 'true_positive_correctly_predicted': 0.0}
    num_examples = 0
    for (routine, additional_info) in test_routines:
        while len(routine) > 0:
            num_examples += 1
            data = test_routines.collate_fn([routine.pop()])
            eval, details = model.step(data)
            gt_tensor, output_tensor, input_tensor = details['gt']['location'][details['evaluate_node']], details['output']['location'][details['evaluate_node']], details['input']['location'][details['evaluate_node']]
            masks = _get_masks(gt_tensor, output_tensor, input_tensor)
            mask_moving_back = ((data['change_type'] == 3).to(bool))[details['evaluate_node']]
            mask_moving_out = ((data['change_type'] == 1).to(bool))[details['evaluate_node']]
            mask_moving_other = ((data['change_type'] == 2).to(bool))[details['evaluate_node']]
            
            all_moving_masks = np.bitwise_or(np.bitwise_or(mask_moving_back, mask_moving_out), mask_moving_other).to(bool)
            assert all(np.equal(all_moving_masks, masks['gt_positives']))

            tp = masks['tp']
            correctly_predicted_tp = (np.bitwise_and(masks['tp'], masks['correct']))
            gt_positives = masks['gt_positives']
            out_positives = masks['out_positives']

            result['all']['true_positive'] += tp.sum()
            result['all']['num_actual_changes'] += gt_positives.sum()
            result['all']['num_predicted_changes'] += out_positives.sum()
            result['all']['true_positive_correctly_predicted'] += correctly_predicted_tp.sum()
            result['accuracy'] += masks['correct'].sum()/output_tensor.size(-1)

            result['take_out']['true_positive'] += tp[mask_moving_out].sum()
            result['take_out']['num_actual_changes'] += gt_positives[mask_moving_out].sum()
            result['take_out']['true_positive_correctly_predicted'] += correctly_predicted_tp[mask_moving_out].sum()

            result['put_away']['true_positive'] += tp[mask_moving_back].sum()
            result['put_away']['num_actual_changes'] += gt_positives[mask_moving_back].sum()
            result['put_away']['true_positive_correctly_predicted'] += correctly_predicted_tp[mask_moving_back].sum()

            result['other']['true_positive'] += tp[mask_moving_other].sum()
            result['other']['num_actual_changes'] += gt_positives[mask_moving_other].sum()
            result['other']['true_positive_correctly_predicted'] += correctly_predicted_tp[mask_moving_other].sum()

    result['accuracy'] = float(result['accuracy'])/num_examples

    result['all'] = complete_metrics(result['all'], num_examples)
    result['take_out'] = complete_metrics(result['take_out'], num_examples)
    result['put_away'] = complete_metrics(result['put_away'], num_examples)
    result['other'] = complete_metrics(result['other'], num_examples)
    
    return result

def pointwise_precision_recall(model, test_routines):
    result = []
    result_typed = {'take_out': [], 'put_away': [], 'other' : []}
    times = []
    for (routine, additional_info) in test_routines:
        while len(routine) > 0:
            data = test_routines.collate_fn([routine.pop()])
            eval, details = model.step(data)
            gt_tensor, output_tensor, input_tensor = details['gt']['location'][details['evaluate_node']], details['output']['location'][details['evaluate_node']], details['input']['location'][details['evaluate_node']]
            masks = _get_masks(gt_tensor, output_tensor, input_tensor)
            tp = masks['tp']
            num_correctly_predicted_tp = (np.bitwise_and(masks['tp'], masks['correct']))
            num_gt_positives = masks['gt_positives']
            num_out_positives = masks['out_positives']
            ep = 0.000000001

            t = float(data['time'])
            result.append([t, float(tp.sum()/(num_gt_positives.sum() + ep)), 
                              float(tp.sum()/(num_out_positives.sum() + ep)), 
                              2*float(tp.sum())/(float(num_gt_positives.sum() + ep + num_out_positives.sum())), 
                              float(num_correctly_predicted_tp.sum()/(num_gt_positives.sum() + ep)), 
                              float(num_correctly_predicted_tp.sum()/(num_out_positives.sum() + ep)), 
                              float(masks['correct'].sum()/gt_tensor.size()[-1])])

            change_type_masks = {}
            change_type_masks['take_out'] = ((data['change_type'] == 1).to(bool))[details['evaluate_node']]
            change_type_masks['other'] = ((data['change_type'] == 2).to(bool))[details['evaluate_node']]
            change_type_masks['put_away'] = ((data['change_type'] == 3).to(bool))[details['evaluate_node']]

            for type in result_typed:
                m = change_type_masks[type]
                result_typed[type].append([t, float(tp[m].sum()/(num_gt_positives[m].sum() + ep)), 
                              float(tp[m].sum()/(num_out_positives[m].sum() + ep)), 
                              2*float(tp[m].sum())/(float(num_gt_positives[m].sum() + num_out_positives[m].sum() + ep)), 
                              float(num_correctly_predicted_tp[m].sum()/(num_gt_positives[m].sum() + ep)), 
                              float(num_correctly_predicted_tp[m].sum()/(num_out_positives[m].sum() + ep))])
            times.append(t)

    plots = {}
    cols = ['time', 'precision', 'recall', 'F-1 score', 'accuracy_moved_objects', 'accuracy_predicted_moved_objects']
    times = list(set(times))
    times.sort()
    avg_result = [[t, np.mean([datapt[-1] for datapt in result if datapt[0] == t and ~isnan(datapt[-1])])] for t in times]
    plots['line_accuracy'] = wandb.plot.line(wandb.Table(data=avg_result, columns = ["time", "accuracy"]), "time", ['time','accuracy'], title='average_accuracy')
    for i,col in enumerate(cols):
        avg_result = [[t, np.mean([datapt[i] for datapt in result if datapt[0] == t and ~isnan(datapt[i])])] for t in times]
        plots['line_'+col] = wandb.plot.line(wandb.Table(data=avg_result, columns = ["time", col]), "time", cols+['accuracy'], title='average_'+col)
        for type,res in result_typed.items():
            avg_result = [[t, np.mean([datapt[i] for datapt in res if datapt[0] == t and ~isnan(datapt[i])])] for t in times]
            plots['line_'+type+'_'+col] = wandb.plot.line(wandb.Table(data=avg_result, columns = ["time", col]), "time", cols+['accuracy'], title='average_'+type+'_'+col)

    # plots['scatter_precision'] = wandb.plot.scatter(wandb.Table(data=result, columns = cols), "time", "precision", title='scatter_precision')
    # plots['scatter_recall'] = wandb.plot.scatter(wandb.Table(data=result, columns = cols), "time", "recall", title='scatter_recall')
    # plots['scatter_F1-score'] = wandb.plot.scatter(wandb.Table(data=result, columns = cols), "time", "F-1 score", title='scatter_F1-score')
    # plots['scatter_accuracy_moved_objects'] = wandb.plot.scatter(wandb.Table(data=result, columns = cols), "time", "accuracy_moved_objects", title='scatter_accuracy_moved_objects')
    # plots['scatter_accuracy_predicted_moved_objects'] = wandb.plot.scatter(wandb.Table(data=result, columns = cols), "time", "accuracy_predicted_moved_objects", title='scatter_accuracy_predicted_moved_objects')
    # plots['accuracy'] = wandb.plot.scatter(wandb.Table(data=result, columns = cols), "time", "accuracy", title='scatter_accuracy')
    # plots['precision-recall'] = wandb.plot.scatter(wandb.Table(data=result, columns = cols), "precision", "recall", title='scatter_precision-recall')

    return plots

def something(model, test_routines):
    for (routine, additional_info) in test_routines:
        while len(routine) > 0:
            data = test_routines.collate_fn([routine.pop()])
            eval, details = model.step(data)
    return 



def evaluate_applications(model, data, cfg, output_dir, logger=None):
    evaluation = {}
    print('ACTIONS')
    evaluation['Actions'] = get_actions(model, deepcopy(data.test_routines), data.node_classes, os.path.join(output_dir, 'actions'), data.node_idx_from_id, lookahead_steps=cfg['PROACTIVE_LOOKAHEAD_STEPS'], action_probability_thresh=cfg['ACTION_PROBABILITY_THRESHOLDS'], deterministic_input_loop=cfg['DETERMINISTIC_INPUT_LOOP'])
    hit_ratios, _ = object_search(model, deepcopy(data.test_routines), cfg['DATA_INFO']['search_object_ids'], data.node_idx_from_id, lookahead_steps=cfg['SEARCH_LOOKAHEAD_STEPS'], deterministic_input_loop=cfg['DETERMINISTIC_INPUT_LOOP'])
    evaluation['Search hits'] = tuple(hit_ratios)
    evaluation['Conditional accuracy drift'] = tuple(multiple_steps(model, deepcopy(data.test_routines)))
    evaluation['Un-Conditional accuracy drift'] = tuple(multiple_steps(model, deepcopy(data.test_routines), unconditional=True))
    print('PREC REC')
    evaluation['1step'] = evaluate_precision_recall(model, deepcopy(data.test_routines))
    print('PREC REC UNCON')
    evaluation['unconditional_graph_accuracy'] = unconditional_accuracy(model, deepcopy(data.test_routines))
    print('PREC REC LOOSE')
    evaluation['5step'] = evaluate_precision_recall_loosely(model, deepcopy(data.test_routines))
    evaluation['3step'] = evaluate_precision_recall_loosely(model, deepcopy(data.test_routines), lookahead_steps=3)
    print('POINTWISE')
    pointwise_scatter_plots = pointwise_precision_recall(model, deepcopy(data.test_routines))
    # print(evaluation)
    print('EVAL_ED')
    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
        json.dump(evaluation,f)

    evaluation_summary = {'Eval':
                         {
                            'actions':
                                {'all':{'good':(evaluation['Actions']['proactive']['good']+evaluation['Actions']['restorative']['good'])/(evaluation['Actions']['proactive']['total']+evaluation['Actions']['restorative']['total']), 
                                        'bad':(evaluation['Actions']['proactive']['bad']+evaluation['Actions']['restorative']['bad'])/(evaluation['Actions']['proactive']['total']+evaluation['Actions']['restorative']['total']),
                                        'good num':(evaluation['Actions']['proactive']['good']+evaluation['Actions']['restorative']['good']), 
                                        'bad num':(evaluation['Actions']['proactive']['bad']+evaluation['Actions']['restorative']['bad']),
                                        'num total':evaluation['Actions']['proactive']['total']+evaluation['Actions']['restorative']['total']},
                                'proactive %':{'good %':evaluation['Actions']['proactive']['good']/evaluation['Actions']['proactive']['total'], 
                                                'bad %':evaluation['Actions']['proactive']['bad']/evaluation['Actions']['proactive']['total'],
                                                'good num':evaluation['Actions']['proactive']['good'], 
                                                'bad num':evaluation['Actions']['proactive']['bad'],
                                                'num total':evaluation['Actions']['proactive']['total']},
                                'restorative':{'good %':evaluation['Actions']['restorative']['good']/evaluation['Actions']['restorative']['total'], 
                                                'bad %':evaluation['Actions']['restorative']['bad']/evaluation['Actions']['restorative']['total'],
                                                'good num':evaluation['Actions']['restorative']['good'], 
                                                'bad num':evaluation['Actions']['restorative']['bad'],
                                                'num total':evaluation['Actions']['restorative']['total']}
                                },
                            'object_search':{'1-hit':sum([h[0] for h in hit_ratios])/len(hit_ratios),
                                            '2-hit':sum([h[1] for h in hit_ratios])/len(hit_ratios),
                                            '3-hit':sum([h[2] for h in hit_ratios])/len(hit_ratios)},
                            'graph_pred_1step':evaluation['1step'],
                            'graph_pred_5step':evaluation['5step'],
                            'graph_pred_3step':evaluation['3step']
                            }
                         }

    if logger is None:
        logger = wandb

    logger.log(evaluation_summary)
    for name, scatterplot in pointwise_scatter_plots.items():
        logger.log({'pointwise_'+name : scatterplot})

    logger.log({'precision-recall' : wandb.plot.scatter(wandb.Table(data=[[evaluation['1step']['all']['precision'], evaluation['1step']['all']['recall']]], columns = ["precision", "recall"]), "precision", "recall", title="precision-recall-1step")})
    # logger.log({'precision-recall-5step-deterministic' : wandb.plot.scatter(wandb.Table(data=[[evaluation['5step']['all']['deterministic']['precision'], evaluation['5step']['all']['deterministic']['recall']]], columns = ["precision", "recall"]), "precision", "recall", title="precision-recall-5step-deterministic")})
    # logger.log({'precision-recall-5step-stochastic' : wandb.plot.scatter(wandb.Table(data=[[evaluation['5step']['all']['stochastic']['precision'], evaluation['5step']['stochastic']['all']['recall']]], columns = ["precision", "recall"]), "precision", "recall", title="precision-recall-5step-stochastic")})
    # logger.log({'precision-recall-3step-deterministic' : wandb.plot.scatter(wandb.Table(data=[[evaluation['3step']['all']['deterministic']['precision'], evaluation['3step']['all']['deterministic']['recall']]], columns = ["precision", "recall"]), "precision", "recall", title="precision-recall-3step-deterministic")})
    # logger.log({'precision-recall-3step-stochastic' : wandb.plot.scatter(wandb.Table(data=[[evaluation['3step']['all']['stochastic']['precision'], evaluation['3step']['stochastic']['all']['recall']]], columns = ["precision", "recall"]), "precision", "recall", title="precision-recall-3step-stochastic")})

    return evaluation_summary