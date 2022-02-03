import torch
import numpy as np


def _get_accuracy_masks(gt_tensor, output_tensor):
    masks = {}
    masks['correct'] = gt_tensor == output_tensor
    masks['wrong'] = gt_tensor != output_tensor
    return masks

def _get_confusion_matrix_masks(gt_tensor, output_tensor, input_tensor):
    masks = {}
    masks['gt_negatives'] = gt_tensor == input_tensor
    masks['gt_positives'] = gt_tensor != input_tensor
    masks['out_negatives'] = output_tensor == input_tensor
    masks['out_positives'] = output_tensor != input_tensor
    masks['tp'] = np.bitwise_and(masks['out_positives'], masks['gt_positives'])
    masks['fp'] = np.bitwise_and(masks['out_positives'], masks['gt_negatives'])
    masks['tn'] = np.bitwise_and(masks['out_negatives'], masks['gt_negatives'])
    masks['fn'] = np.bitwise_and(masks['out_negatives'], masks['gt_positives'])
    return masks


def evaluate_accuracy(gt_tensor, loss_tensor, output_tensor):
    masks = _get_accuracy_masks(gt_tensor, output_tensor)
    result = {'accuracy':None ,'losses':{}}
    result['accuracy'] = (masks['correct'].sum())/torch.numel(gt_tensor)
    result['losses']['mean'] = loss_tensor.mean()
    result['losses']['correct'] = loss_tensor[masks['correct']].sum()/masks['correct'].sum()
    result['losses']['wrong'] = loss_tensor[masks['wrong']].sum()/masks['wrong'].sum()
    return result

def evaluate_precision_recall(gt_tensor, output_tensor, input_tensor):
    masks = _get_confusion_matrix_masks(gt_tensor, output_tensor, input_tensor)
    result = {}
    result['change_prediction_precision'] = masks['tp'].sum()/masks['gt_positives'].sum()
    result['change_prediction_true_changes'] = masks['gt_positives'].sum()
    result['change_prediction_recall'] = masks['tp'].sum()/masks['out_positives'].sum()
    result['change_prediction_predicted_changes'] = masks['out_positives'].sum()
    result['correctly_predicted_changes_accuracy'] = (np.bitwise_and(masks['tp'], gt_tensor == output_tensor)).sum()/masks['tp'].sum()
    result['correctly_predicted_changes_of_actual_changes'] = (np.bitwise_and(masks['tp'], gt_tensor == output_tensor)).sum()/masks['gt_positives'].sum()
    result['correctly_predicted_changes_of_predicted_changes'] = (np.bitwise_and(masks['tp'], gt_tensor == output_tensor)).sum()/masks['out_positives'].sum()
    return result

def evaluate(gt, losses, output, input, evaluate_node):
    gt_tensor = gt[evaluate_node]
    output_tensor = output[evaluate_node]
    loss_tensor = losses[evaluate_node]
    input_tensor = input[evaluate_node]
    result = evaluate_accuracy(gt_tensor, loss_tensor, output_tensor)
    result['CM'] = evaluate_precision_recall(gt_tensor, output_tensor, input_tensor)
    return result