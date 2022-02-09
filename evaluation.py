import torch
import numpy as np


def _get_accuracy_masks(gt_tensor, output_tensor):
    masks = {}
    masks['correct'] = gt_tensor == output_tensor
    masks['wrong'] = gt_tensor != output_tensor
    return masks

def _get_confusion_matrix_masks(gt_tensor, output_tensor, input_tensor):
    masks = {}
    masks['gt_negatives'] = (gt_tensor == input_tensor).cpu()
    masks['gt_positives'] = (gt_tensor != input_tensor).cpu()
    masks['out_negatives'] = (output_tensor == input_tensor).cpu()
    masks['out_positives'] = (output_tensor != input_tensor).cpu()
    masks['tp'] = np.bitwise_and(masks['out_positives'], masks['gt_positives'])
    masks['fp'] = np.bitwise_and(masks['out_positives'], masks['gt_negatives'])
    masks['tn'] = np.bitwise_and(masks['out_negatives'], masks['gt_negatives'])
    masks['fn'] = np.bitwise_and(masks['out_negatives'], masks['gt_positives'])
    return masks


def evaluate_accuracy(gt_tensor, loss_tensor, output_tensor, input_tensor, tn_loss_weight):
    masks = _get_accuracy_masks(gt_tensor, output_tensor)
    result = {}
    result['accuracy'] = (masks['correct'].sum())/torch.numel(gt_tensor)
    if loss_tensor is not None:
        result['losses'] = {}
        if tn_loss_weight is not None:
            cm_masks = _get_confusion_matrix_masks(gt_tensor, output_tensor, input_tensor)
            not_tn = np.bitwise_not(cm_masks['tn'])
            important_losses = loss_tensor[not_tn]
            unimportant_losses = loss_tensor[cm_masks['tn']]
            result['losses']['mean'] = important_losses.mean() + tn_loss_weight * unimportant_losses.mean()
            result['losses']['important'] = important_losses.mean()
            result['losses']['unimportant'] = unimportant_losses.mean()
        else:
            result['losses']['mean'] = loss_tensor.mean()
        result['losses']['correct'] = loss_tensor[masks['correct']].sum()/masks['correct'].sum()
        result['losses']['wrong'] = loss_tensor[masks['wrong']].sum()/masks['wrong'].sum()
    return result

def evaluate(gt, output, input, evaluate_node, losses=None, tn_loss_weight=None):
    gt_tensor = gt[evaluate_node]
    input_tensor = input[evaluate_node]
    output_tensor = output[evaluate_node]
    if losses is not None:
        loss_tensor = losses[evaluate_node]
        result = evaluate_accuracy(gt_tensor, loss_tensor, output_tensor, input_tensor, tn_loss_weight)
    else:
        result = evaluate_accuracy(gt_tensor, None, output_tensor, input_tensor, tn_loss_weight)
    # result['CM'] = evaluate_precision_recall(gt_tensor, output_tensor, input_tensor)
    return result