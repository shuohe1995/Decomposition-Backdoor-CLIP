import argparse
import numpy as np
import pickle
from utils.factory import create_model_and_transforms, get_tokenizer
from torch.nn import functional as F
import torch
from utils.mmd_loss import mmd
import pandas as pd
import os
import random
import einops
from utils.misc import accuracy
from torchmetrics import AUROC
import tqdm


def use_target_data(clean_data, bd_data, all_labels):
    target_index = []
    non_target_index = []
    for i in range(len(all_labels)):
        if all_labels[i] == args.target_label:
            target_index.append(i)
        else:
            non_target_index.append(i)
    target_clean_data = clean_data[target_index]
    target_bd_data = bd_data[non_target_index[:len(target_index)]]
    return target_clean_data, target_bd_data

def calculate_acc(representation, classifier, labels):
    
    return accuracy(torch.from_numpy(representation @ classifier).float(), labels)[0] * 100
def split_data( all_clean_attns, all_bd_attns, all_clean_cls_attns, all_bd_cls_attns, all_clean_mlp, all_bd_mlp, all_labels):

    val_clean_data = {}
    test_bd_data = {}
    test_clean_data = {}
    test_labels = {}

    num_data = len(all_clean_attns)
    num_val_data = int(num_data * args.val_ratio)
    index = [i for i in range(num_data)]
    random.shuffle(index)
    val_index, test_index = index[:num_val_data], index[num_val_data:]

    val_clean_data['attns'] = all_clean_attns[val_index]
    val_clean_data['cls_attns'] = all_clean_cls_attns[val_index]
    val_clean_data['mlp'] = all_clean_mlp[val_index]

    test_bd_data['attns'] = all_bd_attns[test_index]
    test_bd_data['mlp'] = all_bd_mlp[test_index]
    test_bd_data['cls_attns'] = all_bd_cls_attns[test_index]
    #test_bd_data['rep'] = all_bd_rep[test_index]

    test_clean_data['attns'] = all_clean_attns[test_index]
    test_clean_data['cls_attns'] = all_clean_cls_attns[test_index]
    test_clean_data['mlp'] = all_clean_mlp[test_index]
    #test_clean_data['rep'] = all_clean_rep[test_index]

    test_labels['clean_labels'] = all_labels[test_index]
    test_labels['target_labels'] = torch.ones(len(test_index))*args.target_label

    return val_clean_data, test_bd_data, test_clean_data, test_labels

def calculate_baseline(test_bd_data, test_clean_data, test_labels, classifier):

    baseline_bd = test_bd_data['attns'].sum(axis=(1, 2)) + test_bd_data['mlp'].sum(axis=1)
    baseline_clean = test_clean_data['attns'].sum(axis=(1, 2)) + test_clean_data['mlp'].sum(axis=1)
    baseline_asr = calculate_acc(baseline_bd, classifier, test_labels['target_labels'])
    baseline_acc = calculate_acc(baseline_clean, classifier, test_labels['clean_labels'])
    baseline_bd_acc = calculate_acc(baseline_bd, classifier, test_labels['clean_labels'])

    print("Baseline: ASR={}, ACC={}, BD-ACC={}".format(baseline_asr, baseline_acc, baseline_bd_acc))
    return


def calculate_mean_ablate_layer1(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier):

    # mean-ablate accumulated layers

    test_bd_layer = test_bd_data['attns'].sum(axis=2) + test_bd_data['mlp'][:, :-1]
    test_clean_layer = test_clean_data['attns'].sum(axis=2) + test_clean_data['mlp'][:, :-1]
    val_clean_layer = val_clean_data['attns'].sum(axis=2) + val_clean_data['mlp'][:, :-1]

    all_accuracies = []
    all_asr = []
    for layer in range(test_bd_data['attns'].shape[1]):
        clean_current_model = (
            np.sum(
                np.mean(val_clean_layer[:, :layer], axis=0, keepdims=True), axis=1
            )
            + np.mean(val_clean_layer[:, layer], axis=0, keepdims=True)
            + np.sum(test_clean_layer[:, layer + 1:], axis=1)
        )
        bd_current_model = (
            np.sum(
                np.mean(val_clean_layer[:, :layer], axis=0, keepdims=True), axis=1
            )
            + np.mean(val_clean_layer[:, layer], axis=0, keepdims=True)
            + np.sum(test_bd_layer[:, layer + 1:], axis=1)
        )
        current_accuracy = calculate_acc(test_clean_data['mlp'][:, -1] + clean_current_model, classifier, test_labels['clean_labels'])
        current_asr = calculate_acc(test_bd_data['mlp'][:, -1] + bd_current_model, classifier, test_labels['target_labels'])
        all_accuracies.append(current_accuracy)
        all_asr.append(current_asr)
    print("Forward Accumulated Layer ablation: asr={}, acc={}".format(all_asr, all_accuracies))

    all_accuracies2 = []
    all_asr2 = []
    for layer in range(test_bd_data['attns'].shape[1]):
        clean_current_model = (
            np.sum(
                np.mean(val_clean_layer[:, layer:], axis=0, keepdims=True), axis=1
            )
            + np.sum(test_clean_layer[:, :layer], axis=1)
        )
        bd_current_model = (
            np.sum(
                np.mean(val_clean_layer[:, layer:], axis=0, keepdims=True), axis=1
            )
            + np.sum(test_bd_layer[:, :layer], axis=1)
        )
        current_accuracy = (
            accuracy(
                torch.from_numpy((test_clean_data['mlp'][:, -1] + clean_current_model) @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
        )
        current_asr = (
            accuracy(
                torch.from_numpy((test_bd_data['mlp'][:, -1] + bd_current_model) @ classifier).float(),
                test_labels['target_labels'],
            )[0]
            * 100
        )
        all_accuracies2.append(current_accuracy)
        all_asr2.append(current_asr)
    print("Backward Accumulated Layer ablation: asr={}, acc={}".format(all_asr2, all_accuracies2))

    #mean-ablate certain layer

    all_accuracies3 = []
    all_asr3 = []

    for layer in range(test_bd_data['attns'].shape[1]):

        clean_current_model = np.sum(test_clean_layer, axis=1) - test_clean_layer[:, layer] + np.mean(val_clean_layer[:, layer], axis=0, keepdims=True)
        bd_current_model = np.sum(test_bd_layer, axis=1) - test_bd_layer[:, layer] + np.mean(val_clean_layer[:, layer], axis=0, keepdims=True)

        current_accuracy = (
            accuracy(
                torch.from_numpy((np.mean(val_clean_data['mlp'][:, -1], axis=0, keepdims=True) + clean_current_model) @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
        )
        current_asr = (
            accuracy(
                torch.from_numpy((np.mean(val_clean_data['mlp'][:, -1], axis=0, keepdims=True) + bd_current_model) @ classifier).float(),
                test_labels['target_labels'],
            )[0]
            * 100
        )
        all_accuracies3.append(current_accuracy)
        all_asr3.append(current_asr)
    print("Separate layer ablations: asr={}, acc={}".format(all_asr3, all_accuracies3))

    return
def calculate_mean_ablate_layer2(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier):


    all_accuracies = []
    all_asr = []
    for layer in range(test_bd_data['mlp'].shape[1]):
        clean_current_model = (
                np.sum(
                    np.mean(val_clean_data['mlp'][:, :layer], axis=0, keepdims=True), axis=1
                )
                + np.mean(val_clean_data['mlp'][:, layer], axis=0, keepdims=True)
                + np.sum(test_clean_data['mlp'][:, layer + 1:], axis=1)
        )
        bd_current_model = (
                np.sum(
                    np.mean(val_clean_data['mlp'][:, :layer], axis=0, keepdims=True), axis=1
                )
                + np.mean(val_clean_data['mlp'][:, layer], axis=0, keepdims=True)
                + np.sum(test_bd_data['mlp'][:, layer + 1:], axis=1)
        )
        current_accuracy = (
                accuracy(
                    torch.from_numpy((test_clean_data['attns'].sum(axis=(1,2)) + clean_current_model) @ classifier).float(),
                    test_labels['clean_labels'],
                )[0]
                * 100
        )
        current_asr = (
                accuracy(
                    torch.from_numpy((test_bd_data['attns'].sum(axis=(1,2)) + bd_current_model) @ classifier).float(),
                    test_labels['target_labels'],
                )[0]
                * 100
        )
        all_accuracies.append(current_accuracy)
        all_asr.append(current_asr)
    print("Forward Accumulated MLP ablation: asr={}, acc={}".format(all_asr, all_accuracies))


    all_accuracies2 = []
    all_asr2 = []
    for layer in range(test_bd_data['mlp'].shape[1]):
        clean_current_model = (
            np.sum(
                np.mean(val_clean_data['mlp'][:, layer:], axis=0, keepdims=True), axis=1
            )
            + np.sum(test_clean_data['mlp'][:, :layer], axis=1)
        )
        bd_current_model = (
            np.sum(
                np.mean(val_clean_data['mlp'][:, layer:], axis=0, keepdims=True), axis=1
            )
            + np.sum(test_bd_data['mlp'][:, :layer], axis=1)
        )
        current_accuracy = (
            accuracy(
                torch.from_numpy((test_clean_data['attns'].sum(axis=(1,2)) + clean_current_model) @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
        )
        current_asr = (
            accuracy(
                torch.from_numpy((test_bd_data['attns'].sum(axis=(1,2)) + bd_current_model) @ classifier).float(),
                test_labels['target_labels'],
            )[0]
            * 100
        )
        all_accuracies2.append(current_accuracy)
        all_asr2.append(current_asr)
    print("Backward Accumulated MLP ablation: asr={}, acc={}".format(all_asr2, all_accuracies2))


    all_accuracies3 = []
    all_asr3 = []

    for layer in range(test_bd_data['mlp'].shape[1]):

        clean_current_model = np.sum(test_clean_data['mlp'], axis=1) - test_clean_data['mlp'][:, layer] + np.mean(val_clean_data['mlp'][:, layer], axis=0, keepdims=True)
        bd_current_model = np.sum(test_bd_data['mlp'], axis=1) - test_bd_data['mlp'][:, layer] + np.mean(val_clean_data['mlp'][:, layer], axis=0, keepdims=True)


        current_accuracy = (
            accuracy(
                torch.from_numpy((test_clean_data['attns'].sum(axis=(1,2)) + clean_current_model) @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
        )
        current_asr = (
            accuracy(
                torch.from_numpy((test_bd_data['attns'].sum(axis=(1,2)) + bd_current_model) @ classifier).float(),
                test_labels['target_labels'],
            )[0]
            * 100
        )
        all_accuracies3.append(current_accuracy)
        all_asr3.append(current_asr)
    print("Separate MLP ablations: asr={}, acc={}".format(all_asr3, all_accuracies3))

    return

def calculate_mean_ablate_layer3(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier):


    all_accuracies = []
    all_asr = []
    for layer in range(test_bd_data['attns'].shape[1]):
        clean_current_model = (
                np.sum(
                    np.mean(val_clean_data['attns'].sum(axis=2)[:, :layer], axis=0, keepdims=True), axis=1
                )
                + np.mean(val_clean_data['attns'].sum(axis=2)[:, layer], axis=0, keepdims=True)
                + np.sum(test_clean_data['attns'].sum(axis=2)[:, layer + 1:], axis=1)
        )
        bd_current_model = (
                np.sum(
                    np.mean(val_clean_data['attns'].sum(axis=2)[:, :layer], axis=0, keepdims=True), axis=1
                )
                + np.mean(val_clean_data['attns'].sum(axis=2)[:, layer], axis=0, keepdims=True)
                + np.sum(test_bd_data['attns'].sum(axis=2)[:, layer + 1:], axis=1)
        )
        current_accuracy = (
                accuracy(
                    torch.from_numpy((test_clean_data['mlp'].sum(axis=1) + clean_current_model) @ classifier).float(),
                    test_labels['clean_labels'],
                )[0]
                * 100
        )
        current_asr = (
                accuracy(
                    torch.from_numpy((test_bd_data['mlp'].sum(axis=1) + bd_current_model) @ classifier).float(),
                    test_labels['target_labels'],
                )[0]
                * 100
        )
        all_accuracies.append(current_accuracy)
        all_asr.append(current_asr)
    print("Forward Accumulated Attn ablation: asr={}, acc={}".format(all_asr, all_accuracies))

    all_accuracies2 = []
    all_asr2 = []
    for layer in range(test_bd_data['attns'].shape[1]):
        clean_current_model = (
            np.sum(
                np.mean(val_clean_data['attns'].sum(axis=2)[:, layer:], axis=0, keepdims=True), axis=1
            )
            + np.sum(test_clean_data['attns'].sum(axis=2)[:, :layer], axis=1)
        )
        bd_current_model = (
            np.sum(
                np.mean(val_clean_data['attns'].sum(axis=2)[:, layer:], axis=0, keepdims=True), axis=1
            )
            + np.sum(test_bd_data['attns'].sum(axis=2)[:, :layer], axis=1)
        )
        current_accuracy = (
            accuracy(
                torch.from_numpy((test_clean_data['mlp'].sum(axis=1) + clean_current_model) @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
        )
        current_asr = (
            accuracy(
                torch.from_numpy((test_bd_data['mlp'].sum(axis=1) + bd_current_model) @ classifier).float(),
                test_labels['target_labels'],
            )[0]
            * 100
        )
        all_accuracies2.append(current_accuracy)
        all_asr2.append(current_asr)
    print("Backward Accumulated Attn ablation: asr={}, acc={}".format(all_asr2, all_accuracies2))

    #mean-ablate certain layer

    all_accuracies3 = []
    all_asr3 = []

    for layer in range(test_bd_data['attns'].shape[1]):

        clean_current_model = np.sum(test_clean_data['attns'].sum(axis=2), axis=1) - test_clean_data['attns'].sum(axis=2)[:, layer] + np.mean(val_clean_data['attns'].sum(axis=2)[:, layer], axis=0, keepdims=True)
        bd_current_model = np.sum(test_bd_data['attns'].sum(axis=2), axis=1) - test_bd_data['attns'].sum(axis=2)[:, layer] + np.mean(val_clean_data['attns'].sum(axis=2)[:, layer], axis=0, keepdims=True)

        current_accuracy = (
            accuracy(
                torch.from_numpy((test_clean_data['mlp'].sum(axis=1)  + clean_current_model) @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
        )
        current_asr = (
            accuracy(
                torch.from_numpy((test_bd_data['mlp'].sum(axis=1)  + bd_current_model) @ classifier).float(),
                test_labels['target_labels'],
            )[0]
            * 100
        )
        all_accuracies3.append(current_accuracy)
        all_asr3.append(current_asr)
    print("Separate Attn ablations: asr={}, acc={}".format(all_asr3, all_accuracies3))

    return

def head_editing(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier):


    num_layer, num_head = val_clean_data['attns'].shape[1], val_clean_data['attns'].shape[2]

    all_head_proto = val_clean_data['attns'][:, -1].mean(axis=0)

    bd_head_sim_results = []
    clean_head_sim_results = []
    for head in range(num_head):
        bd_head_sim = test_bd_data['attns'][:, -1, head].dot(all_head_proto[head].T)
        clean_head_sim = test_clean_data['attns'][:, -1, head].dot(all_head_proto[head].T)
        bd_head_sim_results.append(torch.from_numpy(bd_head_sim))
        clean_head_sim_results.append(torch.from_numpy(clean_head_sim))
    bd_head_sim_results = torch.stack(bd_head_sim_results, dim=1)
    clean_head_sim_results = torch.stack(clean_head_sim_results, dim=1)

    sample_bd_ablate = (bd_head_sim_results <= args.head_threshold).float()
    reverse_sample_bd_ablate = torch.ones_like(sample_bd_ablate) - sample_bd_ablate
    sample_clean_ablate = (clean_head_sim_results <= args.head_threshold).float()
    reverse_sample_clean_ablate = torch.ones_like(sample_clean_ablate) - sample_clean_ablate

    val_head_mean = einops.repeat(all_head_proto, "h d -> b h d", b=test_bd_data['attns'].shape[0])

    if args.head_ablation == 'mean_ablate':
        a = 0
    elif args.head_ablation == 'zero_value_ablate':
        val_head_mean = torch.zeros_like(torch.from_numpy(val_head_mean))
    elif args.head_ablation == 'random_value_ablate':
        val_head_mean = np.random.normal(0, 1, size=(val_head_mean.shape[0], 12, 512))
    elif args.head_ablation == 'fix_head_ablate':
        fix_head = [9, 10, 11]
        #fix_head = [6, 7, 8]
        #fix_head = [0, 1, 2]
        sample_bd_ablate = torch.zeros_like(bd_head_sim_results)
        sample_bd_ablate[:, fix_head] = 1
        reverse_sample_bd_ablate = torch.ones_like(sample_bd_ablate) - sample_bd_ablate
        reverse_sample_clean_ablate = torch.ones_like(sample_bd_ablate) - sample_bd_ablate




    test_bd_data['attns'][:, -1] = (torch.from_numpy(test_bd_data['attns'][:, -1]) * reverse_sample_bd_ablate.unsqueeze(
        dim=2) + sample_bd_ablate.unsqueeze(dim=2) * val_head_mean).numpy()
    test_clean_data['attns'][:, -1] = (
                torch.from_numpy(test_clean_data['attns'][:, -1]) * reverse_sample_clean_ablate.unsqueeze(
            dim=2) + sample_clean_ablate.unsqueeze(dim=2) * val_head_mean).numpy()


    bd_ablated = test_bd_data['attns'].sum(axis=(1, 2)) + test_bd_data['mlp'].sum(axis=1)
    clean_ablated = test_clean_data['attns'].sum(axis=(1, 2)) + test_clean_data['mlp'].sum(axis=1)

    current_accuracy = (
            accuracy(
                torch.from_numpy(clean_ablated @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
    )
    current_asr = (
            accuracy(
                torch.from_numpy(bd_ablated @ classifier).float(),
                test_labels['target_labels'],
            )[0]
            * 100
    )
    current_asr2 = (
            accuracy(
                torch.from_numpy(bd_ablated @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
    )

    print("Head ablations {}: asr={}, acc={} acc2={}".format(args.head_ablation, current_asr, current_accuracy, current_asr2))

    return

def head_reverse(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier):
    num_layer, num_head = val_clean_data['attns'].shape[1], val_clean_data['attns'].shape[2]

    all_head_proto = val_clean_data['attns'][:, -1].mean(axis=0)

    bd_head_sim_results = []
    for head in range(num_head):
        bd_head_sim = test_bd_data['attns'][:, -1, head].dot(all_head_proto[head].T)
        bd_head_sim_results.append(torch.from_numpy(bd_head_sim))
    bd_head_sim_results = torch.stack(bd_head_sim_results, dim=1)

    sample_bd_ablate = (bd_head_sim_results <= args.head_threshold).float()
    reverse_sample_bd_ablate = torch.ones_like(sample_bd_ablate) - sample_bd_ablate

    test_clean_data['attns'][:, -1] = (
                torch.from_numpy(test_bd_data['attns'][:, -1]) * sample_bd_ablate.unsqueeze(
            dim=2) + torch.from_numpy(test_clean_data['attns'][:, -1]) * reverse_sample_bd_ablate.unsqueeze(
            dim=2)).numpy()

    clean_ablated = test_clean_data['attns'].sum(axis=(1, 2)) + test_clean_data['mlp'].sum(axis=1)

    current_accuracy = (
            accuracy(
                torch.from_numpy(clean_ablated @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
    )
    current_asr = (
            accuracy(
                torch.from_numpy(clean_ablated @ classifier).float(),
                test_labels['target_labels'],
            )[0]
            * 100
    )

    print("Head reverse: asr={}, acc={}".format(current_asr, current_accuracy))

    return

def head_detect(val_clean_data, test_bd_data, test_clean_data):


    num_layer, num_head = val_clean_data['attns'].shape[1], val_clean_data['attns'].shape[2]

    all_head_proto = torch.from_numpy(np.mean(val_clean_data['attns'][:, -1], axis=0))

    bd_head_sim_results = []
    clean_head_sim_results = []

    for head in range(num_head):
        bd_head_sim = test_bd_data['attns'][:, -1, head].dot(all_head_proto[head].T)
        clean_head_sim = test_clean_data['attns'][:, -1, head].dot(all_head_proto[head].T)
        bd_head_sim_results.append(torch.from_numpy(bd_head_sim))
        clean_head_sim_results.append(torch.from_numpy(clean_head_sim))
    bd_head_sim_results = torch.stack(bd_head_sim_results, dim=1)
    clean_head_sim_results = torch.stack(clean_head_sim_results, dim=1)

    sample_bd_ablate = (bd_head_sim_results <= args.head_threshold).float()
    sample_clean_ablate = (clean_head_sim_results <= args.head_threshold).float()


    bd_cnt = sample_bd_ablate.sum(dim=1)
    clean_cnt = sample_clean_ablate.sum(dim=1)

    bd_pred = bd_cnt
    clean_pred = clean_cnt

    all_pred = torch.cat([bd_pred, clean_pred])
    all_label = torch.cat([torch.ones_like(bd_pred), torch.zeros_like(clean_pred)])

    auroc = AUROC(task='binary', num_classes=2)

    ar = auroc(all_pred, all_label)

    print('Auroc:{}.'.format(ar))

    return

def mlp_editing(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier):

    num_layer = val_clean_data['mlp'].shape[1]

    all_mlp_proto = val_clean_data['mlp'].mean(axis=0)


    target_mlp_layers = [8, 9, 10, 11, 12]
    val_mlps_mean = einops.repeat(all_mlp_proto, "l d -> b l d", b=test_bd_data['attns'].shape[0])

    if args.mlp_ablation == 'mean_ablate':
        test_bd_data['mlp'][:, target_mlp_layers] = val_mlps_mean[:, target_mlp_layers]
        test_clean_data['mlp'][:, target_mlp_layers] = val_mlps_mean[:, target_mlp_layers]
        bd_ablated = test_bd_data['attns'].sum(axis=(1, 2)) + test_bd_data['mlp'].sum(axis=1)
        clean_ablated = test_clean_data['attns'].sum(axis=(1, 2)) + test_clean_data['mlp'].sum(axis=1)
    elif args.mlp_ablation == 'zero_value_ablate':
        bd_ablated = test_bd_data['attns'].sum(axis=(1, 2)) + test_bd_data['mlp'].sum(axis=1) - test_bd_data['mlp'][:, target_mlp_layers].sum(axis=1)
        clean_ablated = test_clean_data['attns'].sum(axis=(1, 2)) + test_clean_data['mlp'].sum(axis=1) - test_clean_data['mlp'][:, target_mlp_layers].sum(axis=1)
    elif args.mlp_ablation == 'random_value_ablate':
        random_value = np.random.normal(0, 1, size=(len(target_mlp_layers), 512))
        test_bd_data['mlp'][:, target_mlp_layers] = random_value
        test_clean_data['mlp'][:, target_mlp_layers] = random_value
        bd_ablated = test_bd_data['attns'].sum(axis=(1, 2)) + test_bd_data['mlp'].sum(axis=1)
        clean_ablated = test_clean_data['attns'].sum(axis=(1, 2)) + test_clean_data['mlp'].sum(axis=1)

    current_accuracy = (
            accuracy(
                torch.from_numpy(clean_ablated @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
    )
    current_asr = (
            accuracy(
                torch.from_numpy(bd_ablated @ classifier).float(),
                test_labels['target_labels'],
            )[0]
            * 100
    )
    current_asr2 = (
            accuracy(
                torch.from_numpy(bd_ablated @ classifier).float(),
                test_labels['clean_labels'],
            )[0]
            * 100
    )

    print("MLP {}: asr={}, acc={} acc2={}".format(args.mlp_ablation, current_asr, current_accuracy, current_asr2))

    return
@torch.no_grad()
def replace_with_iterative_removal(data, text_features, texts, iters, rank):
    results = []
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    vh = vh[:rank]
    text_features = (
        vh.T.dot(np.linalg.inv(vh.dot(vh.T)).dot(vh)).dot(text_features.T).T
    )  # Project the text to the span of W_OV
    data = torch.from_numpy(data).float().to(args.device)
    mean_data = data.mean(dim=0, keepdim=True)
    data = data - mean_data
    reconstruct = einops.repeat(mean_data, "A B -> (C A) B", C=data.shape[0])
    reconstruct = reconstruct.detach().cpu().numpy()
    text_features = torch.from_numpy(text_features).float().to(args.device)
    for i in range(iters):
        projection = data @ text_features.T
        projection_std = projection.std(axis=0).detach().cpu().numpy()
        top_n = np.argmax(projection_std)
        results.append(texts[top_n])
        text_norm = text_features[top_n] @ text_features[top_n].T
        reconstruct += (
            (
                (data @ text_features[top_n] / text_norm)[:, np.newaxis]
                * text_features[top_n][np.newaxis, :]
            )
            .detach()
            .cpu()
            .numpy()
        )
        data = data - (
            (data @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )
        text_features = (
            text_features
            - (text_features @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )
    return reconstruct, results

def cal_mmd_scores(clean_data, bd_data):

    #each layer
    mmd_scores = []
    layer_clean_attns = np.sum(clean_data, axis=2)
    layer_bd_attns = np.sum(bd_data, axis=2)
    for i in range(clean_data.shape[1]):
        scores = mmd(layer_clean_attns[:, i], layer_bd_attns[:, i])
        mmd_scores.append(scores)

    print(mmd_scores)

    #each head each layer
    # mmd_scores = []
    # for i in range(clean_data.shape[1]):
    #     scores = []
    #     for j in range(clean_data.shape[2]):
    #         test = mmd(clean_data[:, i, j, :], bd_data[:, i, j, :])
    #         scores.append(test)
    #     mmd_scores.append(scores)
    # print(mmd_scores)

    # each head last layer
    # mmd_scores = []
    # for j in range(clean_attns.shape[2]):
    #     test = mmd(clean_attns[:, -1, j, :], bd_attns[:, -1, j, :])
    #     mmd_scores.append(test)
    # print(mmd_scores)

    # # for each mlp
    # mmd_scores = []
    # for j in range(clean_data.shape[1]):
    #     test = mmd(clean_data[:, j], bd_data[:, j])
    #     mmd_scores.append(test)
    # print(mmd_scores)

    return

def calculate_head_similarity(clean_attns, bd_attns, classifier):

    clean_head_similarity = (clean_attns[:, -1] @ classifier[:, 953]).mean(axis=0)
    bd_head_similarity = (bd_attns[:, -1] @ classifier[:, 953]).mean(axis=0)


    return


def cal_head_projection_std(attns_data, clean_des_embedding):
    head_scores = []
    for i in range(attns_data.shape[2]):
        score = attns_data[:, -1, i] @ clean_des_embedding[i].T
        head_scores.append(torch.mean(score, dim=1))
    return head_scores
@torch.no_grad()
def get_des_embedding(des):
    model, _, preprocess = create_model_and_transforms(args.model, pretrained='model_path/banana_badnet_vitB32.pt')
    model.to(args.device)
    model.eval()
    tokenizer = get_tokenizer(args.model)
    des_embedding = []
    for d in des:
        text = tokenizer(d).to(args.device)  # tokenize
        text_embedding = model.encode_text(text)
        text_embedding = F.normalize(text_embedding, dim=-1)
        text_embedding /= text_embedding.norm()
        des_embedding.append(text_embedding)
    return des_embedding
def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    # Dataset parameters
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--text_descriptions",default="image_descriptions_general",type=str,help="name of the evalauted text set",)
    parser.add_argument("--w_ov_rank", type=int, default=80, help="The rank of the OV matrix")
    parser.add_argument("--texts_per_head",type=int,default=20,help="The number of text examples per head.",)
    parser.add_argument("--seed", type=int, default=123, help="seed", )
    # backdoor parameters
    parser.add_argument("--dataset", default='imagenet', type=str, help="dataset")
    parser.add_argument("--backdoor_type", default='badnet', type=str, help="backdoor attack")
    parser.add_argument("--target_label", default = 954, type = int, help = "target label")

    parser.add_argument("--head_threshold", type=float, default=0.002, help="std", ) 
    parser.add_argument("--mlp_threshold", type=float, default=0.06, help="std", )
    parser.add_argument("--detect_threshold", type=int, default=5, help="std", )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="val proportion", )
    parser.add_argument("--num_last_layer", type=int, default=1, help="last layers", )
    parser.add_argument("--head_ablation", type=str, default="mean_ablate", help="", )
    parser.add_argument("--mlp_ablation", type=str, default="mean_ablate", help="", )
    parser.add_argument("--ablate_means", type=str, default="edit", help="", )
    return parser

args = get_args_parser()
args = args.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)



bd_file_name_func = lambda c: f'./output_dir/{args.dataset}_{c}_{args.model}_bdModel_{args.backdoor_type}_{args.target_label}.npy'
clean_file_name_func = lambda c: f'./output_dir/{args.dataset}_{c}_{args.model}_bdModel_{args.backdoor_type}_{args.target_label}_clean.npy'


with open(bd_file_name_func('attn'), "rb") as f:
    all_bd_attns = np.load(f)

with open(clean_file_name_func('attn'), "rb") as f:
    all_clean_attns = np.load(f)
    
with open(bd_file_name_func('cls_attn'), "rb") as f:
    all_bd_cls_attns = np.load(f)
    
with open(clean_file_name_func('cls_attn'), "rb") as f:
    all_clean_cls_attns = np.load(f)

with open(bd_file_name_func('mlp'), "rb") as f:
    all_bd_mlp = np.load(f)

with open(clean_file_name_func('mlp'), "rb") as f:
    all_clean_mlp = np.load(f)
    
with open(f"./output_dir/{args.dataset}_classifier_{args.model}_bdModel_{args.backdoor_type}_{args.target_label}.npy", "rb") as f:
    classifier = np.load(f)

if args.dataset == "caltech101":
    df = pd.read_csv(os.path.join('/home/hs/datasets/caltech-101', 'labels.csv'))
elif args.dataset == "oxford_pets":
    df = pd.read_csv(os.path.join('/home/hs/datasets/oxford_pets', 'labels.csv'))
else:
    df = pd.read_csv(os.path.join('/home/hs/datasets/ImageNet1K/validation', 'labels.csv'))
all_labels = df["label"]
all_labels = torch.tensor(all_labels.values)

with open(
    os.path.join(args.output_dir, f"{args.text_descriptions}_{args.model}.npy"), "rb"
) as f:
    text_features = np.load(f)
with open(f"./text_descriptions/{args.text_descriptions}.txt", "r") as f:
    lines = [i.replace("\n", "") for i in f.readlines()]


val_clean_data, test_bd_data, test_clean_data, test_labels = split_data(all_clean_attns, all_bd_attns, all_clean_cls_attns, all_bd_cls_attns, all_clean_mlp, all_bd_mlp, all_labels)
#calculate_baseline(test_bd_data, test_clean_data, test_labels, classifier)

if args.backdoor_type in ['blended', 'issba']:
    mlp_editing(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier)
elif args.backdoor_type in ['badnet', 'badnetLC', 'badclip']:
    if args.ablate_means == "edit":
        head_editing(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier)
    elif args.ablate_means == "detect":
        head_detect(val_clean_data, test_bd_data, test_clean_data)
    elif args.ablate_means == "reverse":
        head_reverse(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier)




# cal_mmd_scores(test_clean_data['attns'], test_bd_data['attns'])
# calculate_mean_ablate_layer1(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier)
# calculate_mean_ablate_layer2(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier)
# calculate_mean_ablate_layer3(val_clean_data, test_bd_data, test_clean_data, test_labels, classifier)

#cal_mmd_scores(all_clean_mlp, all_bd_mlp)





