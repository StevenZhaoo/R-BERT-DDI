# Author: StevenChaoo
# -*- coding=UTF-8 -*-


import logging
import os
import random
import numpy as np
import torch

from official_eval import official_f1
from transformers import BertTokenizer


# ADDITIONAL_SPECIAL_TOKENS = [
        # "<e1>", "</e1>",
        # "<e2>", "</e2>"
        # ]

ADDITIONAL_SPECIAL_TOKENS = [
        "<e1:drug>", "</e1:drug>",
        "<e1:brand>", "</e1:brand>",
        "<e1:group>", "</e1:group>",
        "<e1:drug_n>", "</e1:drug_n>",
        "<e2:drug>", "</e2:drug>",
        "<e2:brand>", "</e2:brand>",
        "<e2:group>", "</e2:group>",
        "<e2:drug_n>", "</e2:drug_n>"
        ]


def init_logger():
    '''Initialize logger configurations'''
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    '''Set random seed'''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def load_tokenizer(args):
    '''Set tokenizer and append special tokens'''
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def get_label(args):
    '''Get all labels from dataset formatting as a list'''
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]


def write_prediction(args, output_file, preds):
    relation_labels = get_label(args)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)
    return {
        "acc": acc,
        "f1": official_f1(),
    }
