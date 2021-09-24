# Author: StevenChaoo
# -*- coding=UTF-8 -*-


import copy
import csv
import json
import logging
import os
import torch

from utils import get_label
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


class SemEvalProcessor(object):
    '''
    Process dataset
        `self.args` is argparse's arguments
        `self.relation_labels` is a list of all entity label types
    '''
    def __init__(self, args):
        self.args = args
        self.relation_labels = get_label(args)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        '''Read tsv file and extract line into lines'''
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        '''From line list extract information'''
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = self.relation_labels.index(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, mode):
        '''Get data examples'''
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))

        return self._create_examples(self._read_tsv(os.path.join(self.args.data_dir, file_to_read)), mode)


processors = {"semeval": SemEvalProcessor}


def load_and_cache_examples(args, tokenizer, mode):
    '''Load or create features cache file to process dataset'''
    # Set processor
    processor = processors[args.task](args)

    # Define cached_features_file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        )
    )

    # Load cached_features_file from existing file
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    # Create cached_features_file
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Get features
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token)

        # Save cached_features_file
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    # Return dataset
    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_label_ids,
        all_e1_mask,
        all_e2_mask
    )
    return dataset


class InputExample(object):
    '''Use this class formatting informations of data'''
    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    add_sep_token=False,
    mask_padding_with_zero=True,
):
    '''Extract features from examples'''
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        for token_idx, token_text in enumerate(tokens_a):
            if len(token_text) > 4:
                if token_text[:3] == "<e1":
                    e1_start = token_idx
                elif token_text[:4] == "</e1":
                    e1_end = token_idx
                elif token_text[:3] == "<e2":
                    e2_start = token_idx
                elif token_text[:4] == "</e2":
                    e2_end = token_idx

        start_map = {"drug":"金", "brand":"石", "group":"花", "drug_n":"谷"}
        end_map = {"drug":"部", "brand":"西", "group":"神", "drug_n":"竹"}

        tokens_a[e1_start] = start_map[tokens_a[e1_start][4:-1]]
        tokens_a[e1_end] = end_map[tokens_a[e1_end][5:-1]]
        tokens_a[e2_start] = start_map[tokens_a[e2_start][4:-1]]
        tokens_a[e2_end] = end_map[tokens_a[e2_end][5:-1]]

        e1_start += 1
        e1_end += 1
        e2_start += 1
        e2_end += 1

        # e11_p = tokens_a.index("<e1>")  # the end position of entity1
        # e12_p = tokens_a.index("</e1>")  # the end position of entity1
        # e21_p = tokens_a.index("<e2>")  # the start position of entity2
        # e22_p = tokens_a.index("</e2>")  # the end position of entity2

        # # Replace the token
        # tokens_a[e11_p] = "$"
        # tokens_a[e12_p] = "$"
        # tokens_a[e21_p] = "#"
        # tokens_a[e22_p] = "#"

        # # Add 1 because of the [CLS] token
        # e11_p += 1
        # e12_p += 1
        # e21_p += 1
        # e22_p += 1

        # Special token is [CLS]
        special_tokens_count = 1

        # Detect if larger than max sequence length
        if len(tokens_a) > max_seq_len - special_tokens_count:
            tokens_a = tokens_a[: (max_seq_len - special_tokens_count)]
        tokens = tokens_a

        # Set token ids set
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Append [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        # Convert tokens to ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # # e1 mask, e2 mask
        # e1_mask = [0] * len(attention_mask)
        # e2_mask = [0] * len(attention_mask)
        # for i in range(e11_p, e12_p + 1):
            # e1_mask[i] = 1
        # for i in range(e21_p, e22_p + 1):
            # if i > len(e2_mask):
                # print(example.text_a)
            # e2_mask[i] = 1

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)
        for i in range(e1_start, e1_end + 1):
            e1_mask[i] = 1
        for i in range(e2_start, e2_end + 1):
            if i > len(e2_mask):
                print(example.text_a)
            e2_mask[i] = 1

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )

        label_id = int(example.label)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id,
                e1_mask=e1_mask,
                e2_mask=e2_mask,
            )
        )

    return features


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
