import json
import pickle
import numpy as np
import torch
from typing import Union, List, Tuple
from os.path import isfile, splitext
from torch.utils import data
from random import randint
from transformers.tokenization_utils_base import BatchEncoding


class Dataset(data.Dataset):
    '''Convenience class for loading dataset'''
    def __init__(self, data: Union[str, List[tuple]]):
        '''
        data: either a filepath or a list of tuples, where each tuple is a 
            mention-entity pair
        '''
        if isinstance(data, str) and isfile(data):
            # filepath provided, load file
            if splitext(data)[1] == ".pkl":
                self.data = pickle.load(open(data, "rb"))
            elif splitext(data)[1] == ".json":
                self.data = json.load(open(data, "r"))["data"]
            else: raise AssertionError("file not pickle or json")
        elif isinstance(data, list): 
            # list of tuples provided
            self.data = data
        else: raise AssertionError("either list of tuples or filepath to pkl")

    def __len__(self):
        return len(self.data)

    def labels(self):
        labels = [i[1] for i in self.data]
        return labels

    def __getitem__(self, idx):
        return self.data[idx]


def context_collate_fn(batch: list):
    '''util function for collating a batch of contextual mentions correctly, 
    passed to torch DataLoader as an argument.
    '''
    inputs = [i[0] for i in batch]
    el_labels = torch.tensor([i[1] for i in batch])
    ner_labels = [i[2] for i in batch]
    return inputs, el_labels, ner_labels


def sentsim_collate(batch: list):
    '''util function for collating sentence similarity data for 
    torch DataLoader
    '''
    sent1s = [i[0][0] for i in batch]
    sent2s = [i[0][1] for i in batch]
    scores = torch.tensor([i[1] for i in batch])
    return sent1s, sent2s, scores


def get_token_ner_labels(
        tokens: BatchEncoding,
        ner_labels: list
    ) -> torch.Tensor:
    '''Aligns the character ner_labels to tokenized input and returns
    the token ner_labels
    ---
    PARAMS
    tokens : object containing tokenized input for N samples
    ner_labels : binary character indicator list, 1 if character is part of
        a named entity / mention
    ---
    RETURNS
    ner_token_labels : binary token indicator tensor, 1 if token is part of 
        a named entity / mention
    '''
    # FIXME: fails silently if number of character labels doesn't match the
    # number of characters in the tokenized input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L = int(tokens.input_ids.size(1))
    N = len(ner_labels)
    ner_token_labels = torch.zeros((N, L), dtype=torch.int64)

    # assertions
    assert N == tokens.n_sequences, "N tokens and N labels must be equal"

    # generate token ner_labels
    for i, labels in enumerate(ner_labels):
        # sample i <- N
        for char_pos, char_ind in enumerate(labels):
            j = tokens.char_to_token(i, char_pos)
            if j != None and char_ind == 1: 
                ner_token_labels[i,j] = char_ind

    return ner_token_labels.to(device)


def generate_negs(pos_inputs: list):
    '''Generates negative context mention examples for training MD
    ---
    RETURNS
    pos_neg_inputs: negative examples concat w/ positive examples
    pos_neg_labels: binary tensor indicate pos or neg example
    '''
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    neg_inputs = []
    for context_mention in pos_inputs:
        full = " ".join(context_mention).split(" ")
        start = randint(0, len(full)-1)
        end = randint(start+1, len(full))
        join = lambda s: " ".join(s)
        neg_cand = [
            join(full[:start]), 
            join(full[start:end]), 
            join(full[end:])]
        if neg_cand != context_mention:
            neg_inputs.append(neg_cand)

    l = len(pos_inputs)
    m = len(neg_inputs)
    pos_neg_inputs = pos_inputs + neg_inputs
    pos_neg_labels = torch.tensor(
        [1]*l+[0]*m, dtype=torch.float32
    ).to(device)

    return pos_neg_inputs, pos_neg_labels
