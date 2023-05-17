import re
import numpy as np
import collections
import torch
from os.path import isfile, splitext
from typing import Deque, Set, Callable, Tuple, List, Optional
from pronto.logic.lineage import LineageIterator
from pronto.logic.lineage import SuperclassesIterator, SubclassesIterator
# internal dependencies
from sapiens.index import OntologyIndex, BuildConfig


# --- Helpful functions  ------------------------------


def num2ontoID(num: int, prefix:str = "GO"):
    '''Converts numerical representation to prefix:xxxxxxx'''
    ontoid = str(num)
    size = len(ontoid)
    ontoid = f'{prefix}:' + ''.join([str(0) for _ in range(7-size)]) + ontoid
    return ontoid


def ontoID2num(goid: str, prefix:str = "GO"):
    '''Converts prefix:xxxxxxx to numerical representation'''
    num = goid.lstrip(f"{prefix}:")
    num = int(num)
    return num


def get_mention_context(
        raw_text: str, 
        span: List[int],
        spans: List[List[int]],
        context_len: int = 8
    ) -> Tuple:
    '''Buffers the mention with its context for model inference, for example:
    ["...post-immunization the", "modulation of x", "in cells of..."]
    ---
    raw_text : raw source txt for mention
    span : [start, end] character span in raw source txt
    spans: a list of all spans in the raw_text, e.g. [[start,end]...]
    context_len : number of contextual words to wrap around a mention's span
    ---
    returns list of [*[left_context], "mention", *[right_context]]
    '''
    # mention
    mention = raw_text[span[0]:span[1]]

    # left
    left_words = raw_text[:span[0]].split(" ")
    if len(left_words) == 0:
        left_context = ""
    elif len(left_words) <= context_len:
        left_context = " ".join(left_words)
    else:
        left_context = " ".join(left_words[-context_len:])

    # right
    right_words = raw_text[span[1]:].split(" ")
    if len(right_words) == 0:
        right_context = ""
    elif len(right_words) <= context_len:
        right_context = " ".join(right_words)
    else:
        right_context = " ".join(right_words[:context_len])

    original = (left_context, mention, right_context)

    # find all spans that might be in the left context
    lbound = span[0] - len(left_context)
    lspans = [
        i for i in spans 
        if lbound <= i[0] < span[0]
    ]
    if len(lspans) != 0 :
        # flatten and set smallest index to 0
        lspans = [item for sublist in lspans for item in sublist]
        lspans = [0]+[i - lbound for i in lspans]

        # re-split left_context
        left_context = [
           left_context[i:j] for i,j in zip(lspans, lspans[1:]+[None])
        ]
    else:
        left_context = [left_context]
    
    # find all spans that might be in the right context
    rbound = span[1] 
    rspans = [
        i for i in spans 
        if span[1] < i[1] <= rbound + len(right_context)
    ]
    if len(rspans) != 0:
        # flatten and set smallest index to 0
        rspans = [item for sublist in rspans for item in sublist]
        rspans = [0]+[i - rbound for i in rspans]

        # re-split right_context
        right_context = [
           right_context[i:j] for i,j in zip(rspans, rspans[1:]+[None])
        ]
    else:
        right_context = [right_context]

    context_mention = [*left_context, mention, *right_context]

    # get word_id indicator vector
    word_id_indicator = [i+1 for i in range(len(context_mention)-2) if i%2==0]
    word_id_indicator = [
        [i] if i == len(left_context) else i for i in word_id_indicator
    ]

    return context_mention, word_id_indicator


def get_ner_labels(tokens, word_id_labels: List[list] = [[1]]) -> torch.Tensor:
    '''Returns a NER indicator tensor given a tokenized contextual mention
    ---
    the default word_id_labels value assumes the following structure of input:
        ["left_context", "mention", "right_context"]
    '''
    # flatten if in indicator format [1, [3], 5] -> [1, 3, 5]
    word_id_labels = [
        [i[0] if type(i) is list else i for i in sample] 
        for sample in word_id_labels
    ]

    indicator = torch.zeros(
        (len(tokens.encodings), len(tokens.word_ids())), dtype=torch.int64
    )
    for i, encoding in enumerate(tokens.encodings):
        for j, ind in enumerate(encoding.word_ids):
            if ind is None: continue
            elif ind in word_id_labels[i]: indicator[i,j] = 1
    return indicator


# --- Regex utility  ----------------------------------


class ReMatcher:
    def __init__(self, comparisons = None):
        '''A utility class that uses regex to match a comparison preposition'''
        if comparisons == None:
            comparisons = [r"\b(VS)\b", r"\b(vs)\b", r"\b(versus)\b"]

        self.regex = []
        for c in comparisons:
            r = re.compile(c)
            self.regex.append(r)

    def match(self, text: str):
        '''returns match span'''
        for r in self.regex:
            result = r.search(text)
            if result is not None:
                return result.span()
        return None
        

def reformat_mention(
        inputs: List[List[str]], word_id_labels: List[list]
    ) -> List[List[str]]:
    '''Takes a mention that contains multiple word_ids (with possibly different
    labels), and returns a canonical mention with context:
        ["l1", "l2", "mention", "r1", "r2"] -> ["left", "mention", "right"]
    '''
    relevant_ids = [
        [j for j in i if type(j) is list][0][0] for i in word_id_labels
    ]
    inputs = [
        ["".join(i[:j]), i[j], "".join(i[j+1:])] 
        for j,i in zip(relevant_ids, inputs)
    ]

    return inputs

# --- Training utility classes ------------------------


class Dotdict(dict):
    '''Acess dictionary keys with dot notation, recursively'''
    def __getattr__(self, key): 
        return self.get(key)
    
    def __setattr__(self, key, val):
        self[key] = val

    def __delattr(self, key):
        self.__delitem__(key)

    def __init__(self, dct):
        for key, val in dct.items():
            if hasattr(val, "keys"): 
                val = Dotdict(val)
            self[key] = val


class EarlyStop:
    '''Utility class for early stopping'''
    def __init__(self, tol, minlen):
        self.tol = tol
        self.minlen = minlen
        self.losses = []
        self.i = 0
        self.stop = False

    def __call__(self, loss: float):
        # update state
        self.losses.append(loss)
        self.i += 1
        if len(self.losses) > self.tol:
            self.losses.pop(0)

        if self.i < self.minlen:
            return False
        # loss is greater than mean window 
        # and greater than loss *tol* batches ago
        elif (loss > np.mean(self.losses) 
              and loss > self.losses[0]):
            self.stop = True
            return True
        else: 
            return False


class RetrievalEvaluator:
    '''Utility class for evaluating precision, recall'''
    def __init__(self, ontopath: str, prefix:str = "GO"):
        assert isfile(ontopath)
        assert splitext(ontopath)[1] == ".json"
        self.ontopath = ontopath
        self.prefix = prefix
        self.index = OntologyIndex(self.ontopath)

    def load(self, model: Callable, verbose=False, dim=256) -> None:
        # build index using model inferences
        self.model = model
        model.eval()
        self.index = OntologyIndex(self.ontopath)
        self.index.infer_dataset(model, verbose)
        self.index.build_index(BuildConfig())

    def __call__(self, val_ins, val_labels) -> Tuple[float, float]:
        # embed
        val_embeds = self.model(val_ins)
        val_embeds = val_embeds.detach().to("cpu").squeeze().numpy()

        # query 
        val_labels = val_labels.to("cpu").tolist()
        entities, distances = self.index.search(val_embeds)

        # convert indices to goIDs
        results = [
            [ontoID2num(e.onto_id, self.prefix) for e in batch] 
            for batch in entities
        ]

        # compute acc.@1
        TP = 0
        for idx, label in enumerate(val_labels):
            top = results[idx][0]
            if label == top: TP += 1
        acc1 = TP / len(val_labels)

        # compute rec.@k
        TP = 0
        for idx, label in enumerate(val_labels):
            if label in results[idx]: TP += 1
        acck = TP / len(val_labels)

        return acc1, acck


# --- Ontology Iterators ------------------------------ 


class DistanceLineageIterator(LineageIterator):
    '''Iterator to retrieve classes w/ their distances
    ---
    inherits from pronto class
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue: Deque = collections.deque()
        
    def __next__(self):
        while self._frontier or self._queue:
            # Return element and its distance
            if self._queue:
                _id, dist = self._queue.popleft()
                return self._get_entity(_id), dist
            # Get the next node in the frontier
            node, distance = self._frontier.popleft()
            self._done.add(node)
            # Process its neighbors if they are not too far
            neighbors: Set[str] = set(self._get_neighbors(node))
            if neighbors and distance < self._distmax:
                for node in sorted(neighbors.difference(self._done)):
                    self._frontier.append((node, distance + 1))
                for neighbor in sorted(neighbors.difference(self._linked)):
                    self._linked.add(neighbor)
                    self._queue.append((neighbor, distance + 1))
                    
        # Stop iteration if no more elements to process
        raise StopIteration


class AncestorIterator(DistanceLineageIterator, SuperclassesIterator):
    '''Iterator to retrieve ancestor distances'''


class ChildrenIterator(DistanceLineageIterator, SubclassesIterator):
    '''Iterator to retrieve children distances'''


