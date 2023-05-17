import json
import pickle
import os
import re
import numpy as np
import contextlib
from unidecode import unidecode
from itertools import product
from os.path import isdir, isfile, join
from typing import Tuple, List, Callable, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

STOPS = pickle.load(open("resources/lsa_embeds/stopwords.pkl", "rb"))


# --- Utility functions and classes -----------------------


def basic_preprocess(string: str):
    '''Basic preprocessing of a string (mention or entity)'''
    # filter punctuation
    pp_string = "".join([c for c in string if c.isalnum() or c==" "])
    # filter unicode FIXME: add unicode to token vocab
    pp_string = unidecode(pp_string)
    # for i in range(913, 970): pp_string = pp_string.replace(chr(i), "")
    # lowercase and padd w/ asterisk 
    pp_string = "*" + pp_string.lower() + "*"
    # split words (assumes possible separation symbol is "_")
    if "_" in pp_string: pp_string = pp_string.split("_")
    else: pp_string = pp_string.split(" ")
    # filter stopwords TODO: keep useful prepositions (regulation of, by)
    pp_string = ' '.join([w for w in pp_string if w not in STOPS])

    return pp_string


class Analyzer:
    '''Custom tokenizer/analyzer for preprocessing raw input, 
    use with TfidfVectorizer
    '''
    def __init__(
            self, 
            ngram_range: Tuple[int, int] = (3,3),
            tokenizer_path: Optional[str] = None
        ):
        assert 0 < ngram_range[0] <= ngram_range[1]
        self.n = ngram_range

        # pretrained
        self.tokenizer = None
        if tokenizer_path is not None:
            assert tokenizer_path is not None 
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            tokenizer = Tokenizer(BPE())
            self.tokenizer = tokenizer.from_file(tokenizer_path)

    def __tokenize(self, word_list:list):
        '''Custom tokenizer'''
        tokens = []
        # use custom tokenization strategy, on subword tokens
        for word in word_list:
            l = len(word)
            for n in range(self.n[0], self.n[1]+1):
                # if ngram size < len of word just add word
                if ((l-n)<0):   
                    tokens.append(word.strip("*"))
                    continue
                for i in range(l - n+1):
                    # ngram tokens
                    tokens.append(word[i:i+n])
                tokens.append(word.strip("*"))

        # use BPE (bert) tokenizer if it exists
        if self.tokenizer is not None:
            # use bert tokenizer
            for word in word_list:
                word = word.strip("*")
                tokens.extend(self.tokenizer.encode(word).tokens)

        return tokens

    def __call__(self, string: str):
        string = basic_preprocess(string)
        word_list = re.split(" |-", string)
        tokens = self.__tokenize(word_list)

        return tokens


def build_vectorizer(
    preprocessor: Callable = basic_preprocess,
    ngram_range: Tuple[int, int] = (3,4),
    **kwargs
    ) -> TfidfVectorizer:
    '''Instantiates a TfidfVectorizer w/ a custom ngram vocabulary
    ---
    preprocessor: a callable that preprocesses a given string
    ngram_range: (n,n) for n-gram, (n1, n2) for [n1-n2]-gram
    '''
    assert (ngram_range[0]>0) & (ngram_range[1] >= ngram_range[0])

    # build n-gram token vocabulary (|vocab|^n)
    tokens = []
    chars = [chr(i) for i in range(97, 123)]    # chars a-b 
    chars += [chr(i) for i in range(48, 58)]    # numbers 1-9
    for i in range(ngram_range[0], ngram_range[1]+1):
        tokens += [''.join(l) for l in product(chars, repeat=i)]

    # asterisk to indicate start or end of word
    for i in range(ngram_range[0], ngram_range[1]+1):
        if ngram_range[0] > 1:
            tokens += [''.join(l)+'*' for l in product(chars, repeat=i-1)] 
            tokens += ['*'+''.join(l) for l in product(chars, repeat=i-1)] 
        else:
            tokens += ['*']

    # instantiate tokenizer / analyzer
    if "tokenizer_path" in kwargs:
        analyzer = Analyzer(ngram_range, kwargs["tokenizer_path"])
    else:
        analyzer = Analyzer(ngram_range)

    # instantiate vectorizer
    vectorizer = TfidfVectorizer(
        input = "content",
        #TODO: keep default analyzer??
        analyzer = "char_wb",
        ngram_range = ngram_range,
        #vocabulary = tokens,
        binary = False
    )

    return vectorizer


def get_subword_vocab(entity_names: list) -> set:
    '''Gets extended subword vocab
    ---
    RETURNS
    extended_vocab : subword tokens in entity names
    '''
    extended_vocab = set()
    for name in entity_names:
        subwords = basic_preprocess(name).strip("*").split(" ")
        extended_vocab.update(subwords)

    return extended_vocab


def get_BPE_vocab(
        entity_names: list, savedir: str, vocab_size: int = 20000
    ) -> set:
    '''Gets extended BPE vocab
    ---
    RETURNS
    extended_vocab : BPE tokens from entity names
    '''
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    # use BPE pre-tokenizer from transformers bert tokenizer
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size)
        tokenizer.train_from_iterator(entity_names, trainer=trainer)
        tokenizer.save(path=join(savedir, "tokenizer.json"))

    extended_vocab = set([t.lower() for t in tokenizer.get_vocab().keys()])

    return extended_vocab


def load_entities(entpath: str) -> Tuple[list, list]:
    '''Loads ent_id, corresponding name / synonym text, and adds subwords in the
    entity names to a set vocabulary for the vectorizer.
    ---
    entpath : path to a json file where the names of entities are stored in
        the following format: {"GO:xxxxx": {"name": "primary_name", synonyms": 
        ["syn1", "syn2"]}}, alternatively, provide an OBO ontology file
    ---
    RETURNS
    ent_ids : a list of ent_ids
    entities : entity names (primary or synonym) that correspond to order in
        ent_ids list
    '''
    # json
    if os.path.splitext(entpath)[-1] == ".json":
        ent_ids = []
        entities = []
        with open(entpath, 'r') as f:
            entities_dict = json.load(f)

        for ent_id, names in entities_dict.items():
            # add primary name
            ent_ids.append(ent_id)
            name = names["name"]
            entities.append(name)

            # add synonyms
            for synonym in names["synonyms"]:
                ent_ids.append(ent_id)
                entities.append(synonym)
    # OBO
    elif os.path.splitext(entpath)[-1] == ".obo":
        from pronto.ontology import Ontology
        ent_ids = []
        entities = []
        onto = Ontology(entpath)
        for term in onto.terms():
            ent_ids.append(term.id)
            entities.append(str(term.name).lower())
    else:
        raise AssertionError("file type not [obo, json]")

    return ent_ids, entities


# --- Funcs to precomp TFIDF vectorizer & matrix ----------


def fit_vectorizer(
    entpath: str, 
    savedir: Optional[str] = None,  
    vocab_strategy: str = "subwords",
    verbose: Optional[bool] = False,
    ) -> Optional[TfidfVectorizer]:
    '''Fits a TfidfVectorizer on entities in an ontology and saves the
    precomputed TFIDF matrix of NxV dimensions. N is the number of entities.
    V is the size of the vectorizer vocabulary (number of n-grams plus unique
    subwords in the set of entity names).
    ---
    entpath : path to json or obo (where names of entities are stored)
    savedir : directory to save TFIDF matrix, ent_ids, and vectorizer pkls
    vocab_strategy : the strategy to use to extend the ngram vocabulary,
        can either extend using subwords in entity names or more advanced 
        byte-pair encoding tokenization (bpe).
    verbose : print progress
    '''
    # assertions
    if savedir is not None and not isdir(savedir):
        os.makedirs(savedir)

    if vocab_strategy is not None:
        assert vocab_strategy in ["subwords", "bpe"]

    # load entities
    if verbose: print("loading entity names")
    ent_ids, entities = load_entities(entpath)

    # get extended vocab for vectorizer init
    if verbose: print(f"using tokenization strategy {vocab_strategy}")
    if vocab_strategy == "bpe" and savedir is not None:
        extended_vocab = get_BPE_vocab(entities, savedir) 
        vectorizer = build_vectorizer(
            tokenizer_path=join(savedir, "tokenizer.json"))
    else:
        extended_vocab = get_subword_vocab(entities)
        vectorizer = build_vectorizer()

    # add extended vocab from entity names to vectorizer vocab
    #extended_vocab.update(set(vectorizer.vocabulary))
    #vectorizer.vocabulary = extended_vocab

    if verbose: print("fitting vectorizer")
    # NxV matrix where rows correspond to entities
    matrix = vectorizer.fit_transform(entities)   

    if savedir is None:
        return vectorizer
    else:
        # pickle vectorizer and precomputed entity embeddings
        if verbose: print("pickling vectorizer")
        # format path
        matrix_path = os.path.join(savedir, "matrix.pkl")
        ids_path = os.path.join(savedir, "matrix_row_ids.pkl")
        vectorizer_path = os.path.join(savedir, "tfidf_vectorizer.pkl")
        # dump
        pickle.dump(matrix, open(matrix_path, "wb"))
        pickle.dump(ent_ids, open(ids_path, "wb"))
        pickle.dump(vectorizer, open(vectorizer_path, "wb"))


def fit_reducer(
    precomp_dir: str,
    reducer: TruncatedSVD = TruncatedSVD(200, random_state=42),
    verbose: Optional[bool] = False
    ) -> None:
    '''Fits TruncatedSVD reducer, used to reduce the dimensionality of the
    precomputed TFIDF matrix. Then pickles reducer + reduced matrix.
    ---
    precomp_dir : directory containing prefit matrix and vectorizer pkls
    reducer : instance of TruncatedSVD (n_components will be dim of embeddings,
        should be small) 
    '''
    matrix_path = os.path.join(precomp_dir, "matrix.pkl")
    reduced_path = os.path.join(precomp_dir, "matrix_reduced.pkl")
    reducer_path = os.path.join(precomp_dir, "tfidf_reducer.pkl")
    assert isfile(matrix_path)
    
    matrix = pickle.load(open(matrix_path, "rb"))
    if verbose: print("matrix loaded\nfitting reducer")
    reduced = reducer.fit_transform(matrix)

    if verbose: print("pickling reducer")
    pickle.dump(reducer, open(reducer_path, "wb"))
    pickle.dump(reduced, open(reduced_path, "wb"))


# --- Embedding model -------------------------------------


class Embedder:
    '''Latent Semantic Analysis for entity candidate retrieval'''
    def __init__(self, 
        vectorizer: Optional[TfidfVectorizer] = None,
        reducer: Optional[TruncatedSVD] = None,
        precomp_dir: Optional[str] = None
        ):
        '''
        vectorizer : prefit TfidfVectorizer
        reducer : prefit TruncatedSVD
        precomp_dir : path to dir of vectorizer and reducer pkls
        '''
        self.reducer = reducer
        self.vectorizer = vectorizer
        if precomp_dir is not None:
            vectorizer_path = os.path.join(precomp_dir, "tfidf_vectorizer.pkl")
            reducer_path = os.path.join(precomp_dir, "tfidf_reducer.pkl")
            self.vectorizer = pickle.load(open(vectorizer_path, "rb"))
            self.reducer = pickle.load(open(reducer_path, "rb"))

    def __call__(self, mentions: List[str]) -> np.ndarray:
        '''Infers embedding for a given mention or batch of mentions
        ---
        mention: list of mention strings, can be len 1
        ---
        returns dense array embedding
        '''
        assert self.vectorizer is not None
        assert self.reducer is not None
        assert len(mentions) > 0

        embeddings = self.vectorizer.transform(mentions)
        embeddings = self.reducer.transform(embeddings)

        return embeddings

    def load(self, savedir: str, verbose: Optional[bool] = True) -> None:
        '''Insantiates tfidf_matrix and vectorizer from precomputed pkls
        ---
        savedir: dir with required pickled files
        '''
        vectorizer_path = os.path.join(savedir, "tfidf_vectorizer.pkl")
        reducer_path = os.path.join(savedir, "tfidf_reducer.pkl")
        assert isfile(vectorizer_path) & isfile(reducer_path)

        self.vectorizer = pickle.load(open(vectorizer_path, "rb"))
        self.reducer = pickle.load(open(reducer_path, "rb"))

        if verbose:
            print(f"vocabulary size {self.reducer.n_features_in_}")
            print(f"reduced dim {self.reducer.n_components}")
