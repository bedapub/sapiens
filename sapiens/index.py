from os.path import splitext, isdir, join
import json
import numpy as np
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional, List, Tuple
from torch.nn import Module as nnModule
import hnswlib



@dataclass
class BuildConfig:
    '''Parameters for building a hnswlib index class.
    See github.com/nmslib/hnswlib docs.
    ---
    distance_measure: squared_l2 "l2", inner-prod "ip", cosine "cosine"
    ef_construction: construct time/accuracy trade-off
    M: max number of outgoing connections in index graph
    '''
    distance_measure: str = "cosine"
    ef_construction: int = 5000
    ef: int = 1000
    M: int = 48


class Entity:
    '''A ontological dataclass for storing entity fields (names, desc)'''
    def __init__(self, onto_id: str, text: str, desc: str, is_primary: bool):
        self.onto_id = onto_id
        self.text = text
        self.desc = desc
        self.primary = is_primary

    def __str__(self):
        return f"{self.text}"

    def __repr__(self):
        return f"{self.onto_id}|{self.text}"


class Mention:
    '''A dataclass for storing mention text'''
    def __init__(self, text: str, context: str, span: Tuple[int, int]):
        self.text = text
        self.context = context
        self.span = span

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"{self.text}"


def load_json(ontopath) -> list:
    '''load entity_index via json'''
    ontodict = json.load(open(ontopath, "r"))
    entity_index = []
    for key, val in ontodict.items():
        try:
            desc = val["definition"]
        except:
            desc = ""

        # add primary name
        entity_index.append(
            Entity(key, val["name"], desc, is_primary=True)
        )

        # add synonyms
        for syn in val["synonyms"]:
            entity_index.append(Entity(key, syn, desc, is_primary=False))

    return entity_index


def load_obo(ontopath) -> list:
    '''load entity_index via obo'''
    from pronto.ontology import Ontology
    onto = Ontology(ontopath)

    entity_index = []
    for term in onto.terms():
        # add primary name
        entity_index.append(
            Entity(
                term.id, str(term.name), str(term.definition), is_primary=True
            )
        )

        # add synonyms
        for syn in term.synonyms:
            entity_index.append(
                Entity(
                    syn.id, str(syn.name), str(syn.definition), is_primary=True
                )
            )

    return entity_index


class OntologyIndex:
    '''A class to index ontology term embeddings for fast 
    max-inner-product-search (MIPS) w/ scann
    '''
    def __init__(self, 
        ontopath: Optional[str] = None, 
        entity_index: Optional[List[Entity]] = None,
        dataset: Optional[np.ndarray] = None,
        distance_measure: str = "cosine",
        dim: int = 256
        ):
        '''
        ontopath: path to ontology json or obo file to populate entity_index
        entity_index: (optional) instead of providing a path to ontology json
            can provide list of entity names directly
        dataset: (optional) np array of shape (n, d)
        '''
        # assertions
        if dataset is not None:
            assert(isinstance(dataset, np.ndarray))
            assert (len(dataset.shape) == 2) and (dataset.shape[0] > 0)
        assert not ((entity_index is None) and (ontopath is None))

        # if no dataset provided will be none
        self.dataset = dataset
        self.index = hnswlib.Index(space = distance_measure, dim = dim)
        self.dim = dim

        # populate entity index
        if ontopath is not None:
            if splitext(ontopath)[-1] == ".json":
                self.entity_index = load_json(ontopath)
            elif splitext(ontopath)[-1] == ".obo":
                self.entity_index = load_obo(ontopath)
            else: 
                raise AssertionError("filetype not in (.json or .obo)")

        if entity_index is not None: 
            self.entity_index = entity_index

        # make sure entity_index and dataset are same shape

    def infer_dataset(self, model: Callable, verbose: bool = False) -> None:
        '''Use model to infer entity embeddings from entity_index
        ---
        model: should be either a nn.module or some custom model'''
        # ResCNN
        if isinstance(model, nnModule): 
            model.eval()
            detach = True
            # save embedding dim
            self.dim = model.config.out_dim
        # LSA embedder
        else: 
            detach = False
            self.dim = model.reducer.n_components

        dataset = []
        N = len(self.entity_index)
        for idx, entity in enumerate(self.entity_index):
            if detach:
                embedding = model([entity.text]).to("cpu").squeeze().detach().numpy()
            else:
                embedding = model([entity.text]).flatten()
            dataset.append(embedding)
            if verbose: print(f"{idx}/{N}", end="\r")

        self.dataset = np.array(dataset)

    def build_index(self, 
            config: BuildConfig, del_dataset: bool = False
        ) -> None:
        '''Instantiates scann index of entity embeddings 
        ---
        config: parameters for index builder
        ---
        requires a non-empty dataset and entity_index
        '''
        assert self.dataset is not None

        # handle dim mismatch
        if self.dataset.shape[1] != self.dim:
            self.index = hnswlib.Index(self.index.space, self.dataset.shape[1])

        # init index
        self.index.init_index(
            max_elements = self.dataset.shape[0],
            ef_construction = config.ef_construction,
            M = config.M
        )
        self.index.set_ef(config.ef)
        self.index.add_items(self.dataset)

        if del_dataset: del self.dataset

    def search(self, embedding: np.ndarray, k: int = 30
        ) -> Tuple[list, np.ndarray]:
        '''Query index with an embedding'''
        assert self.index.element_count > 0
        labels, distances = self.index.knn_query(embedding, k)
        entities = [[self.entity_index[i] for i in sample] for sample in labels]
        return entities, distances

    def serialize(self, storedir: str) -> None:
        '''Serializes a scann index to a specified directory
        ---
        storedir: contains scann artefacts
        '''
        assert isdir(storedir)
        assert self.index is not None

        self.index.save_index(join(storedir, "index.bin"))
        with open(join(storedir, "info.txt"), "w") as f:
            f.write(f"date: {date.today()}\n")
            f.write(f"embed_dim: {self.dim}")

    def load_index(self, precomp_path: str) -> None:
        '''Loads index from a serialized index
        ---
        precomp_path: dir to index.bin
        '''
        assert isdir(precomp_path)
        self.index.load_index(join(precomp_path, "index.bin"))



