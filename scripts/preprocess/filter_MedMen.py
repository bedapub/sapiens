# Remaps training data mentions to high-level GO classes and filters out
# irrelevant branches of the hierarchy

import argparse
import json
import pickle
import re
from pronto.ontology import Ontology
from datetime import date
from os.path import isfile, splitext
from tqdm import tqdm


def filter_maps(filepath, ontopath, outpath, level):
    data = json.load(open(filepath, "r"))
    onto = Ontology(ontopath)

    # irrelevant branches
    class_filter = [
        "GO:0000003", "GO:0022414", "GO:0048511", "GO:0044419", "GO:0043473",
        "GO:0040011", "GO:0051703", "GO:0016032", "GO:0051179", "GO:0110148"]

    # relevant classes
    lvl_classes = []
    if level == 1:
        for term in onto.get_term("GO:0008150").subclasses(1, False): 
            if term.id not in class_filter: lvl_classes.append(term)
    else:
        previous_lvl = []
        for term in onto.get_term("GO:0008150").subclasses(1, False): 
            if term.id not in class_filter: previous_lvl.append(term)
        lvl_classes.extend(previous_lvl)
        for lvl in range(level-1):
            current_lvl = []
            for term in previous_lvl:
                current_lvl.extend(list(term.subclasses(1, False)))
                lvl_classes.extend(current_lvl)
                previous_lvl = current_lvl

    # remap low level classes to high level
    lvl_classes = set(lvl_classes)
    remapped_pairs = set()
    for pair in tqdm(data["data"]):
        term = onto.get_term(pair[1])
        ancestors = list(term.superclasses(with_self=False))
        lvl = [t for t in ancestors if t in lvl_classes]
        for t in lvl:
            tid = int(t.id.lstrip("GO:"))    # convert ID to numerical
            remapped_pairs.add((pair[0], tid))

    # add classes as training data
    for term in lvl_classes:
        tid = int(term.id.lstrip("GO:"))    
        remapped_pairs.add((term.name, tid))

    # overwrite
    data["data"] = list(remapped_pairs)
    data["meta"] = {"date": str(date.today()), "size": len(remapped_pairs)}

    # dump to json
    json.dump(data, open(outpath, "w"))
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        default="../../datasets/MedMentions/corpus_st21pv.json",
        help="path to training data json"
    )
    parser.add_argument(
        "--ontopath",
        default="../../resources/ontologies/GO/go.obo",
        help="gene ontology OBO file"
    )
    parser.add_argument(
        "--outpath",
        default="../../datasets/MedMentions/corpus_st21pv_lvl1.json",
        help="outpath (json)"
    )
    parser.add_argument(
        "--level",
        default=1,
        help='''depth in GO class hierachy to retrieve classes from, depth=0 is
        biological_process'''
    )
    args = parser.parse_args()
    assert(isfile(args.filepath))
    assert(isfile(args.ontopath))
    assert(splitext(args.outpath)[1] == ".json")
    filter_maps(args.filepath, args.ontopath, args.outpath, args.level) 
