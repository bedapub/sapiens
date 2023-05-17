import argparse
import json
import pickle
import re
import numpy as np
from datetime import date
from os.path import isfile, splitext
from typing import Tuple, List
from utils import get_mention_context


def filter_greek(string: str):
    '''Replaces greek unicode w/ anglicization filter all other unicode'''
    ge_dict = {
        "\u03b1": "alpha",
        "\u03b2": "beta",
        "\u03b3": "gamma",
        "\u03b4": "delta",
        "\u03ba": "kappa",
    }
    for key, val in ge_dict.items():
        string = string.replace(key, val)
    string = re.sub("u0.{0,3}", "", string)
    return string


class Filter_go:
    def __init__(self, go_subset):
        self.subset = go_subset
    
    def __call__(self, pair):
        if pair[1] in self.subset: return True
        else: return False


def add_processed_mentions(
        rawtext: str, mention_spans: list, go_ids: list, annotations: list
    ) -> None:
    '''Adds mention annotations and spans of other mentions within the context 
    to article-specific list of annotations
    ---
    PARAMS
    rawtext : source article txt
    mention_spans : character spans of all mentions in raw_text
    go_ids : GO IDs corresponding to mention_spans
    annotations : list of article specific annotations to append to
    ---
    '''
    if rawtext == "" or mention_spans == []: pass

    # generate character indicator vector for raw text
    indicator = np.zeros((len(rawtext)), dtype=int)
    for span in mention_spans: 
        indicator[span[0]:span[1]] = 1

    # add contextual mention, goid, and indicator subvector to annotations
    for span, goid in zip(mention_spans, go_ids):
        mention_context, m_spans = get_mention_context(
            rawtext, span, mention_spans
        )
        annotations.append((mention_context, goid, m_spans))


def reset_tmp_vars():
    '''Reset temporary variables'''
    return [], []


def pubtator_to_json(filepath, mapspath, filterpath, outpath):
    # init vars
    go_ids, mention_spans = reset_tmp_vars()
    raw_text = ""
    annotations = []
    
    # load mappings
    with open(mapspath, 'r') as f: 
        umls2go_maps = json.load(f)
    

    # process annotated corpus
    with open(filepath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            line_list = line.split("\t")
            if len(line.split("|")) > 1: 
                ### process a title
                if line[9] == "t": 
                    # add previously processed mentions to annotations
                    add_processed_mentions(
                        raw_text, 
                        mention_spans,
                        go_ids,
                        annotations
                    )
                    # reset article-specific temp vars
                    go_ids, mention_spans = reset_tmp_vars()
                    raw_text = line.split("|")[2]
                # process an abstract
                elif line[9] == "a":  
                    raw_text += line.split("|")[2]
                continue
            # ignore line if not delineated by a tab
            elif len(line_list) == 1: continue

            # filter out irrelevant codes
            T_code = line_list[4]
            if T_code != "T038": continue    

            # preprocess and map from UMLS to GO
            umls_cui = line_list[-1].rstrip().lstrip("UMLS:")
            if umls_cui in umls2go_maps:
                # get mention and filter
                mention = line_list[3]
                mention = filter_greek(mention)
                go_id = umls2go_maps[umls_cui]

                # get context of mention
                mention_span = [int(i) for i in line_list[1:3]]

                # add to mention_pairs
                mention_spans.append(mention_span)
                go_ids.append(go_id)


        # filter out GO terms that are not in a specified GO subset
        go_subset = pickle.load(open(filterpath, "rb"))
        filter_go = Filter_go(go_subset)
        annotations = [i for i in annotations if filter_go(i)]
        annotations = [
            (i[0], int(i[1].lstrip("GO:")), i[2]) for i in annotations
        ]

        # dump to json
        meta = {"date": str(date.today()), "size": len(annotations)}
        json.dump({"data": annotations, "meta": meta}, fout)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        default="../../datasets/MedMentions/corpus_st21pv.txt",
        help="path to corpus.txt"
    )
    parser.add_argument(
        "--mapspath",
        default="../../resources/xrefs/umls2go.json",
        help="path to json of mappings from UMLS to GO"
    )
    parser.add_argument(
        "--filterpath",
        default="../../resources/ontologies/GO/go_subset.pkl",
        help="path to a relevant subset of GO terms (optional)"
    )
    parser.add_argument(
        "--outpath",
        default="../../datasets/MedMentions/corpus_st21pv_lvln_context.json",
        help="path to corpus.json"
    )
    args = parser.parse_args()
    assert(isfile(args.filepath))
    assert(isfile(args.mapspath))
    assert(splitext(args.mapspath)[1] == ".json")
    pubtator_to_json(args.filepath, args.mapspath, args.filterpath, args.outpath)
