'''Generates contextual mentions for training''' 
import argparse
import json
import numpy as np
import pickle
from tqdm import tqdm
from os import listdir
from os.path import isfile, splitext
#from utils import get_mention_context
from typing import List
from datetime import date
from sapiens.utils import num2ontoID, get_mention_context


#--- main -------------------------------------------------


def main(dirpath, outpath):
    # list of [[contextual_mention, indicator], ...]
    annotations = []

    # process article-specific files
    for fname in tqdm(listdir(dirpath)):
        if splitext(fname)[1] == ".json":
            # get mentions and raw txt
            data = json.load(open(dirpath + fname,"r"))

            # split article text into list of spans and non-spans
            article_txt = data["article_txt"]
            spans = [m[1] for m in data["data"]]

            # extract contextual mentions using mention spans
            for _, span, go_id in data["data"]:
                m, ind = get_mention_context(article_txt, span, spans)
                annotations.append([m, go_id, ind])

    # dump to json
    meta = {
        "date": str(date.today()),
        "size": len(annotations),
        "comment": "processed CRAFT GO_BP concept annotations for training"
    }
    json.dump({"data": annotations, "meta": meta}, open(outpath, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirpath",
        default="../../datasets/CRAFT/CL/json/",
        help="path to preprocessed CRAFT annotations (json files)"
    )
    parser.add_argument(
        "--outpath",
        default="../../datasets/CRAFT/CL/CL_filtered.json",
        help="path to output file"
    )
    args = parser.parse_args()
    assert splitext(args.outpath)[1] == ".json"
    main(args.dirpath, args.outpath)
