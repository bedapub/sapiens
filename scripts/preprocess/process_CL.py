import json
import argparse
from pronto.ontology import Ontology
from os.path import isfile, splitext


def ont2json(filepath, outpath):
    '''Parses CL ontology (obo) and writes classes (and their synonyms) to 
    json
    ---
    filepath : path to the .obo file
    outpath : path to a .json file where the processed class names will be
        saved.
    '''
    onto = Ontology(filepath)
    terms = {}
    for term in onto.terms():
        cl_id = term.id
        name = term.name
        syns = [s.description for s in term.synonyms]
        terms[cl_id] = {"name": name, "synonyms": syns}

    with open(outpath, 'w') as f:
        json.dump(terms, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        default="../../resources/ontologies/CL/cl-basic.obo",
        help="path to go.owl or go.obo"
    )
    parser.add_argument(
        "--outpath",
        default="../../resources/ontologies/CL/cl-basic.json",
        help="path to go.owl"
    )

    args = parser.parse_args()
    assert(isfile(args.filepath))
    assert(splitext(args.filepath)[1] == ".obo")
    assert(splitext(args.outpath)[1] == ".json")

    ont2json(args.filepath, args.outpath)
