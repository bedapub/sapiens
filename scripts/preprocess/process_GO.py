import json
import argparse
from pronto.ontology import Ontology
from os.path import isfile, splitext


def ont2json(filepath, outpath):
    '''parses GO ontology (obo) and
    writes classes (and their synonyms) to json'''
    onto = Ontology(filepath)
    terms = {}
    for term in onto.terms():
        # only keep the biological process branch
        if term.namespace != "biological_process": continue
        go_id = term.id
        name = term.name
        definition = str(term.definition)
        syns = [s.description for s in term.synonyms]
        terms[go_id] = {
            "name": name, "synonyms": syns, "definition": definition
        }

    with open(outpath, 'w') as f:
        json.dump(terms, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        default="../../resources/ontologies/GO/go.obo",
        help="path to go.owl or go.obo"
    )
    parser.add_argument(
        "--outpath",
        default="../../resources/ontologies/GO/go_BP.json",
        help="path to go.owl"
    )

    args = parser.parse_args()
    assert(isfile(args.filepath))
    assert(splitext(args.filepath)[1]  == ".obo")
    assert(splitext(args.outpath)[1]  == ".json")

    ont2json(args.filepath, args.outpath)
