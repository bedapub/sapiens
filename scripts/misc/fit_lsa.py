'''Script to prefit LSA embedder.
'''
import argparse
from os.path import isdir
from sapiens.semantic import fit_vectorizer, fit_reducer
from sklearn.decomposition import TruncatedSVD


def main(args):
    # fit TFIDF vectorizer
    fit_vectorizer(
        entpath=args.ontopath,
        savedir=args.savedir,
        verbose=args.verbose
    )
    # fit SVD reducer
    fit_reducer(
        precomp_dir=args.savedir,
        reducer=TruncatedSVD(args.dim, random_state=42),
        verbose=args.verbose
    )
    print("LSA embedder fit complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--ontopath",
        default = "resources/ontologies/GO/go_BP_subset.json",
        help = "json or obo file of entities"
    )
    parser.add_argument(
        "--dim",
        default = 128,
        help = "dimension of embeddings",
        type = int
    )
    parser.add_argument(
        "--savedir",
        default = "resources/lsa_embeds/lsa_GO/",
        help = "path to save fitted vectorizer, reducer"
    )
    parser.add_argument(
        "--verbose",
        default = True
    )
    args = parser.parse_args()
    assert isdir(args.savedir)
    main(args)
