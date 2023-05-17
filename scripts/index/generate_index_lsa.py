'''Script to infer and build a ScaNN index from a prefit LSA embedder.
'''
import argparse
from os.path import isdir
from sapiens.index import OntologyIndex, BuildConfig
from sapiens.semantic import Embedder


def main(args):
    # init and load model checkpoint
    print("loading model")
    try:
        # load LSA model
        model = Embedder(precomp_dir=args.precomp_dir)
    except Exception as e:
        raise RuntimeError(
            '''loading model unsuccesful, check whether correct paths
            was provided or you are running this script from the highest lvl
            directory.''') from e

    # infer dataset
    print("call embedder to infer class vectors")
    index = OntologyIndex(ontopath=args.ontopath)
    index.infer_dataset(model, verbose=True)

    # build index
    print("building index")
    N = args.leaves * len(index.entity_index)
    N2 = args.leaves_search * N
    index.build_index(BuildConfig())

    # save index
    print(f"saving index to {args.savepath}")
    index.serialize(args.savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--ontopath",
        default = "resources/ontologies/GO/go_BP_subset.json",
        help = "json file of entities"
    )
    parser.add_argument(
        "--leaves",
        default = 0.75,
        help = "leaves to partition"
    )
    parser.add_argument(
        "--leaves_search",
        default = 0.75,
        help = "leaves to search"
    )
    parser.add_argument(
        "--k",
        default = 25,
        help = "number of top-k entities to retrieve per search"
    )
    parser.add_argument(
        "--savepath",
        default = "resources/index/index_lsa_GO/",
        help = "path to save computed index"
    )
    parser.add_argument(
        "--precomp_dir",
        default = "resources/lsa_embeds/lsa_GO/",
        help = "model checkpoint to load before inference"
    )
    args = parser.parse_args()
    assert isdir(args.savepath)
    assert isdir(args.precomp_dir)
    main(args)
