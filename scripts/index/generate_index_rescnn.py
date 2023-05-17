'''Script to infer and build a ScaNN index from a pretrained ResCNN 
model checkpoint.
'''
import argparse
from os.path import isdir
from sapiens.index import OntologyIndex, BuildConfig
from sapiens.cnn import ResCNN, ResCNNConfig
from torch.backends import mps


def main(args):
    # init and load model checkpoint
    print("loading model")
    try:
        model = ResCNN(ResCNNConfig(
            embedding_path = "resources/pretrained/embedding_biolinkbert.pt",
            tokenizer = "michiyasunaga/BioLinkBERT-large",
            vocab_size = 28895,
            aggregate = "attentionpool",
            dropout = 0,
            depth=2,
            in_dim=1024,
            block_dim=256,
            out_dim=256,
            max_length=100
        ))
        model.load_ckpt(args.checkpoint)
        model.eval()
    except Exception as e:
        raise RuntimeError(
            '''loading model unsuccesful, check whether correct paths
            provided in config or you are running this script from the base 
            directory.''') from e


    # infer dataset
    print("call embedder to infer class vectors")
    index = OntologyIndex(ontopath=args.ontopath)
    index.infer_dataset(model, verbose=True)

    # build index
    print("building index")
    index.build_index(BuildConfig(ef_construction=1000, ef=500, distance_measure="cosine"))

    # save index
    print(f"saving index to {args.savepath}")
    index.serialize(args.savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--ontopath",
        default = "./resources/ontologies/GO/go_signalling_subset.json",
        help = "json file of entities"
    )
    parser.add_argument(
        "--savepath",
        default = "resources/index/index_STIM/",
        help = "path to save computed index"
    )
    parser.add_argument(
        "--checkpoint",
        default = "./resources/checkpoints/run6/checkpoint_e0.pt",
        help = "model checkpoint to load before inference"
    )
    args = parser.parse_args()
    assert isdir(args.savepath)
    main(args)
