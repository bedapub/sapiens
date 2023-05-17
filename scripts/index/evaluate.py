import argparse
import json
from torch import tensor
from sapiens.utils import RetrievalEvaluator
from sapiens.cnn import ResCNN, ResCNNConfig


def main(args):
    '''evaluate and print results'''
    evaluator = RetrievalEvaluator(args.ontojsonpath)
    model = ResCNN(ResCNNConfig(
        checkpoint=args.checkpoint,
        device=args.device
    ))
    model.eval()

    # infer dataset
    evaluator.load(model, args.verbose)

    # eval testset
    testset = json.load(open(args.testset))["data"]
    inputs = [i[0] for i in testset]
    labels = tensor([i[1] for i in testset])

    results = evaluator(inputs, labels)
    print("RESULTS:")
    print(f"acc@1:{results[0]}, acc@k:{results[1]}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="./resources/checkpoints/ResCNN_checkpoint_noCRF.pt",
        help="rescnn checkpoint"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="model device"
    )
    parser.add_argument(
        "--ontojsonpath",
        default="./resources/ontologies/GO/go_BP_subset.json",
        help="path to onto file where entities are saved"
    )
    parser.add_argument(
        "--testset",
        default="./datasets/GO/test/test_GO_BP.json",
        help="path to onto file where entities are saved"
    )
    parser.add_argument(
        "--verbose",
        default=True
    )
    args = parser.parse_args()
    main(args)
