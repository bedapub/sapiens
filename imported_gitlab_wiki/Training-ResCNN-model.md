## Model overview

Here are some key points on the architecture, for a detailed overview see the page on [ResCNN model overview](ResCNN-model-overview).

* Default is to used pre-trained SapBERT embeddings (66mb) and BERT tokenizer from `huggingface/transformers`
* Vocab size for SapBERT is 30522
* LSTMs generally require far more parameters, without yielding much benefit for our use case. We use convolutional filters to aggregate local token context instead, `depth` specifies number of conv-blocks, and thus is a proxy for the maximum distance for which contextual information is aggregated.
* The conv layers preserve input sequence length `l`, and a final attention-weighted pooling procedure averages the `l` token embeddings to obtain the final embedding.

## Training procedure used for SAPIENs API

- Training data for `GO:biological_process` entity linking is precomputed using labels from MedMentions and CRAFT. These can be found in `datasets/train`. Subset used for evaluation is found at `datasets/val`. The scripts used to generate these can be found at `scripts/preprocess/`.
- ResCNN was trained on this EL dataset for 4 epochs using the params in `training_config.toml`, e.g. `ontologicalloss`. Final EL performance was 67acc@1 and 89acc@30
- Further contrastive training was done using the model checkpoint to improve entity representations. The `go_pretrain.json` dataset was used here with a `multisimilarityloss` and `lr=5e-5`, all other params identical. This improves recall at higher k.
- (Optional) a very small manually curated set of sentence pairs `datasets/BIOSSES` and `datasets/MsigDB/` can be used to further fine-tune. This showed promising but variable results and needs to be experimented with to avoid over-fitting.

## Loss functions

1. `multisimilarityloss` is taken from the pytorch metric-learning library, see [here](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/). It's recommended to use this loss function for development and testing.
2. `ontologicalloss` is a custom loss function for this project that weights contrastive pairs by their "dendritic distance", which we define as the distance to the least common ancestor between two entities in an ontology. The default distance measure used to contrast embeddings is cosine similarity. Computing this distance leads to a significant overhead, but leads the model to converge after fewer epochs. This loss function also yields better retrieval performance (+5%acc@10). See `sapiens/loss.py` for the implementation.

## Practical training details

1. Pre-trained token embeddings can be downloaded [here](blank). They are omitted from the repository due to their size (>250mb combined). Make sure the corresponding path is then at the `embedding_path` attribute of `ResCNNConfig`.

2. An example `bsub` job script can be found in `resources/`. 
    - Submit this on the HPC with `bsub < example_job.bsub`.
    - Training takes approximately 8 hours using `ontologicalloss` and 1.5 hours using `multisimilarityloss`, the former is recommended for entity-linking performance.
    - The default training configuration is found in `configs/train_config.toml`, it is only recommended to adjust the loss function and training hyperparameters after referring to the source code in `train.py` or `train_gpt.py`. Comments describing the relevance of each hyperparam are in the config file.
