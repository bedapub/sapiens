Source code at `sapiens/cnn.py`

## Model architecture

ResCNN uses one-dimensional convolutions to aggregate local information. A basic forward pass consists of the following components:

1. Tokenize: `"Stimulation of CD4 T-cells" -> ["stimulation", "of", "CD4" "T-#", "##cells"]`
2. Embed tokens using pre-trained embeddings, here we use the first layer from one of  
    - `cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token`
    - `michiyasunaga/BioLinkBERT-large`
    - `microsoft/biogpt` (experimental, not yet production ready due to use of a non-fast tokenizer from `transformers` library)
3. Linear projection of embeddings to lower dimension
4. ID-CNN blocks (see paper on dilated convolutions) with residual connections
4. Self-attention for weighted aggregation, this can be thought of as "averaging" the token embeddings
5. Linear projection of aggregated embedding to lower dimension

Please refer to the following papers for a better understanding of some of the architectural choices:
1. [ResCNN](https://arxiv.org/abs/2109.02237)
2. [iterated dilated (ID) convolutions](https://aclanthology.org/D17-1283.pdf)
