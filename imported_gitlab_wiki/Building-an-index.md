## Background

For fast entity retrieval, SAPIENS uses `hnswlib`. This allows one to quickly retrieve entities using approximate nearest neighbors search to get the top-k closest embeddings in a precomputed index. SAPIENs currently uses cosine similarity as the default distance measure, due to contrastive training with the same measure. Please refer to the `BuildConfig` class in `sapiens/index.py` if a larger index (>100K) is necessary to adjust the build parameters.

To build an index one needs the following:

* Pretrained / pre-fit embedding model, e.g. `sapiens.cnn.ResCNN` or `sapiens.semantic.Embedder`
* A set of entity names, e.g. `json` or in `obo`

Please see the wiki pages on training `ResCNN` or fitting the LSA `semantic.Embedder` for more information on training embedding models.

## Basic usage

Two quick-and-easy scripts for building an index using a custom embedder are implemented as `scripts/generate_index_*.py`. Classes and utils for index construction are implemented in `sapiens/index.py`, for reference.

To build an LSA index minimally:

```
python scripts/fit_lsa.py
python scripts/generate_index_lsa.py
```

To build a ResCNN index, first train ResCNN model. See the argument docs for the required input and then do:
```
python scripts/generate_index_rescnn.py
```

To note:

* The default arguments assume one has pretrained/prefit embedder, and corresponding path to the serialized model at `resources/checkpoints/...`.
* Inference time using the `ResCNN` model is extremely fast due the optimised tensor operations in PyTorch, whereas the LSA embedder `semantic.Embedder` is at least an order of magnitude slower. This can be optimized by the maintainer at some point by using the `onnxruntime` environment. See [sklearnonnx](http://onnx.ai/sklearn-onnx/auto_examples/plot_tfidfvectorizer.html) for more info. The `ResCNN` model can also be potentially optimized.
