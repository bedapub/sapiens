To train our mention-detection and entity linking CNN, we use the following corpora:

- [CRAFT v.5.0.2](https://github.com/UCDenver-ccp/CRAFT/releases/tag/v5.0.2)
- [MedMentions SHA-1457823](https://github.com/chanzuckerberg/MedMentions/commit/14578232bb6cdf290fea9dd7fee60a3e01ee6fd5)

For fine-tuning entity representations we use the following manually constructed datasets for this project:

- `datasets/MsigDB/*.json`
- `datasets/GO/go_pretrain.json`

## Overview

Both CRAFT and MedMentions contain abstracts or article extracts, where entity spans have been highlighted and linked to a structured vocabulary / ontology. There are formatting differences between corpora, and they use different structured vocabularies. To combine examples from both corpora we performed extensive preprocessing:

- map entities from different vocabularies, e.g. `UMLS` to `GO`
- preprocess entity spans into a unified dataset

Scripts for preprocessing can be found at `scripts/preprocess/`. As an example, to construct the dataset used to train the `GO` model you first need to run 

```
python scripts/preprocess/process_MedMen_GO.py
python scripts/preprocess/filter_MedMen.py
```

Cross-references and specific mappings between structured vocabularies can be found at `resources/xrefs/`, these are required by some of the preprocessing scripts. As an example, we created `umls2go.json` to map between `UMLS` and `GO`, these were extracted from [`MRCONSO.RFF`](https://www.ncbi.nlm.nih.gov/books/NBK9685/) which provides vocabulary sources for concepts in the UMLS.

## Example of sample in training data

This would be a single example from the `datasets/CL/train/*.json` file:

```
[
  [
   "present in PP, namely B-cells, ", 
   "T-cells", 
   "", 
   "B-cells", 
   ", T-cells and ", 
   "dendritic cells", 
   "). This observation is in accordance with the"
  ], 
  451, 
  [1, 3, [5]]
]
```

- The first item in the list is a list of strings that contains both the mention and its context. We use a list of strings since there may be other entities in the local context.
- The second item is the numerical representation of the ontology ID, e.g. `CL:0000451`
- The third indicates which items in the list are entities, the index enclosed in brackets `[5]` is the primary entity that corresponds to `CL:0000451`

## Preprocessing scripts

- `preprocess_*.py` extracts mention spans and vocabulary identifiers, then maps these to the relevant vocabulary (e.g. `GO`)
- `filter_*.py` was used to filter irrelevant terms and reduce the size of the final index, this isn't strictly necessary but highly recommended to improve the retrieval times of the final index.

## Data splits

Datasets are currently split randomly using an 80-10-10 split. For an evaluation of zero-shot generalisation a different split is required.
