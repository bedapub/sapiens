# 🐵 SAPIENS

Signature Annotation Pipeline for Entity Normalisation.

## Brief Description

SAPIENS uses neural information retrieval to annotate gene signatures with classes from a structured ontology, e.g. some subclass of `GO:biological_process` or `CL`. 

For fast retrieval, SAPIENS uses a lightweight CNN as an encoder, jointly embedding classes and signatures in the same latent space using text metadata, e.g. signature title and description. Results are fetched from a precomputed embedding index with milisec retrieval times.

Please see the [wiki](https://code.roche.com/PMDA/pred-bioinformatics/gems-suite/sapiens/-/wikis/home) for instructions and information on how to setup SAPIENS, or go to [SAPIENS_API](https://code.roche.com/PMDA/pred-bioinformatics/gems-suite/sapiens_api) for instructions on deployment.

## Example Output

An example of the results over the C7 subset of MSigDB can be found [here](https://docs.google.com/spreadsheets/d/1luLeEuUF3oTQ-RE-kGpJRzRCIBzNlML7qftWbLiYjB0/edit?usp=sharing).

## Disclaimer

SAPIENS has high retrieval accuracy but is imperfect. Some known issues are

* Character-level sensitivity:
    due to the use of pre-trained token embeddings, the static vocabulary does not correctly segment long abbreviations or allow for character-level invariances to spelling mistakes. This should eventually be fixed by incorporating character-level embeddings.
* NIL queries:
    if a relevant term does not exist within the structured vocabulary (GO or CL), then spurrious results may be returned. A NIL component should be incorporated into the pipeline eventually to account for this. Alternatively, more terms can be added from other vocabularies, but this may increase the sensitivity to noise in the query.


## Contributors 

- Miles Henderson (*miles.henderson@roche.com*, alt: *mdhenderson@protonmail.com*) `maintainer` 
- Chia-Huey Ooi (*chia-huey.ooi@roche.com*) `owner` 
# sapiens
