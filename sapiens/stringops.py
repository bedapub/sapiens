import re
import spacy
from spacy.matcher import DependencyMatcher
from typing import Optional


class CheckStimulus:
    def __init__(self):
        '''Regex to check if perturbation by some stimulus'''
        patterns = [
            r"(?i)(t)reat\w+",
            r"(?i)(u)ntreat\w+",
            r"(?i)(s)tim\w+",
            "response to",
            "exposure to",
            "by",
        ]
        self.patterns = [re.compile(p) for p in patterns]
        
    def __call__(self, text) -> bool:
        result = any([bool(p.search(text)) for p in self.patterns])
        return result


class SubstringExtractor:
    def __init__(self, nlp = None):
        '''Extracts object of stimulus to refine entity retrieval'''
        # load model
        if nlp is None: self.nlp = spacy.load("en_core_sci_sm")
        else: self.nlp = nlp

        self.nlp.add_pipe("merge_noun_chunks")
        self.nlp.add_pipe("merge_entities")
        self.checkstim = CheckStimulus()
        self.matcher = DependencyMatcher(self.nlp.vocab)

        # add verb-object patterns "stimulated -> IL2"
        self.matcher.add("left-right",
            [[{"RIGHT_ID": "verb", 
                "RIGHT_ATTRS": {"POS": "VERB", "ORTH": {"NOT_IN": ["compared"]}}
              }, 
              {"LEFT_ID": "verb",
               "REL_OP": ">>", "RIGHT_ID": "obj",
               "RIGHT_ATTRS": {
                   "POS": {"IN": ["NOUN", "PROPN"]},
                   "DEP": {"IN": ["nmod", "conj"]}
                }
              }], 
             [{"RIGHT_ID": "response", 
               "RIGHT_ATTRS": {"ORTH": "response"}
              },
              {"LEFT_ID": "response",
               "REL_OP": ">", "RIGHT_ID": "obj",
               "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
              },
              {"LEFT_ID": "response",
               "REL_OP": ">>", "RIGHT_ID": "to",
               "RIGHT_ATTRS": {"ORTH": "to"}
              }],
             [{"RIGHT_ID": "exposure", 
               "RIGHT_ATTRS": {"ORTH": "exposure"}
              },
              {"LEFT_ID": "exposure",
               "REL_OP": ">", "RIGHT_ID": "obj",
               "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
              }]
            ]
        )

        # add compound noun patterns "IL2 <- stimulation"
        self.matcher.add("right-left",
            [[{"RIGHT_ID": "stim", 
                "RIGHT_ATTRS": {"ORTH": {"IN": ["stimulation", "exposure"]}}
              }, 
              {"LEFT_ID": "stim",
               "REL_OP": ">", "RIGHT_ID": "agent",
               "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
              }]
            ]
        )

        # add control vs. perturbation patterns "control / untreated vs. X"
        self.matcher.add("compare-perturb",
            [[{"RIGHT_ID": "compare", 
                "RIGHT_ATTRS": {
                    "ORTH": {"IN": ["versus", "vs", "compared"]}
                }
              }, 
              {"LEFT_ID": "compare",
               "REL_OP": ".*", "RIGHT_ID": "perturb",
               "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["GGP", "CHEBI"]}}
              }
              ]
            ]
        )

    def __call__(self, text:str) -> set:
        if self.checkstim(text):
            doc = self.nlp(text)   
            matches = self.matcher(doc)
            matches = [m[1] for m in matches]
            if len(matches) > 0:
                l = len(matches[0])
                results = set(
                    ["response to " + str(doc[m[-1]]) for m in matches]
                )
                return results
            else: return set()
        else: return set()
