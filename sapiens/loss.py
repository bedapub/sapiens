import torch
from pronto.ontology import Ontology
from typing import List, Tuple, Union
from torch.nn import MSELoss
from pytorch_metric_learning.reducers import MeanReducer 
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.distances import CosineSimilarity
from sapiens.utils import AncestorIterator, num2ontoID


class OntologicalTripletLoss(BaseMetricLossFunction):
    """A custom adpative triplet loss that computes a "dendritic margin" that
    weights each triplet by the distance of two entity labels from their 
    common ancestor. Assumes that target labels are structured in an ontology.
    ---
    ontopath: path to a .obo ontology file
    margin: the margin range to use when computing the dendritic margin,
        recommend choosing a range such as (0.1, 0.3). Few of the weighted 
        margins will appear near the maximum.
    swap: Use the positive-negative distance instead of anchor-negative distance,
          if it violates the margin more.
    smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
        self,
        ontopath,
        margin=(0.1, 0.3),
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        max_depth=11,
        distance=CosineSimilarity(),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.distance = distance
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.onto = Ontology(ontopath) 
        self.max_depth = max_depth
        self.device = "cpu"
        self.precomp_ancs_map()

    def precomp_ancs_map(self) -> None:
        self.ancestor_map = {}
        for term in self.onto.terms():
            self.ancestor_map[term.id] = [(term.id, 0.0)] + [
                (a.id, d) for a,d in AncestorIterator(term)
            ]

    def get_dendritic_distance(self, pair: List[int]) -> float:
        '''Uses ontology to compute distance to LCA between an anchor
        entity and its negative entity'''
        # convert from numerical to string
        go1 = self.onto.get_term(num2ontoID(pair[0]))
        go2 = self.onto.get_term(num2ontoID(pair[1]))
        # get ancestors 
        ancs1 = self.ancestor_map[go1.id]
        ancs2 = self.ancestor_map[go2.id]

        # find Least Common Ancestor (LCA)
        for (ent1, dist1) in ancs1: 
            for (ent2, dist2) in ancs2:
                if ent1==ent2: return max(dist1, dist2)

        # edge case
        return 0.0

    def get_triplet_weights(self,
        anchor_neg_tuples: List[tuple]
        ) -> torch.Tensor:
        '''Given a list of (anchor, negative) tuples, returns weighted 
        triplet margins computed using the distance to their LCA
        '''
        triplet_weights = [
            self.get_dendritic_distance(pair) 
            for pair in anchor_neg_tuples
        ]
        triplet_weights = torch.tensor(triplet_weights).to(self.device)
        triplet_weights = (triplet_weights - 1) / (self.max_depth - 1)
        triplet_weights = (self.margin[0] 
            + (self.margin[1]-self.margin[0]) * triplet_weights)
        return triplet_weights

    def compute_loss(self, 
        embeddings, labels, 
        indices_tuple=None, ref_emb=None, ref_labels=None
        ):
        # get triplets
        indices_tuple = lmu.get_all_triplets_indices(labels)
        anchor_idx, positive_idx, negative_idx = indices_tuple

        # each class only has one instance in the sample (skip)
        if len(anchor_idx) == 0:
            return self.zero_losses()

        # get triplet weights using dendritic distance
        long_labels = labels.unsqueeze(1)
        anchor_neg_tuples = torch.concat(
            (long_labels[anchor_idx], long_labels[negative_idx]), dim=1
        )
        anchor_neg_tuples = anchor_neg_tuples.tolist()
        triplet_weights = self.get_triplet_weights(anchor_neg_tuples)

        # compute triplet distances
        mat = self.distance(embeddings)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        # compute margin violations and loss
        current_margins = self.distance.margin(ap_dists, an_dists)
        violation = current_margins + triplet_weights
        if self.smooth_loss:
            loss = torch.nn.functional.softplus(violation)
        else:
            loss = torch.nn.functional.relu(violation)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def get_default_reducer(self):
        return MeanReducer()


class CosineSimilarityLoss:
    '''Loss function for computing sentence similarity'''
    def __init__(self):
        self.sim = CosineSimilarity()
        self.error = MSELoss()
        
    def __call__(self, source: torch.Tensor, target: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        y_pred = self.sim(source, target)
        loss = self.error(y_pred, score)
        return loss
