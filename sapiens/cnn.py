import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.nn.init import kaiming_normal_
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from scipy.spatial.distance import cosine
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass 
from os.path import isfile
from warnings import warn


# --- Configs ---------------------------------------------
@dataclass
class CNNConfig:
    '''
    CNN block config, out_dim = len(k_sizes) * n_filters
    ---
    in_dim: dim of input token embeddings 
    out_dim: dim of output token embeddings
    k_sizes: sequence span of conv filter (i.e. num of tokens)
    stride: conv filter stride
    paddings: padding along sequence length
    dropout: dropout
    dilations: 
    '''
    in_dim: int = 256 
    n_filters: int = 256
    out_dim: int = 256
    k_sizes: Tuple[int,...] = (1, 3, 5)
    paddings: Tuple[int,...] = (0, 1, 2)
    k_stride: int = 1
    dilations: Tuple[int,...] = (1, 2, 1)
    dropout: float = 0.1


@dataclass
class ResCNNConfig:
    '''
    Model Configuration
    ---
    embedding_path: path to pretrained word embedding state_dict
    tokenizer: huggingface hub address to model tokenizer
        - cambridgeltl/SapBERT-from-PubMedBERT-fulltext
        - michiyasunaga/BioLinkBERT-large
        NOTE: make sure tokenizer matches embedding
    checkpoint: path to a checkpoint.pt
    block_dim: dim of CNN blocks
    out_dim: final out dim
    depth: number of CNN blocks
    max_length: max length of the *after truncation* for an embedding
    dropout: dropout probability
    init_weights: whether or not to use kaiming normal weight init
    aggregate: "maxpool", "selfattention"
    tagger: whether to include layers for mention detection
    num_labels: number of CRF labels (for binary classes use 2)
    device: "cpu", "cuda", "mps"
    '''
    embedding_path: str = "resources/pretrained/embedding_biolinkbert.pt"
    tokenizer: str = "michiyasunaga/BioLinkBERT-large"
    checkpoint: Optional[str] = None
    vocab_size: int = 28895
    in_dim: int = 1024
    block_dim: int = 256
    out_dim: int = 256
    depth: int = 2
    max_length: int = 25
    dropout: float = 0.15
    init_weights: bool = False
    aggregate: str = "attentionpool"
    tagger: bool = True
    num_labels: int = 2
    device: str = "cpu"

    def load(self, dct: dict):
        for key, val in dct.items():
            setattr(self, key, val)

        return self


# --- Modules ----------------------------------------------


class CNNblock(nn.Module):
    '''CNN block captures local token dependencies in a sentence matrix.
    Adapted from https://arxiv.org/abs/1408.5882
    '''
    def __init__(self, config: CNNConfig = CNNConfig()):
        super().__init__()
        # setup parameters
        c = config
        assert len(c.k_sizes) == len(c.paddings)

        # convolution layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=c.in_dim, 
                out_channels=c.n_filters, 
                kernel_size=k, 
                stride=c.k_stride, 
                padding=p
            ) 
            for k, p in zip(c.k_sizes, c.paddings)
        ]) 

        # projection ensure input and output shapes match
        h_dim = len(c.k_sizes) * c.n_filters
        self.ffnn = nn.Linear(h_dim, c.out_dim)
        self.dropout = nn.Dropout(c.dropout)
        
    def forward(self, X):
        '''
        X: has shape (N, D, L)
        Di: dimension of embedding
        L: length of sequence
        k: number of unique kernel sizes
        ---
        returns H: has shape (N, D_out, L)
        '''
        residual = X                                      # (N, D0, L)
        outputs = [conv(X) for conv in self.convs]        # [(N, D1, L),...]
        outputs = [F.relu(h) for h in outputs]            # [(N, D1, L),...]
        H = torch.cat(outputs, dim=1)                     # (N, k*D1, L)
        H = self.ffnn(H.mT).mT                            # (N, D2, L)
        H = self.dropout(H)
        H += residual

        return H


class ID_CNNblock(nn.Module):
    '''ID-CNN block of iterated dilated convolutions.
    Adapted from https://aclanthology.org/D17-1283.pdf
    '''
    def __init__(self, config: CNNConfig = CNNConfig()):
        super().__init__()
        # setup parameters
        c = config
        assert len(c.k_sizes) == len(c.paddings)

        # convolution layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=c.in_dim, 
                out_channels=c.n_filters, 
                kernel_size=3, 
                dilation=d,
                padding=d,
            ) 
            for d in c.dilations
        ]) 

        # projection ensure input and output shapes match
        h_dim = len(c.k_sizes) * c.n_filters
        self.ffnn = nn.Linear(c.out_dim, c.out_dim, bias=False)
        self.dropout = nn.Dropout(c.dropout)
        
    def forward(self, X):
        '''
        X: has shape (N, D, L)
        Di: dimension of embedding
        L: length of sequence
        k: number of unique kernel sizes
        ---
        returns H: has shape (N, D_out, L)
        '''
        residual = X                                      # (N, D0, L)
        H = X
        for conv in self.convs:
            H = F.relu(conv(H))                           # [(N, D1, L),...]
        H = self.ffnn(H.mT).mT
        H = self.dropout(H)
        H += residual

        return H


class AttentionPool(nn.Module):
    '''A self-attention pooling procedure'''
    def __init__(self, in_dim: int):
        super().__init__()
        self.Q = nn.Linear(in_dim, 1, bias=False)

    def forward(self, X):
        '''X: has shape (N, D, L)
        ---
        returns H: has shape (N, D)
        '''
        q = self.Q(X.mT)                                 # (N, L, D)
        attn = F.softmax(q, dim=1)                       # (N, L, 1)
        H = torch.sum(attn * X.mT, dim=1)                # (N, D)

        return H


class CrossAttention(nn.Module):
    '''A cross-attention layer'''
    def __init__(self, dim: int):
        super().__init__()
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.sqrt_dim = torch.sqrt(torch.tensor(dim))

    def forward(self, X1: torch.Tensor , X2: torch.Tensor):
        '''
        X1: has shape (N, D1, L)
        X2: has shape (N, D2, L)
        ---
        returns H: has shape (N, D1, L)
        '''
        attn = F.softmax(self.Q(X1.mT) @ self.K(X2.mT).mT, dim=-1)
        H = (attn @ self.V(X2.mT)).mT

        return H


class ScaledDotProdAttention(nn.Module):
    '''A scaled dot-product attention layer'''
    def __init__(self, dim: int):
        super().__init__()
        self.Q = torch.nn.Linear(dim, dim)
        self.K = torch.nn.Linear(dim, dim)
        self.V = torch.nn.Linear(dim, dim)
        self.sqrt_dim = torch.sqrt(torch.tensor(dim))

    def forward(self, X):
        '''X: has shape (N, D, L)
        ---
        returns H: has shape (N, D, L)
        '''
        q = self.Q(X.mT)                                 # (N, L, D)
        k = self.K(X.mT)                                 # (N, L, D)
        attn = F.softmax(q @ k.mT / self.sqrt_dim, dim=-1)
        H = (attn @ self.V(X.mT)).mT

        return H


class ResCNN(nn.Module):
    '''A lightweight model for entity linking and mention detection
    ---
    modified and adapted from https://doi.org/10.48550/arXiv.2109.02237
    '''
    def __init__(self, config: ResCNNConfig = ResCNNConfig()):
        super().__init__()
        self.config = config
        self.device = config.device

        # token embedder
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.embedder = BertEmbeddings(
            BertConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.in_dim
            )
        )
        if isfile(self.config.embedding_path): self.__load_embeddings()

        # projection to lower dimension after bert embedding
        self.proj = nn.Sequential(
            nn.Linear(config.in_dim, 512),
            nn.Linear(512, config.block_dim),
        )

        # conv layers: CNNblocks
        layers = []
        for _ in range(config.depth):
            layers.append(
                ID_CNNblock(
                    CNNConfig( 
                        in_dim = config.block_dim,
                        out_dim = config.block_dim,
                        dropout = config.dropout
                    )
                )
            )
            layers.append(
                ScaledDotProdAttention(config.block_dim)
            )
        self.layers = nn.Sequential(*layers)
        
        # aggregation procedure
        if config.aggregate == "maxpool":
            self.agg = lambda X: torch.max(X, dim = -1)[0]
        elif config.aggregate == "attentionpool":
            self.agg = AttentionPool(config.block_dim)
        else: raise AssertionError("aggregation option not available.")

        # projection to output
        self.ffnn = nn.Linear(config.block_dim, config.out_dim)

        # CRF tagger for mention detection
        if config.tagger:
            try: 
                from allennlp.modules import ConditionalRandomField
            except ImportError as e:
                raise Exception("Install allennlp to use CRF tagger") from e

            self.span_mask = None
            self.proj_md = nn.Linear(config.in_dim, config.block_dim)
            self.tag_layers = nn.Sequential(
                nn.Linear(config.block_dim, config.block_dim),
                nn.Linear(config.block_dim, config.block_dim),
                nn.Linear(config.block_dim, config.num_labels)
            )
            self.CRF = ConditionalRandomField(config.num_labels)


        # init weights
        if config.init_weights: 
            self.__init_weights()

        # if checkpoint provided
        if config.checkpoint is not None: self.load_ckpt(config.checkpoint)

        # device and freeze
        self.to(self.device)
        self.freeze_embeds()

    def __init_weights(self):
        '''kaiming normal init weights'''
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, nonlinearity='relu')

    def __load_embeddings(self):
        '''Load token embeddings, freeze, load tokenizer'''
        # load pretrained embedder weights
        ckpt = torch.load(self.config.embedding_path, map_location=self.device)
        self.embedder.load_state_dict(ckpt, strict=False)
        # freeze embedding layers
        for param in self.embedder.parameters(): 
            param.requires_grad = False

    def load_ckpt(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(ckpt["model_state_dict"])

    def freeze_embeds(self, on:bool = True):
        '''freeze token embeddings'''
        for param in self.embedder.parameters(): 
            param.requires_grad = not on
    
    def get_span_mask(self, 
            tokens:BatchEncoding,
            word_id:int = 1
        ) -> torch.Tensor:
        '''Retrieve the span mask from tokens object'''
        h = lambda L: [i if i==word_id else 0 for i in L.word_ids]
        g = lambda L: [h(i) for i in L.encodings]
        f = lambda L: torch.tensor(
            [m if sum(m)>0 else [1]*len(m) for m in g(L)]
        )
        mask = f(tokens).unsqueeze(1).to(self.device)
        return mask

    def bert_base(
        self, batch: Union[List[str], List[List[str]], tuple], 
        return_option: Optional[str] = None
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, BatchEncoding]]:
        '''Tokenizes and retrieves pretrained token embeddings
        ---
        batch: an iterable of strings, has 1D shape (batch_size)
        '''
        if isinstance(batch, tuple): batch = list(batch)

        # tokenize input
        tokens = self.tokenizer(
            batch,
            max_length=self.config.max_length,
            truncation=True,
            padding=True,
            return_token_type_ids=False,
            return_tensors="pt",
            is_split_into_words=True
        )

        # get token ids
        token_ids = tokens["input_ids"].to(self.device) 
        token_mask = tokens["attention_mask"].unsqueeze(-1).to(self.device)
        self.span_mask = self.get_span_mask(tokens)
        
        # embed and apply attention mask
        X = self.embedder(token_ids)
        X = X * token_mask

        # return options
        if return_option == "toks_only":
            return tokens
        elif return_option == "pad_mask":
            return X, tokens["attention_mask"].to(torch.bool).to(self.device)

        return X

    def tagger(
        self,
        batch: Union[List[str], List[List[str]], tuple],
        X: Optional[torch.Tensor] = None,
        pad_mask: Optional[Union[torch.Tensor, BatchEncoding]] = None,
        ner_labels: Optional[torch.Tensor] = None,
        loss: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''NER tagger using a conditional-random-field (CRF) head
        ---
        DIMS
        N : batch sample size
        L : length of sequence
        D : hidden dimension
        M : number of possible entity labels
        ---
        '''
        if X is None:
            # (N, L, 768) 
            X, pad_mask = self.bert_base(batch, return_option="pad_mask")     

        H = self.proj_md(X).mT                      # (N, D, L)
        H = self.layers(H)                          # (N, D, L) conv layers
        S = self.tag_layers(H.mT)                   # (N, M, L)
        tags = self.CRF.viterbi_tags(S, pad_mask)   # (N, 1, L)

        if loss and ner_labels is not None:
            likelihood = self.CRF(S, ner_labels, pad_mask)
            return -likelihood

        return tags

    def forward(self, batch: Union[tuple, list]) -> torch.Tensor:
        '''
        batch : an iterable of lists, has shape [[l, m, r], ...]
        ---
        returns embedding
        '''
        if isinstance(batch, tuple): batch = list(batch)

        X = self.bert_base(batch)              # (N, L, 767) 
        H = self.proj(X).mT                    # (N, D0, L)
        H = self.layers(H)                     # (N, D1, L) conv layers
        H = H * self.span_mask                 # (N, D1, L) mask mention contxt 
        H = self.agg(H)                        # (N, D1) maxpool, self attn
        H = self.ffnn(H)                       # (N, D2) ffnn

        return H

    def debug(self, X: torch.Tensor):
        '''
        X: pre-embedded tokens
        ---
        returns all intermediate hidden layers for debugging
        '''
        hidden = [] 
        H = self.proj(X).mT                     # (N, D0, L)
        hidden.append(H)
        H = self.layers(H)                      # (N, D1, L) conv layers
        hidden.append(H)
        H = self.agg(H)                         # (N, D1) maxpool, self attn
        hidden.append(H)
        H = self.ffnn(H)                        # (N, D2) ffnn
        hidden.append(H)

        return hidden

    def get_uncertainty(self, mention:str, entity:str, iters:int=600) -> float:
        '''Get MCDropout uncertainty, keeps dropout on to conduct
        bayesian inference
        '''
        self.train()
        hats = [
            cosine(
                self([mention]).detach().numpy()[0],
                self([entity]).detach().numpy()[0]
            )
            for i in range(iters)
        ]
        uncertainty = float(np.var(hats))
        self.eval()
        return uncertainty


class Ranker(nn.Module):
    '''A lightweight model for ranking
    ---
    modified and adapted from https://doi.org/10.48550/arXiv.2109.02237
    '''
    def __init__(self, config: ResCNNConfig = ResCNNConfig()):
        super().__init__()
        self.config = config
        self.device = config.device

        # token embedder
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.embedder = BertEmbeddings(
            BertConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.in_dim
            )
        )
        if isfile(self.config.embedding_path): self.__load_embeddings()

        # projection to lower dimension after bert embedding
        self.proj = nn.Sequential(
            nn.Linear(config.in_dim, 512),
            nn.Linear(512, config.block_dim),
        )

        # conv layers: CNNblocks and Attn
        layers = []
        for _ in range(config.depth):
            # CNN block
            layers.append(
                ID_CNNblock(
                    CNNConfig( 
                        in_dim = config.block_dim,
                        out_dim = config.block_dim,
                        dropout = config.dropout
                    )
                )
            )
            # Attention
            layers.append(
                ScaledDotProdAttention(config.block_dim)
            )

        self.layers = nn.Sequential(*layers)

        # Cross attention
        self.cross_attn = CrossAttention(config.block_dim)
        
        # aggregation procedure
        if config.aggregate == "maxpool":
            self.agg = lambda X: torch.max(X, dim = -1)[0]
        elif config.aggregate == "attentionpool":
            self.agg = AttentionPool(config.block_dim)
        else: raise AssertionError("aggregation option not available.")

        # projection to output
        self.ffnn = nn.Linear(config.block_dim, config.out_dim)

        # init weights
        if config.init_weights: 
            self.__init_weights()

        # if checkpoint provided
        if config.checkpoint is not None: self.load_ckpt(config.checkpoint)

        # device
        self.to(self.device)

    def __init_weights(self):
        '''kaiming normal init weights'''
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, nonlinearity='relu')

    def __load_embeddings(self):
        '''Load token embeddings, freeze, load tokenizer'''
        # load pretrained embedder weights
        ckpt = torch.load(self.config.embedding_path, map_location=self.device)
        self.embedder.load_state_dict(ckpt, strict=False)
        # freeze embedding layers
        for param in self.embedder.parameters(): 
            param.requires_grad = False

    def load_ckpt(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(ckpt["model_state_dict"])

    def freeze_embeds(self, on:bool = True):
        '''freeze token embeddings'''
        for param in self.embedder.parameters(): 
            param.requires_grad = not on

    def bert_base(
        self, batch: Union[List[str], List[List[str]], tuple], 
        return_option: Optional[str] = None
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, BatchEncoding]]:
        '''Tokenizes and retrieves pretrained token embeddings
        ---
        batch: an iterable of strings, has 1D shape (batch_size)
        '''
        if isinstance(batch, tuple): batch = list(batch)

        # tokenize input
        tokens = self.tokenizer(
            batch,
            max_length=self.config.max_length,
            truncation=True,
            padding=True,
            return_token_type_ids=False,
            return_tensors="pt",
            is_split_into_words=True
        )

        # get token ids
        token_ids = tokens["input_ids"].to(self.device) 
        token_mask = tokens["attention_mask"].unsqueeze(-1).to(self.device)
        
        # embed and apply attention mask
        X = self.embedder(token_ids)
        X = X * token_mask

        # return options
        if return_option == "toks_only":
            return tokens
        elif return_option == "pad_mask":
            return X, tokens["attention_mask"].to(torch.bool).to(self.device)

        return X

    def forward(self, query: list, doc: list) -> torch.Tensor:
        '''
        query: an iterable of lists, has shape [[""], ...]
        doc: an iterable of lists, has shape [[""], ...]
        ---
        returns rank score
        '''

        X1, X2 = self.bert_base(query), self.bert_base(doc)  # (N, L, D) 
        H1, H2 = self.proj(X1).mT, self.proj(X2).mT          # (N, D0, L)
        H1, H2 = self.layers(H1), self.layers(H2)            # (N, D1, L)
        H = self.cross_attn(H1, H2)                          # (N, D1, L)
        H = self.agg(H)                                      # (N, D1)
        H = self.ffnn(H)                                     # (N, D2) 

        return H
