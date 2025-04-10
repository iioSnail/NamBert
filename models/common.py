import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.activations import GELUActivation


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class BERT(nn.Module):
    tokenizer = None

    def __init__(self, model_path="hfl/chinese-roberta-wwm-ext", dropout=0.1):
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_path)

    def forward(self, inputs):
        return self.bert(**inputs)

    @staticmethod
    def get_tokenizer(model_path="hfl/chinese-roberta-wwm-ext"):
        if BERT.tokenizer is None:
            BERT.tokenizer = AutoTokenizer.from_pretrained(model_path)

        return BERT.tokenizer

    @staticmethod
    def get_bert_inputs(sentences, tokenizer=None, max_length=256, model_path="hfl/chinese-roberta-wwm-ext"):
        if tokenizer is None:
            tokenizer = BERT.get_tokenizer(model_path)

        inputs = tokenizer(sentences,
                           padding=True,
                           max_length=max_length,
                           return_tensors='pt',
                           truncation=True)
        return inputs


class BertOnlyMLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, activation='gelu', layer_num=1):
        super().__init__()
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

        self.activation = None
        if activation == 'gelu':
            self.activation = GELUActivation()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise Exception("Please add activation function here.")

        self.heads = []

        for i in range(layer_num):
            self.heads.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                self.activation,
                nn.LayerNorm(hidden_size, eps=1e-12, elementwise_affine=True),
            ))

        self.predictions = nn.Sequential(
            *self.heads,
            self.decoder
        )

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        return self.predictions(sequence_output)
