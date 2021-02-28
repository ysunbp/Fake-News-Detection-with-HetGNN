from typing import List, Tuple, Optional
import torch
import os
from transformers import XLMRobertaModel, XLMRobertaTokenizer

class TextEmbedder(torch.nn.Module):
    """An embedder for string-to-2D-Tensor conversion with XLM-RoBERTa or word2vec"""

    def __init__(self, max_seq_len : int, model_name: str, model_path: Optional[str] = ''):
        super(TextEmbedder, self).__init__()
        """
        Parameters
        ----------
        max_len : int

        model_name : str
            The name of the model, should be one of 'word2vec', 'xlm-roberta-base', 'xlm-roberta-large'
        model_path : str, optional
            The path to the w2v file / finetuned Transformer model path. Required for w2v.
        """
        
        assert model_name in ['word2vec', 'xlm-roberta-base', 'xlm-roberta-large']
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        model_path = model_name if model_path == '' else model_path
        print('TextEmbedder: Loading model {} ({})'.format(model_name, model_path))
        if model_name == 'word2vec':
            assert os.path.isfile(model_path)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', model_max_length=self.max_seq_len+1)
            self._load_weibo_w2v(model_path)
        else:
            assert model_path in ['xlm-roberta-base', 'xlm-roberta-large'] or os.path.isdir(model_path)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, model_max_length=self.max_seq_len)
            self.model = XLMRobertaModel.from_pretrained(model_path, return_dict=True)
        print('TextEmbedder: Finished loading model {}'.format(model_name))

    def forward(self, text_list: List[str], return_tokens: Optional[bool] = False) -> Tuple[torch.Tensor, List[List[str]]]:
        """Embeds a list of text into a torch.Tensor

        Parameters
        ----------
        text_list : List[str]
            Each item is a piece of text
        return_tokens: Optional[bool] = False
            For debug only. It slows down everything if you're using a Transformer, so don't use it unless necessary.

        Returns 
        ----------
        outputs: torch.Tensor
            Embedding of shape (len(text_list), max_seq_len, embed_dim)
        tokens: List[List[str]]
            (if return_tokens=True) A list of tokenized original text (padded for Transformers; not padded for w2v)
        """

        if self.model_name == 'word2vec':
            tokens = [self.tokenizer.tokenize(text)[1: 1 + self.max_seq_len] for text in text_list]
            tokens = [[token.replace('▁', '') for token in doc] for doc in tokens]  # NOTE '▁' and '_' (underscore) are different
            outputs = self._w2v_embed(tokens)
            return (outputs, tokens) if return_tokens else outputs
        else:
            inputs = self.tokenizer(text_list, return_tensors="pt", max_length=self.max_seq_len, padding='max_length', truncation=True)
            outputs = self.model(**inputs)
            if return_tokens:
                tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']]
                return outputs, tokens
            else:
                return outputs
        

    def _load_weibo_w2v(self, model_path):
        self.w2v = dict()
        with open(model_path) as w2v_file:
            lines = w2v_file.readlines()
        info, vecs = lines[0], lines[1:]
        
        info = info.strip().split()
        self.vocab_size, self.embed_dim = int(info[0]), int(info[1])

        for vec in vecs:
            vec = vec.strip().split()
            self.w2v[vec[0]] = [float(val) for val in vec[1:]]

    def _w2v_embed(self, docs):
        ### wait, what are the vocab of the pre-trained? 
        outputs = torch.zeros((len(docs), self.max_seq_len, self.embed_dim))  # no valid words => zeros
        for i, doc in enumerate(docs):
            for j, token in enumerate(doc[:self.max_seq_len]):
                outputs[i, j] = torch.tensor(self.w2v.get(token, 0))
        return outputs


if __name__ == '__main__':
    text_list = ['酷！艾薇儿现场超强翻唱Ke$ha神曲TiK ToK！超爱这个编曲！ http://t.cn/htjA04', '转发微博。']
    word2vec_path = '/rwproject/kdd-db/shsuaa/fyp/word2vec/sgns.weibo.bigram-char'
    # finetuned_transformer_path = '/rwproject/kdd-db/20-rayw1/language_models/xlm-roberta-base'
    finetuned_transformer_path = '/rwproject/kdd-db/20-rayw1/language_models/xlm-roberta-base-post'
    max_seq_len = 49

    embedder = TextEmbedder(max_seq_len, 'xlm-roberta-base', finetuned_transformer_path)
    outputs, tokens = embedder(text_list, return_tokens=True)
    print(outputs.last_hidden_state.shape)
    print(outputs.pooler_output.shape)
    print(tokens)

    embedder = TextEmbedder(max_seq_len, 'word2vec', word2vec_path)
    outputs, tokens = embedder(text_list, return_tokens=True)
    print(outputs.shape)
    print(tokens)
