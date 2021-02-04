from typing import List, Tuple, Union, Optional
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer

class TextEmbedder:
    """An embedder for string-to-2D-Tensor conversion with XLM-RoBERTa or word2vec"""

    def __init__(self, model_name: str, w2v_path: Optional[str] = '/rwproject/kdd-db/shsuaa/fyp/word2vec/sgns.weibo.bigram-char'):
        """
        Parameters
        ----------
        model_name : str, optional
            The name of the model, should be one of 'word2vec', 'xlm-roberta-base', 'xlm-roberta-large'
        w2v_path : str, optional
            The path to the w2v file. Defaults to the one on server
        """
        
        self.model_name = model_name
        print('TextEmbedder: Loading models...')
        if model_name == 'word2vec':
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base') ### what vocab?
            self._load_weibo_w2v(w2v_path)
        elif model_name in ['xlm-roberta-base', 'xlm-roberta-large']:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
            self.model = XLMRobertaModel.from_pretrained(model_name, return_dict=True)
        else:
            raise Exception("model_name should be one of 'word2vec', 'xlm-roberta-base', 'xlm-roberta-large'")
        print('TextEmbedder: Finished loading models')

    def __call__(self, text_list: List[str], return_tokens: Optional[bool] = False) -> Tuple[torch.Tensor, List[List[str]]]:
        """Embeds a list of text into a torch.Tensor

        Parameters
        ----------
        text_list : List[str]
            Each item is a piece of text

        Returns 
        ----------
        outputs: torch.Tensor
            Embedding of shape (len(text_list), x, embed_dim), where x is the max len of tokens in text_list
        tokens: List[List[str]]
            A list of tokenized original text
        """

        if self.model_name == 'word2vec':
            tokens = [self.tokenizer.tokenize(text)[1:] for text in text_list]
            tokens = [[token.replace('▁', '') for token in doc] for doc in tokens]  # NOTE '▁' and '_' (underscore) are different
            outputs = self._w2v_embed(tokens)
            return (outputs, tokens) if return_tokens else outputs
        else:
            inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)['last_hidden_state']
            if return_tokens:
                tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']]
                return outputs, tokens
            else:
                return outputs
        

    def _load_weibo_w2v(self, w2v_path):
        self.w2v = dict()
        with open(w2v_path) as w2v_file:
            lines = w2v_file.readlines()
        info, vecs = lines[0], lines[1:]
        
        info = info.strip().split()
        self.vocab_size, self.embed_dim = int(info[0]), int(info[1])

        for vec in vecs:
            vec = vec.strip().split()
            self.w2v[vec[0]] = [float(val) for val in vec[1:]]

    def _w2v_embed(self, docs):
        ### wait, what are the vocab of the pre-trained? 
        max_seq_len = max([len(doc) for doc in docs])
        outputs = torch.zeros((len(docs), max_seq_len, self.embed_dim))  # no valid words => zeros
        for i, doc in enumerate(docs):
            for j, token in enumerate(doc):
                outputs[i, j] = torch.tensor(self.w2v.get(token, 0))
        return outputs


if __name__ == '__main__':
    text_list = ['酷！艾薇儿现场超强翻唱Ke$ha神曲TiK ToK！超爱这个编曲！ http://t.cn/htjA04', '转发微博。']

    embedder = TextEmbedder('xlm-roberta-base')
    outputs, tokens = embedder(text_list, return_tokens=True)

    embedder = TextEmbedder('word2vec')
    outputs, tokens = embedder(text_list, return_tokens=True)

