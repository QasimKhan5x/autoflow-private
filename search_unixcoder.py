import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            outputs = self.encoder(
                code_inputs, attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs*code_inputs.ne(1)
                       [:, :, None]).sum(1)/code_inputs.ne(1).sum(-1)[:, None]
            return F.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(
                nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)
                       [:, :, None]).sum(1)/nl_inputs.ne(1).sum(-1)[:, None]
            return F.normalize(outputs, p=2, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')
config = RobertaConfig.from_pretrained('microsoft/unixcoder-base')
model = RobertaModel.from_pretrained('microsoft/unixcoder-base')
model = Model(model)
model.load_state_dict(torch.load(
    'search/search_model.bin', map_location=device))
model.to(device)
model.eval()


class InputFeatures:
    """A single training/test features for a example."""

    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, url):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


def convert_examples_to_features(js, tokenizer):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(
        js['code_tokens']) is list else js['code']
    code_tokens = tokenizer.tokenize(code)[:256-4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>",
                   tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = 256 - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    nl = ' '.join(js['docstring_tokens']) if type(
        js['docstring_tokens']) is list else js['docstring']
    nl_tokens = tokenizer.tokenize(nl)[:128-4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>",
                 tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = 128 - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids,
                         js['url'] if 'url' in js else js['retrieval_idx'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)

        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer))

        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                print("*** Example ***")
                print("idx: {}".format(idx))
                print("code_tokens: {}".format(
                    [x.replace('\u0120', '_') for x in example.code_tokens]))
                print("code_ids: {}".format(
                    ' '.join(map(str, example.code_ids))))
                print("nl_tokens: {}".format(
                    [x.replace('\u0120', '_') for x in example.nl_tokens]))
                print("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids),
                torch.tensor(self.examples[i].nl_ids))


def code_search_inference(code_dataset, embeddings, model, tokenizer, top_k=3,
                          query=None, query_dataset=None, i=None):
    '''
    Given a natural language query Q,
    Find the code segments inside the given code dataset
    that are the most semantically similar to Q

    Either provide a query dataset and an index for the ith query
    OR
    Provide your own query

    Returns a list of top_k code segments that are the most similar to Q
    '''
    if query is None:
        assert query_dataset is not None and i is not None, "Provide NL input OR Query Dataset + Index"
        nl_input = query_dataset[i][1]
    else:
        nl_tokens = tokenizer.tokenize(query)[:128-4]
        nl_tokens = [tokenizer.cls_token, "<encoder-only>",
                     tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = 128 - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id] * padding_length
        nl_input = torch.tensor(nl_ids)
    nl_input = nl_input.unsqueeze(0).to(device)
    nl_embeddings = model(nl_input).cpu().detach().numpy()
    sort_ids = np.argsort(np.matmul(nl_embeddings, embeddings['code'].T),
                          kind='stable')[:, ::-1][0]
    code_examples = code_dataset.examples
    results = []
    for i, code_id in enumerate(sort_ids):
        if i == top_k:
            break
        result = code_examples[code_id]
        code = tokenizer.decode(result.code_ids, skip_special_tokens=True)[14:]
        url = result.url
        results.append((code, url))
    return results


if __name__ == "__main__":
    query_dataset = TextDataset(
        tokenizer, "search/dataset/CSN/python/test.jsonl")
    code_dataset = TextDataset(
        tokenizer, "search/dataset/CSN/python/codebase.jsonl")
    embeddings = np.load('search/embeddings.npz')
