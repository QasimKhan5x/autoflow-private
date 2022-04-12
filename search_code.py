import argparse
import os
import pickle
import re
import tokenize
from io import StringIO

import torch
from rapidfuzz.process import extractOne
from sentence_transformers.util import semantic_search
from transformers import RobertaModel, RobertaTokenizer

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = 'models/search'

search_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
search_model = RobertaModel.from_pretrained(model_path).to(device)


def preprocess(source, lang):
    # remove extra new lines
    source = re.sub("\n+", "\n", source)
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def create_embeddings(input_json, seq_length=64, lang='python'):
    '''create embeddings for user codebase'''
    tokenized_code_segments = list()
    # tokenize the programs and add them to tokenized_code_tensors
    for program in input_json:
        # convert to dict
        if not isinstance(program, dict):
            program = dict(program)
        contents = program['content']
        # split by double new lines to separate code blocks
        contents_list = re.split(r'\n{2,}', contents)
        for content in contents_list:
            pp_contents = preprocess(content, lang)
            # tokenized list with id's
            tokens = search_tokenizer.encode(pp_contents)
            # remove this document if #(tokens) < 3
            if len(tokens) < 3:
                continue
            # append the first segment of length seq_length
            tokenized_code_segments.append(tokens[:seq_length])
            # append every other segment
            for i in range(seq_length, len(tokens), seq_length - 1):
                # next seq_length tokens
                next_n_tokens = tokens[i:i + seq_length - 1]
                # remove this segment if #(tokens) < 3
                if len(next_n_tokens) < 3:
                    break
                # add [CLS] tokens
                code_segment_tokens = [0] + next_n_tokens
                # append to list of tensors
                tokenized_code_segments.append(code_segment_tokens)
    # initialize code_embeddings
    code_embeddings = None
    for segment_tensor in tokenized_code_segments:
        # create tensor from tokens
        code_tensor = torch.tensor(segment_tensor).unsqueeze(0).to(device)
        # disable gradient computation
        with torch.no_grad():
            # create embedding from tensor
            code_vec = search_model(code_tensor)[1].to(device)
        # first tensor
        if code_embeddings == None:
            code_embeddings = code_vec
        # concatenate with all embeddings
        else:
            code_embeddings = torch.cat((code_embeddings, code_vec), 0)
    # save objects
    with open("tokenized_code_segments.pkl", "wb") as f:
        pickle.dump(tokenized_code_segments, f)
    torch.save(code_embeddings, "code_embeddings.pt")
    # free memory
    del tokenized_code_segments
    del code_embeddings
    torch.cuda.empty_cache()


def get_most_similar(query, top_k):
    '''
    Get the embeddings in the code embedddings
    that are most similar to the query string
    '''
    code_embeddings = torch.load("./code_embeddings.pt", map_location=device)
    query_tokens = search_tokenizer.encode(
        query, return_tensors="pt").to(device)
    query_embedding = search_model(query_tokens)[1]
    # Find the closest top_k sentences of the corpus for each query sentence based on cosine similarity
    hits = semantic_search(query_embedding, code_embeddings, top_k=top_k)
    # Get the hits for the first query
    hits = hits[0]
    return hits


def filter_hits(hits, threshold):
    '''Keep only those hits that are >= threshold'''
    new_hits = []
    for hit in hits:
        if hit['score'] >= threshold:
            new_hits.append(hit)
    return new_hits


def get_code_from_hits(hits):
    '''Get the corresponding code from the most similar hit(s)'''
    if len(hits) > 0:
        code_segments = []
        for hit in hits:
            # the index inside the corpus
            code_segment_index = hit['corpus_id']
            # get the list of all code segments
            with open("tokenized_code_segments.pkl", "rb") as f:
                tokenized_code_segments = pickle.load(f)
            # the tokens list for this code segment
            code_segment_tokens = tokenized_code_segments[code_segment_index]
            # free memory
            del tokenized_code_segments
            # the original code segment
            code_segment = search_tokenizer.decode(code_segment_tokens,
                                                   skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=True
                                                   )
            code_segments.append(code_segment)
        return code_segments
    else:
        return ''


def search_for_code_segment(query, lang='python', top_k=1,
                            threshold=0.4, recreate=False, input_json=None):
    '''
    Pass a query string
    Return a JSON array with intended code and its filepath
    '''
    # Create embeddings if they don't exist
    # recreate if recreate=True
    if not os.path.exists("./code_embeddings.pt") or recreate:
        if input_json != None:
            create_embeddings(
                input_json, seq_length=search_tokenizer.model_max_length // 4, lang=lang)
        else:
            print("ERROR: EMBEDDINGS DO NOT EXIST")
            return ''

    hits = get_most_similar(query, top_k)
    hits_above_thresh = filter_hits(hits, threshold)
    code_segments = get_code_from_hits(hits_above_thresh)
    # just get the first one...
    return code_segments[0]


def get_original_code_segment(query, input_json, lang):
    '''Use fuzzy string matching to find the code segment
    in the original code using the preprocessed code segment
    found after semantic search'''
    code_segment = search_for_code_segment(query)
    lines = []
    for program in input_json:
        if not isinstance(program, dict):
            program = dict(program)
        content = program['content']
        # split by code segments separated by more than 2 newlines
        segments = re.split(r'\n{2,}', content)
        lines.extend(segments)
    return extractOne(code_segment, lines, score_cutoff=0.7)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('query',
                        help='query string to search for code')
    parser.add_argument('--input_json',
                        help='path to json containing codebase files and their content')

    args = parser.parse_args()
    result = search_for_code(args.query, input_json=args.input_json)
    print(result)
