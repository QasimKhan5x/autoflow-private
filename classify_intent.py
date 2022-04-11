import argparse
import json

import torch
from sentence_transformers import SentenceTransformer, util

# intents
with open("task2prompt.json") as f:
    task2prompt = json.load(f)

prompt2task = {
    prompt: task for task in task2prompt for prompt in task2prompt[task]}
# Corpus with example sentences
corpus = list(prompt2task.keys())

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
corpus_embeddings = torch.load('intent_embeddings.pt', device)


def get_most_similar(query, corpus_embeddings, top_k):
    '''
    Get the embeddings in the corpus that are most similar to
    the query string
    '''
    with torch.no_grad():
        # generate query embedding
        query_embedding = embedder.encode(
            query, convert_to_tensor=True)
        # move to device
        query_embedding = query_embedding.to(device)  # type: ignore
    # Find the closest top_k sentences of the corpus for each query sentence based on cosine similarity
    hits = util.semantic_search(
        query_embedding, corpus_embeddings, top_k=top_k)
    # free gpu memory
    torch.cuda.empty_cache()
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


def get_task_from_hits(hits, corpus):
    '''Get the corresponding tasks from the most similar hits'''
    tasks = set()
    for hit in hits:
        corpus_id = hit['corpus_id']
        prompt = corpus[corpus_id]
        task = prompt2task[prompt]
        tasks.add(task)
    return list(tasks)


def get_task_from_query(query, top_k=3, threshold=0.5):
    '''
    Pass a query string
    Get a list of the tasks that may have been intended in the query
    '''
    global corpus_embeddings
    global corpus
    hits = get_most_similar(query, corpus_embeddings, top_k)
    hits_above_thresh = filter_hits(hits, threshold)
    list_of_tasks = get_task_from_hits(hits_above_thresh, corpus)
    return list_of_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the task(s) for a query')
    parser.add_argument('query', type=str, help='the query string')
    args = parser.parse_args()
    tasks = get_task_from_query(args.query)
    print(tasks)
