import os
import json
import pandas as pd
from I_constants import *
from II_index import load_all_documents
from itertools import product
from tqdm import tqdm
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def evaluate_index(db, test_data, k=3):
    """
    Computes the average F1 score for retrieval across all test questions.
    Contains a nested get_f1 function for self-contained logic.
    """
    
    def get_f1(true_set, pred_set):
        """Calculates F1 score between two sets of sources."""
        if not pred_set or not true_set:
            return 0.0
        
        tp = len(true_set.intersection(pred_set))
        precision = tp / len(pred_set)
        recall = tp / len(true_set)
        
        if (precision + recall) == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)

    retriever = db.as_retriever(search_kwargs={"k": k})
    f1s = []
    
    for entry in test_data:
        query = entry['text']
        true_sources = set(entry['sources'])
        
        # retrieve docs and extract filenames from metadata
        retrieved_docs = retriever.invoke(query)
        pred_sources = {doc.metadata["source"].split('/')[-1] for doc in retrieved_docs}
      
        f1s.append(get_f1(true_sources, pred_sources))
    
    return np.mean(f1s)


if __name__ == "__main__":
    # load test data and docs
    test_data = json.load(open('../squad/squad_multiple_contexts.json', 'r'))
    all_docs = load_all_documents(source_path)
    
    # search space
    chunk_sizes = [125, 250, 500]    # chunk size: chunck the sections into menegeable chunks depending on the type on embedding u use
    overlaps = [0, 25, 50, 75]       # chunk overlap: ensures if chunks have interdependencies, you make sure you have a good coverage, good
    k_values = [1, 3, 5]             # k:nr. of chunks, to play around with, if too high --> noisy; if to low: you loose critical info; context size = k x chuck_size
    
    results = []
    
    # initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, cache_folder=models_path)

    for size, overlap in product(chunk_sizes, overlaps):
        print(f"\nTesting Size: {size}, Overlap: {overlap}")
        
        # split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        split_texts = splitter.split_documents(all_docs)
        
        # temporary in-memory index
        db = Chroma.from_documents(
            documents=split_texts,
            embedding=embeddings,
            collection_name="temp_eval"
        )
        
        for k in k_values:
            score = evaluate_index(db, test_data, k=k)
            print(f"K={k} -> F1: {score:.4f}")
            results.append({
                "chunk_size": size,
                "overlap": overlap,
                "k": k,
                "f1_score": score
            })
            
        db.delete_collection()

    # summarize
    df = pd.DataFrame(results).sort_values("f1_score", ascending=False)

    # save
    df.to_csv("hyperparameter_results.csv", index=False)


# Findings: Smaller chunks (125–250 tokens) work better for this task, because they reduce irrelevant noise and seem to produce more precise embeddings.
# Adding overlap is important when chunks cut context mid-sentence. Increasing K generally boosts F1 at first, but adding more chunks often yields little benefit or even hurts performance.
# The sweet spot here is the original k=3, extra retrieved chunks are more likely to add noise than useful information.