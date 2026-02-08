# IV MeSH prediction and evaluation  (attemted to solve task, but can not test it... )
# i tried it with the 3 pickle files i was able to process with my setup...
# it runs, but the results are not meaningful (both 0.0)

import os
import pickle
from random import shuffle
from tqdm import tqdm
from whoosh.fields import Schema, TEXT, ID
from whoosh import index, scoring
import shutil
#from I_fetch_pubmed import medline_folder
from whoosh.qparser import QueryParser

# constants
medline_folder = "pmid2contents"  
train_size = 2       #1000000      
test_size = 1        #1000         
index_dir = "mesh_eval_index"  

if os.path.exists(index_dir):
    shutil.rmtree(index_dir)
os.mkdir(index_dir)

# schema
schema = Schema(pmid=ID(stored=True, unique=True),content=TEXT(stored=True))

ix = index.create_in(index_dir, schema)
writer = ix.writer()


# load records
all_pmids = []

# create test set
for pkl_file in os.listdir(medline_folder):
    if pkl_file.endswith(".pkl"):
        with open(os.path.join(medline_folder, pkl_file), "rb") as f:
            all_pmids.extend(list(pickle.load(f).keys()))

# shuffle (rearranges data in random order)
shuffle(all_pmids)
test_set_pmids = set(all_pmids[:test_size])
test_data = {}



# create and index training set
for pkl_file in os.listdir(medline_folder):
    if not pkl_file.endswith(".pkl"):
        continue
    with open(os.path.join(medline_folder, pkl_file), "rb") as f:
        obj = pickle.load(f)
        for pmid, data in obj.items():
            if len(data) < 3 or not data[2]:
                continue
            title, abstract, mesh = data
            text = f"{title} {abstract}"

            if pmid in test_set_pmids:
                test_data[pmid] = (text, mesh)
            else:
                writer.add_document(pmid=str(pmid), content=text)

writer.commit()

# evaluation
with ix.searcher(weighting=scoring.BM25F()) as searcher: # okapi BM25 (best matching), tf-idf?
    qp = QueryParser("content", ix.schema, group=OrGroup)  # or grouping

# sanity check
def mesh_to_query(mesh):
    return " ".join(mesh) if isinstance(mesh, list) else str(mesh)

# counters
correct_at_1 = 0
correct_at_5 = 0
total = 0

# searcing using best matching and or logic
with ix.searcher(weighting=scoring.BM25F()) as searcher:
    qp = QueryParser("content", ix.schema, group=OrGroup)

    print("Evaluating test abstracts...")
    for pmid, (_, mesh) in tqdm(test_data.items()):
        query_str = mesh_to_query(mesh)
        query = qp.parse(query_str)

        results = searcher.search(query, limit=5)
        hit_pmids = [hit["pmid"] for hit in results]

        total += 1
        if hit_pmids:
            if hit_pmids[0] == str(pmid):
                correct_at_1 += 1
            if str(pmid) in hit_pmids:
                correct_at_5 += 1

# accuracies
acc_at_1 = correct_at_1 / total
acc_at_5 = correct_at_5 / total

print("Evaluation:")
print(f"Accuracy@1: {acc_at_1:.4f}")
print(f"Accuracy@5: {acc_at_5:.4f}")