import os
import pickle
import shutil
from tqdm import tqdm
from whoosh import index
from I import medline_folder
from whoosh.fields import Schema, TEXT, ID, KEYWORD


# Schema
schema = Schema(
        doc_id=ID(stored=True, unique=True),
        title=TEXT(stored=True),
        abstract=TEXT(stored=True),
        mesh=KEYWORD(stored=True, commas=True)
)

# Index
index_dir = "pubmed_index"


def get_index():
    """create or open whoosh index."""
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    if index.exists_in(index_dir):
        ix=index.open_dir(index_dir)
    else:
        ix=index.create_in(index_dir, schema)

    return ix


def index_data():
    """Load pickle files and add documents to the whoosh index"""
    pickle_path=os.path.join(medline_folder, "pmid2content.pkl")
    if not os.path.isfile(pickle_path):
        raise FileNotFoundError("Pickle file not found. Run I first.")
    
    with open(pickle_path, "rb") as F:
        pmid2content = pickle.load(F)

    ix=get_index()
    writer = ix.writer(limitmb=512, procs=1, multiseqment=True)  # i have low RAM

    for pmid, data in tqdm(pmid2content.items()):
        writer.add_document(
            doc_id=str(pmid),
            title=data["title"],
            abstract=data["abstract"],
            mesh=",".join(data["mesh_terms"]) if data["mesh_terms"] else ""
        )
    
    writer.commit()
    print(f"Indexed{len(pmid2content)} documents.")



if __name__ == "__main__":
    get_index()
    index_data()