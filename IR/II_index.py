import os
import pickle
import shutil
from tqdm import tqdm
from whoosh import index
from I_fetch_pubmed import medline_folder
from whoosh.fields import Schema, TEXT, ID


# Schema
schema = 'Define your schema here'

# Index
index_dir = "pubmed_index"


def get_index():
    # Create your index here
    return

if __name__ == "__main__":
    get_index()

