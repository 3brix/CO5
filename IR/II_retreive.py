from whoosh import index
from II_index import index_dir
from whoosh.qparser import MultifieldParser, OrGroup, FuzzyTermPlugin, PhrasePlugin

# Load your index here
ix = index.open_dir(index_dir)

with ix.searcher() as searcher:
    # Define parser and search logic
    # Search here
    pass



