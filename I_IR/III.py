from whoosh import index
from II import index_dir
from whoosh.qparser import MultifieldParser, OrGroup, FuzzyTermPlugin, PhrasePlugin

# Load your index here
ix = index.open_dir(index_dir)

def search(text, mode="or", fuzzy=False, limit=5):
    with ix.searcher() as searcher:
        # Define parser and search logic
        group = OrGroup if mode == "or" else AndGroup
        parser=MultifieldParser(["title","body"], schema=ix.schema, group=group)

        if fuzzy:
            parser.add_plugin(FuzzyTermPlugin())

        parser.add_plugin(PhrasePlugin())

        query=parser.parse(text)
        
        results=searcher.search(query, limit=limit)

        for hit in results:
            print(hit["id"], hit["title"])


# output in IR_I-IV_modal.ipynb
search("schizophrenia")                             # basic search
search("schizophrenia albinism")                    # mode: orgroup (default)
search("schizophrenia albinism", mode="and")        # mode: andgroup
search("schiz*phrenia", fuzzy=True)                 # fuzzy search (handles misspelling /typos)
search("behavioral therapy for schizophrenia")      # phrase search


