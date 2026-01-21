from whoosh import index
from II import index_dir
from whoosh.qparser import MultifieldParser, OrGroup, FuzzyTermPlugin, PhrasePlugin

# Load your index here
ix = index.open_dir(index_dir)

with ix.searcher() as searcher:
    # Define parser and search logic
    parser=MultifieldParser(["title","abstract"], schema=ix.schema)
    query=parser.parse("diabetes")
    
    results=searcher.search(query, limit=5)

    for hit in results:
        print(hit["doc_id"], hit["title"])


