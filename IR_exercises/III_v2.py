from whoosh import index
from II import index_dir
from whoosh.qparser import MultifieldParser, OrGroup, AndGroup, FuzzyTermPlugin, PhrasePlugin

# load index
ix = index.open_dir(index_dir)

def search_example(query_str, group="OR", fuzzy=False, phrase=False, top_k=5):
    """
    Perform a search with optional OR/AND, fuzzy, and phrase.
    """
    # OR or AND
    if group.upper()=="OR":
        parser=MultifieldParser(["title", "abstract"], schema=ix.schema, group=OrGroup)
    else:
        parser=MultifieldParser(["title", "abstract"], schema=ix.schema, group=AndGroup)

    # plugins
    if fuzzy:
        parser.add_plugin(FuzzyTermPlugin())
    if phrase:
        parser.add_plugin(PhrasePlugin())

    # parse 
    query = parser.parse(query_str)

    # search
    with ix.searcher() as searcher:
        results = searcher.search(query, limit=top_k)
        print(f"\nQuery: {query_str} | Group: {group} | Fuzzy: {fuzzy} | Phrase: {phrase}")
        print(f"Found {len(results)} results.\n")
        for hit in results:
            print("PMID:", hit["doc_id"])
            print("Title:", hit["title"])
            print("Abstract:", hit["abstract"][:200], "...")
            print("-" * 200)

if __name__ == "__main__":
    # query examples
    search_example("diabetes", group="OR")
    search_example("diabetes", group="AND")
    search_example('"diabetes"', phrase=True)
    search_example("diabtes~", fuzzy=True)
