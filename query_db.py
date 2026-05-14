import chromadb
from chromadb.utils import embedding_functions
from difflib import SequenceMatcher


DB_PATH = "./swift_vec_db"
COLLECTION_NAME = "taylor_lyrics"
MODEL_NAME = "all-MiniLM-L6-v2"

# Determine if two lyric chunks are near-duplicates.
def too_similar(a, b, threshold=0.82):
    return SequenceMatcher(None, a, b).ratio() >= threshold

# Query extra results, then keep only the best diverse matches.
def query_diverse(collection, query, final_k=5, fetch_k=25):
    results = collection.query(
        query_texts=[query],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"]
    )

    filtered = []
    seen_song_sections = set()

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        key = (meta["song"], meta["section"])

        # Skip repeated song/section matches
        if key in seen_song_sections:
            continue

        # Skip lyrics that are too similar to results already kept
        if any(too_similar(doc, kept["document"]) for kept in filtered):
            continue

        filtered.append({
            "document": doc,
            "metadata": meta,
            "distance": distance
        })

        seen_song_sections.add(key)

        if len(filtered) == final_k:
            break

    return filtered


# 1. Connect to ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)

# 2. Load same embedding model used during ingestion
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME
)

# 3. Load collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func
)

# 4. Query with diversity filtering
matches = query_diverse(
    collection,
    query="I want to break up with my boyfriend",
    final_k=5,
    fetch_k=25
)

# 5. Print results
for match in matches:
    meta = match["metadata"]

    print("=" * 50)
    print(f"Song: {meta['song']}")
    print(f"Album: {meta['album']}")
    print(f"Section: {meta['section']}")
    print(f"Distance: {match['distance']}")
    print()
    print(match["document"])