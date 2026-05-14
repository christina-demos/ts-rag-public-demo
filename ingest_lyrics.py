from pathlib import Path
import hashlib
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize ChromaDB
client = chromadb.PersistentClient(path="./swift_vec_db")

# 2. Embedding model
model_name = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_name
)

# 3. Reset collection
try:
    client.delete_collection("taylor_lyrics")
    print("Old collection deleted.")
except Exception:
    pass

collection = client.create_collection(
    name="taylor_lyrics",
    embedding_function=embedding_func
)

def make_chunk_id(row):
    raw = f"{row['album']}|{row['song']}|{row['section']}|{row['text']}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

# 4. Load all chunk CSVs
chunk_files = sorted(Path("data/chunks").glob("*_chunks.csv"))

total_chunks = 0

for file in chunk_files:
    print(f"Ingesting {file.name}...")

    df = pd.read_csv(file)

    # Remove empty lyric chunks
    df = df.dropna(subset=["text"])

    # Remove duplicate rows inside CSV
    df = df.drop_duplicates(subset=["album", "song", "section", "text"])

    documents = [
        # Create one string per row
        f"Song: {row['song']}\nAlbum: {row['album']}\nSection: {row['section']}\nLyrics:\n{row['text']}"
        for _, row in df.iterrows()
    ]

    metadatas = [
        {
            "song": row["song"],
            "album": row["album"],
            "section": row["section"]
        }
        for _, row in df.iterrows()
    ]

    ids = [
        make_chunk_id(row)
        for _, row in df.iterrows()
    ]

    # Send everything to DB
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    total_chunks += len(df)

print(f"Database populated with {total_chunks} chunks!")