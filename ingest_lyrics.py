from pathlib import Path
import hashlib
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration & Paths ---
DB_PATH = "./swift_vec_db"           # Directory where ChromaDB will save data files
COLLECTION_NAME = "taylor_lyrics"    # Vector database collection
MODEL_NAME = "all-MiniLM-L6-v2"      # Transformer model
CHUNKS_DIR = Path("data/chunks")     # Location of pre-processed CSV lyrics
PROFILE_FILE = Path("data/song_profiles.json")  # Context data generated for songs


# Load song profiles (containing themes, moods, perspectives, etc.) if they exist to provide context
def load_song_profiles():
    if not PROFILE_FILE.exists():
        print("No data/song_profiles.json found. Continuing without profiles.")
        return {}

    with open(PROFILE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# Reformat profile dictionary into easy-to-read text block
def get_profile_text(profile):
    if not profile:
        return ""

    themes = ", ".join(profile.get("themes", []))
    moods = ", ".join(profile.get("moods", []))
    situations = ", ".join(profile.get("situations", []))
    perspective = profile.get("perspective", "")
    speaker_role = profile.get("speaker_role", "")
    relationship_stage = profile.get("relationship_stage", "")
    summary = profile.get("summary", "")

    return f"""
    Song Profile:
    Themes: {themes}
    Moods: {moods}
    Perspective: {perspective}
    Speaker Role: {speaker_role}
    Relationship Stage: {relationship_stage}
    Situations: {situations}
    Summary: {summary}
    """.strip()


# Generate ID using MD5 hash of row content, prevents duplicates upon re-running ingestion
def make_chunk_id(row):
    raw = f"{row['album']}|{row['song']}|{row['section']}|{row['text']}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def main():
    # 1. Initialize ChromaDB Persistent Client
    # Ensures vector database saves locally to disk instead of in-memory (RAM)
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # 2. Set up Local Embedding Function
    # Manages tokenization and embeddings
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    # 3. Reset Collection Lifecycle Management
    # Deletes any existing collection
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Old collection deleted.")
    except Exception:
        pass

    # Instantiates fresh Chroma collection for storing lyric vectors + metadata
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    # 4. Load External Thematic Profiles
    song_profiles = load_song_profiles()

    # 5. Pipeline: Process and Ingest Local CSV Data Chunks
    # Processes lyric chunk sheets alphabetically by album name
    chunk_files = sorted(CHUNKS_DIR.glob("*_chunks.csv"))
    total_chunks = 0

    for file in chunk_files:
        print(f"Ingesting {file.name}...")

        # Read specific album chunk sheet into memory
        df = pd.read_csv(file)

        # Data Cleaning: Drop rows missing lyrics
        df = df.dropna(subset=["text"])

        # Data Deduplication: Filter out repeating row definitions to prevent redundant vector operations
        df = df.drop_duplicates(subset=["album", "song", "section", "text"])

        # Initialize local batch arrays, ChromaDB's collection.add() demands all three
        documents = []
        metadatas = []
        ids = []

        # Iterate over individual entries (rows) within album dataframe
        for _, row in df.iterrows():
            # Build look-up key matching structure inside song_profiles.json
            profile_key = f"{row['album']}|||{row['song']}"
            profile = song_profiles.get(profile_key, {})
            profile_text = get_profile_text(profile)

            # Context Stuffing: Assemble text to convert into vector embeddings
            # Merges raw lyrics with AI song profiles, enables vector math to evaluate abstract emotional vibes 
            document = f"""
            Song: {row['song']}
            Album: {row['album']}
            Section: {row['section']}

            {profile_text}

            Lyrics:
            {row['text']}
            """.strip()

            documents.append(document)

            # Metadata Object Mapping: Stored alongside vectors to facilitate frontend display, 
            # database post-filtering (e.g., filtering by Era), or custom scoring functions
            metadatas.append({
                "song": row["song"],
                "album": row["album"],
                "section": row["section"],
                "themes": ", ".join(profile.get("themes", [])),
                "moods": ", ".join(profile.get("moods", [])),
                "situations": ", ".join(profile.get("situations", [])),
                "perspective": profile.get("perspective", ""),
                "speaker_role": profile.get("speaker_role", ""),
                "relationship_stage": profile.get("relationship_stage", ""),
                "summary": profile.get("summary", "")
            })

            # Append stable calculated unique hex string 
            ids.append(make_chunk_id(row))

        # 6. Database Upsert Operations
        # Executes an atomic batch insert/update to ChromaDB, generating text vectorizations concurrently
        if documents:
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        total_chunks += len(df)

    print(f"Database populated with {total_chunks} chunks!")


if __name__ == "__main__":
    main()
