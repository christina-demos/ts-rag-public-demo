from pathlib import Path
import hashlib
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration & Paths ---
DB_PATH = "./swift_vec_db_public"
COLLECTION_NAME = "taylor_lyrics"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNKS_DIR = Path("data/chunks")
PROFILE_FILE = Path("data/song_profiles.json")


# Key is album|||song
def load_song_profiles():
    if not PROFILE_FILE.exists():
        print("No data/song_profiles.json found. Continuing without profiles.")
        return {}

    with open(PROFILE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# Helper: convert profile list into comma-separated strings
def list_to_text(value):
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


# Convert profile metadata into searchable text
def get_profile_text(profile):
    if not profile:
        return ""

    themes = list_to_text(profile.get("themes", []))
    moods = list_to_text(profile.get("moods", []))
    situations = list_to_text(profile.get("situations", []))
    good_for_user_roles = list_to_text(profile.get("good_for_user_roles", []))
    bad_for_user_roles = list_to_text(profile.get("bad_for_user_roles", []))

    return f"""
Song Profile:
Themes: {themes}
Moods: {moods}
Perspective: {profile.get("perspective", "")}
Speaker Role: {profile.get("speaker_role", "")}
Narrator Agency: {profile.get("narrator_agency", "")}
Narrator Hurt Status: {profile.get("narrator_hurt_status", "")}
Relationship Stage: {profile.get("relationship_stage", "")}
Timeline State: {profile.get("timeline_state", "")}
Good For User Roles: {good_for_user_roles}
Bad For User Roles: {bad_for_user_roles}
Situations: {situations}
Summary: {profile.get("summary", "")}
""".strip()

# Create stable ID from row content
def make_chunk_id(row):
    raw = f"{row['album']}|{row['song']}|{row['section']}|{row['text']}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def main():
    # 1. Connect to local persistent Chroma database.
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    # 2. Use the same embedding model for ingestion and querying.
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    # 3. Delete old collection so updated profile-enriched documents replace old embeddings.
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Old collection deleted.")
    except Exception:
        pass

    # 4. Create fresh collection for lyric vectors and metadata.
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    # 5. Load song-level profiles, JSON keyed by "album|||song"
    song_profiles = load_song_profiles()

    # 6. Load all chunk CSVs.
    chunk_files = sorted(CHUNKS_DIR.glob("*_chunks.csv"))
    total_chunks = 0

    for file in chunk_files:
        print(f"Ingesting {file.name}...")

        df = pd.read_csv(file)

        df = df.dropna(subset=["text"])
        df = df.drop_duplicates(subset=["album", "song", "section", "text"])

        # Create batches
        documents = []
        metadatas = []
        ids = []

        for _, row in df.iterrows():
            profile_key = f"{row['album']}|||{row['song']}"
            profile = song_profiles.get(profile_key, {}) # Dictionary, good for .get()
            profile_text = get_profile_text(profile)     # Text block, good for embedding

            # This is the text that gets embedded
            # It includes song-level POV/timeline context plus the specific lyric chunk
            document = f"""
Song: {row['song']}
Album: {row['album']}
Section: {row['section']}

{profile_text}
""".strip()

            documents.append(document)

            metadatas.append({
                "song": row["song"],
                "album": row["album"],
                "section": row["section"],

                "themes": list_to_text(profile.get("themes", [])),
                "moods": list_to_text(profile.get("moods", [])),
                "situations": list_to_text(profile.get("situations", [])),

                "perspective": profile.get("perspective", ""),
                "speaker_role": profile.get("speaker_role", ""),
                "narrator_agency": profile.get("narrator_agency", ""),
                "narrator_hurt_status": profile.get("narrator_hurt_status", ""),
                "relationship_stage": profile.get("relationship_stage", ""),
                "timeline_state": profile.get("timeline_state", ""),

                "good_for_user_roles": list_to_text(profile.get("good_for_user_roles", [])),
                "bad_for_user_roles": list_to_text(profile.get("bad_for_user_roles", [])),

                "summary": profile.get("summary", "")
            })

            ids.append(make_chunk_id(row))

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