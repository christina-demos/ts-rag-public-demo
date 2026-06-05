import streamlit as st

# CHANGE:
# Import load_collection so Streamlit can cache the Chroma collection/model.
from retrieve_and_rerank_public import run_lyric_search, load_collection


st.set_page_config(
    page_title="Taylor Swift Lyric Match",
    page_icon="🎧",
    layout="centered"
)


# CHANGE:
# Cache the Chroma collection + embedding model.
# This prevents Streamlit from reloading them every time the user searches.
@st.cache_resource(show_spinner=False)
def get_cached_collection():
    return load_collection()


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fff7fb 0%, #f8f4ff 45%, #ffffff 100%);
    }

    h1 {
        color: #2f243a;
        letter-spacing: -0.04em;
    }

    .subtitle {
        font-size: 1.08rem;
        color: #6f6078;
        margin-bottom: 1.5rem;
    }

    div.stButton > button {
        border-radius: 999px;
        padding: 0.8rem 1.4rem;
        font-weight: 700;
        background: #6f4aa2;
        color: white;
        border: none;
    }

    div.stButton > button:hover {
        background: #5f3f8d;
        color: white;
        border: none;
    }

    textarea {
        border-radius: 18px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.88);
        border-radius: 22px;
        box-shadow: 0 10px 30px rgba(80, 50, 100, 0.07);
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Taylor Swift Lyric Match")

st.markdown(
    '<div class="subtitle">Describe a situation. Get song-section matches ranked by emotion, POV, and timeline.</div>',
    unsafe_allow_html=True
)

# CHANGE:
# Added a clear public-demo note so the wait feels intentional, not broken.
st.caption(
    "Public demo: lyric text is hidden for copyright reasons. Matches are based on song-section metadata, so some results may be less precise."
    "Matching may take up to a minute, especially on the first search."
)


user_query = st.text_area(
    "What are you going through?",
    placeholder="Example: I miss my ex but I know they're bad for me",
    height=130
)

# CHANGE:
# Fixed extra space in button text.
search_clicked = st.button("Find my matches")

NUM_RESULTS = 5
MIN_SCORE = 7


if search_clicked:
    if not user_query.strip():
        st.warning("Please enter a situation first.")
    else:
        # CHANGE:
        # Load the cached collection once and pass it into the search function.
        with st.spinner("Loading model and finding the best matches..."):
            collection = get_cached_collection()

            matches = run_lyric_search(
                user_query=user_query,
                num_results=NUM_RESULTS,
                min_score=MIN_SCORE,
                collection=collection
            )

        if not matches:
            st.info("No strong matches found. Try rephrasing your situation.")
        else:
            st.success(f"Found {len(matches)} match{'es' if len(matches) != 1 else ''}.")

            for i, match in enumerate(matches, start=1):
                meta = match.get("metadata", {})

                score = match.get("rerank_score", "")
                song = meta.get("song", "Unknown song")
                album = meta.get("album", "Unknown album")
                section = meta.get("section", "")
                reason = match.get("reason", "")
                analysis = match.get("analysis", "")
                match_type = match.get("match_type", "")
                state_alignment = match.get("state_alignment", "")
                section_is_about = match.get("section_is_about", "")
                narrator_state = match.get("narrator_state", "")
                distance = match.get("distance", "")
                low_confidence = match.get("low_confidence", False)

                with st.container(border=True):
                    if low_confidence:
                        st.caption("Low-confidence near match")

                    st.subheader(f"{i}. {song}")
                    st.caption(f"{album} • {section}")

                    st.markdown(f"**Score:** `{score}/10`")

                    if reason:
                        st.markdown("**Why it matches:**")
                        st.write(reason)

                    st.info("Lyrics are hidden in the public demo for copyright reasons.")
                    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

                    with st.expander("More details"):
                        st.write(f"**Match type:** {match_type}")
                        st.write(f"**State alignment:** {state_alignment}")
                        st.write(f"**Section is about:** {section_is_about}")
                        st.write(f"**Narrator state:** {narrator_state}")
                        st.write(f"**Distance:** {distance}")

                        if analysis:
                            st.write("**Analysis:**")
                            st.write(analysis)