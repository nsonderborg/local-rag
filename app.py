"""
Streamlit webgrænseflade til lokal RAG.
Importerer al RAG-logik fra rag.py uden at ændre den.
Kør med: streamlit run app.py
"""

import ollama
import streamlit as st

from rag import LLM_MODEL, build_index, build_prompt, retrieve

st.set_page_config(page_title="Lokal RAG", page_icon="📚", layout="centered")
st.title("📚 Lokal RAG")
st.caption("HuggingFace · ChromaDB · Ollama")


@st.cache_resource(show_spinner="Indlæser indeks og embedding-model...")
def load_index():
    return build_index()


collection, embed_model = load_index()

query = st.chat_input("Stil et spørgsmål...")

if query:
    with st.chat_message("user"):
        st.write(query)

    chunks = retrieve(query, collection, embed_model)
    prompt = build_prompt(query, chunks)

    with st.chat_message("assistant"):
        def token_stream():
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for part in stream:
                yield part["message"]["content"]

        try:
            st.write_stream(token_stream())
        except Exception as e:
            st.error(f"Fejl ved kald til Ollama: {e}")
            st.info("Kontroller at Ollama kører og at modellen 'intel-analyst' er tilgængelig.")
            st.stop()

        if chunks:
            st.divider()
            st.markdown("**Kilder:**")
            seen = set()
            for chunk in chunks:
                key = (chunk["source"], chunk["page"])
                if key not in seen:
                    seen.add(key)
                    st.markdown(
                        f"- `{chunk['source']}`, side {chunk['page']} &nbsp; "
                        f"<span style='color:gray;font-size:0.85em'>relevans: {chunk['score']}</span>",
                        unsafe_allow_html=True,
                    )
