import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


@st.cache_resource
def load_index_and_data():
    index = faiss.read_index("law_index.faiss")
    with open("law_mapping.pkl", "rb") as f:
        df = pickle.load(f)
    return index, df

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_gemini():
    genai.configure(api_key="")  # use the gemini api key
    return genai.GenerativeModel("gemini-2.5-pro")

index, df = load_index_and_data()
embedding_model = load_embedding_model()
gemini_model = load_gemini()


st.title("⚖️ LawSense India")
st.write("Ask any question about Indian laws. The bot retrieves sections and gives answers.")

user_query = st.text_input("Enter your legal question:")

if user_query:
    query_embedding = embedding_model.encode([user_query])
    query_embedding = np.array(query_embedding).astype("float32")

    k = 5
    distances, indices = index.search(query_embedding, k)

    results = [df.iloc[i]["content"] for i in indices[0]]
    context = "\n\n".join(results)

    prompt = f"""
    User asked: "{user_query}"

    Relevant Indian law sections:
    {context}

    Answer clearly and cite the section(s).
    """

    with st.spinner("Thinking..."):
        response = gemini_model.generate_content(prompt)

    st.subheader("📌 Answer")
    st.write(response.text)

    with st.expander("🔎 Retrieved law sections"):
        for r in results:
            st.write(r)
