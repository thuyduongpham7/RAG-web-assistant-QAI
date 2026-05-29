# -*- coding: utf-8 -*-
"""
Created on Fri May  8 23:11:25 2026

@author: Duong
"""

import streamlit as st
import os

from rag_pipeline import create_rag

# ----------------------------
# API KEY
# ----------------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ----------------------------
# SESSION LIMIT CONFIG
# ----------------------------
MAX_QUESTIONS = 3

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Démo RAG",
    page_icon="📚"
)

# ----------------------------
# UI
# ----------------------------
st.title("📚 Démo RAG - Assistant IA basé sur une page web")

st.markdown("""
Bienvenue dans cette démonstration RAG (Retrieval-Augmented Generation).

Cette application utilise :
- LangChain
- OpenAI
- ChromaDB
- Streamlit

⚠️ Cette démo publique est limitée à 3 questions par session afin de limiter les coûts API.
""")

# ----------------------------
# SHOW REMAINING QUESTIONS
# ----------------------------
remaining = MAX_QUESTIONS - st.session_state.question_count

st.info(f"🟢 Questions restantes dans cette session : {remaining}")

# ----------------------------
# URL INPUT
# ----------------------------
url = st.text_input(
    "Entrez l'URL d'une page web",
    "https://fr.wikipedia.org/wiki/Qualit%C3%A9_de_l%27air_int%C3%A9rieur"
)

# ----------------------------
# BUILD RAG (CACHE)
# ----------------------------
@st.cache_resource
def load_rag(url):
    return create_rag(url)

# ----------------------------
# LOAD RAG
# ----------------------------
if url:

    with st.spinner("Chargement et indexation de la page web..."):

        rag_chain = load_rag(url)

    st.success("✅ La page web a été indexée avec succès !")

    # ----------------------------
    # QUESTION INPUT
    # ----------------------------
    question = st.text_input("Posez votre question (ex: quel est le sujet principal de cette page web ?)")

    # ----------------------------
    # SESSION LIMIT CHECK
    # ----------------------------
    if st.session_state.question_count >= MAX_QUESTIONS:

        st.warning("""
🚫 Limite de session atteinte.

Cette démo publique est limitée à 3 questions par session afin de contrôler les coûts d'utilisation de l'API OpenAI.

Veuillez actualiser la page pour démarrer une nouvelle session.
""")

        st.stop()

    # ----------------------------
    # PROCESS QUESTION
    # ----------------------------
    if question:

        with st.spinner("Génération de la réponse..."):

            response = rag_chain.invoke(question)

            st.session_state.question_count += 1

            st.markdown("### 🤖 Réponse")
            st.write(response.content)

            remaining = MAX_QUESTIONS - st.session_state.question_count

            if remaining > 0:
                st.info(f"🟢 Questions restantes : {remaining}")
            else:
                st.warning("🚫 Vous avez atteint la limite de 3 questions pour cette session.")
