# -*- coding: utf-8 -*-
"""
Created on Fri May  8 23:16:03 2026

@author: Duong
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# 1. LOAD DOCUMENTS
# ============================================================
def load_documents(url: str):
    """
    Charge le contenu de la page web.

    WebBaseLoader :
    - récupère le contenu HTML de la page,
    - extrait le texte principal,
    - retourne une liste de Documents LangChain.
    """

    loader = WebBaseLoader(url)

    docs = loader.load()

    return docs


# ============================================================
# 2. SPLIT DOCUMENTS
# ============================================================
def split_documents(docs):
    """
    Découpe les documents en plusieurs chunks.

    Pourquoi découper ?
    -------------------
    Les modèles LLM ne travaillent pas efficacement sur de très
    longs textes.

    Le découpage permet :
    - une meilleure recherche sémantique,
    - une récupération plus précise,
    - une réduction des hallucinations.

    Paramètres choisis :
    --------------------
    chunk_size=1200 :
        Taille plus grande que la version précédente.
        Cela évite de couper les phrases importantes.

    chunk_overlap=200 :
        Chevauchement entre les chunks afin de conserver
        du contexte entre les morceaux.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    return chunks


# ============================================================
# 3. BUILD VECTOR DATABASE
# ============================================================
def build_vectorstore(chunks):
    """
    Crée la base vectorielle ChromaDB.

    Étapes :
    --------
    1. Création des embeddings OpenAI
    2. Conversion des chunks en vecteurs
    3. Stockage dans ChromaDB

    Les embeddings permettent :
    - la recherche sémantique,
    - la récupération des passages pertinents.
    """

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# ============================================================
# 4. BUILD RAG CHAIN
# ============================================================
def build_rag_chain(vectorstore):
    """
    Construit la chaîne RAG complète.

    Cette chaîne :
    1. récupère les chunks pertinents,
    2. construit le prompt,
    3. envoie le contexte au LLM,
    4. génère une réponse basée uniquement sur ce contexte.
    """

    # --------------------------------------------------------
    # LLM CONFIGURATION
    # --------------------------------------------------------
    llm = ChatOpenAI(

        # GPT-4.1-mini :
        # - meilleur suivi d'instructions,
        # - moins d'hallucinations,
        # - très bon compromis coût/performance.
        model="gpt-4.1-mini",

        # Température à 0 :
        # rend les réponses plus déterministes
        # et réduit la créativité/hallucination.
        temperature=0,

        # Limite la longueur maximale des réponses :
        # - réduit les coûts,
        # - évite les réponses trop longues,
        # - diminue les risques d'hallucination.
        max_tokens=300
    )

    # --------------------------------------------------------
    # RETRIEVER CONFIGURATION
    # --------------------------------------------------------
    retriever = vectorstore.as_retriever(

        # Utilisation d'un seuil de similarité.
        #
        # Cela permet d'éviter de récupérer des chunks
        # peu pertinents.
        #
        # Très important pour limiter les hallucinations.
        search_type="similarity_score_threshold",

        search_kwargs={

            # Nombre maximum de chunks récupérés.
            #
            # Un nombre trop élevé peut :
            # - injecter du bruit,
            # - perturber le LLM.
            #
            # 3 est souvent un bon compromis.
            "k": 3,

            # Seuil minimum de similarité.
            #
            # Si les chunks ne sont pas suffisamment proches
            # de la question, ils ne seront pas retournés.
            #
            # Cela réduit fortement les réponses inventées.
            "score_threshold": 0.5
        }
    )

    # --------------------------------------------------------
    # FORMAT RETRIEVED DOCUMENTS
    # --------------------------------------------------------
    def format_docs(docs):
        """
        Transforme les documents récupérés en texte brut.

        Cas important :
        ----------------
        Si aucun document pertinent n'est trouvé,
        on retourne un mot-clé spécial.

        Cela permet au prompt de détecter explicitement
        l'absence d'information.
        """

        # Aucun document pertinent trouvé
        if not docs:
            return "NO_INFORMATION"

        # Concaténation des chunks récupérés
        return "\n\n".join(
            [doc.page_content for doc in docs]
        )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------
    prompt = ChatPromptTemplate.from_template("""
You are a web page analysis assistant.

You must answer ONLY using the information contained in the provided context.

Do not invent information.
Do not use external knowledge.
Do not make assumptions.
Do not infer missing information.

If the answer is not explicitly present in the context,
reply ONLY with:
"I cannot find the answer in the provided web page."

If the context contains "NO_INFORMATION",
reply ONLY with:
"I cannot find the answer in the provided web page."

Answer in the same language as the user's question.

Keep answers concise, factual, and precise.

Context:
{context}

Question:
{question}
""")

    # --------------------------------------------------------
    # LCEL RAG CHAIN
    # --------------------------------------------------------
    chain = (
        {

            # Étape 1 :
            # récupération des chunks pertinents
            # puis formatage du contexte.
            "context": retriever | format_docs,

            # Étape 2 :
            # transmission directe de la question utilisateur.
            "question": RunnablePassthrough()
        }

        # Étape 3 :
        # construction du prompt final.
        | prompt

        # Étape 4 :
        # génération de la réponse par le LLM.
        | llm
    )

    return chain


# ============================================================
# 5. FULL RAG PIPELINE
# ============================================================
def create_rag(url: str):
    """
    Pipeline RAG complet.

    Étapes :
    --------
    1. Chargement de la page web
    2. Découpage en chunks
    3. Création des embeddings
    4. Construction de la base vectorielle
    5. Construction de la chaîne RAG
    """

    # Chargement des documents web
    docs = load_documents(url)

    # Découpage des documents
    chunks = split_documents(docs)

    # Création de la base vectorielle
    vectorstore = build_vectorstore(chunks)

    # Construction de la chaîne RAG
    rag_chain = build_rag_chain(vectorstore)

    return rag_chain