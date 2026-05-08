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


# ----------------------------
# 1. LOAD DOCUMENTS
# ----------------------------
def load_documents(url: str):

    loader = WebBaseLoader(url)

    docs = loader.load()

    return docs


# ----------------------------
# 2. SPLIT DOCUMENTS
# ----------------------------
def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    return chunks


# ----------------------------
# 3. BUILD VECTOR DATABASE
# ----------------------------
def build_vectorstore(chunks):

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# ----------------------------
# 4. BUILD RAG CHAIN
# ----------------------------
def build_rag_chain(vectorstore):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    retriever = vectorstore.as_retriever()

    # ----------------------------
    # FORMAT RETRIEVED DOCS
    # ----------------------------
    def format_docs(docs):

        return "\n\n".join(
            [doc.page_content for doc in docs]
        )

    # ----------------------------
    # PROMPT
    # ----------------------------
    prompt = ChatPromptTemplate.from_template("""
Tu es un assistant intelligent spécialisé dans l'analyse de pages web.

Réponds uniquement avec les informations présentes dans le contexte fourni.

Réponds dans la même langue que la question de l'utilisateur.

Si la réponse n'est pas présente dans le contexte, réponds :
"Je ne trouve pas la réponse dans la page web fournie."

Contexte :
{context}

Question :
{question}
""")

    # ----------------------------
    # LCEL RAG CHAIN
    # ----------------------------
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain


# ----------------------------
# 5. FULL RAG PIPELINE
# ----------------------------
def create_rag(url: str):

    docs = load_documents(url)

    chunks = split_documents(docs)

    vectorstore = build_vectorstore(chunks)

    rag_chain = build_rag_chain(vectorstore)

    return rag_chain