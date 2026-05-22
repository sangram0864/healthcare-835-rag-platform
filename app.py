import streamlit as st
import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Healthcare 835 RAG Platform",
    layout="wide"
)

st.title("🏥 Healthcare 835 RAG Platform")


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

model = load_embedding_model()


# =====================================================
# LOAD OPEN SOURCE LLM
# =====================================================

@st.cache_resource
def load_llm():

    return pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

llm = load_llm()


# =====================================================
# CREATE CHROMADB CLIENT
# =====================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="healthcare_claims"
)


# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload EDI 835 File",
    type=["txt"]
)


# =====================================================
# PROCESS FILE
# =====================================================

if uploaded_file:

    content = uploaded_file.read().decode("utf-8")

    lines = content.splitlines()

    documents = []

    structured_data = []

    validation_errors = []


    # =====================================================
    # VALIDATE + PARSE CLAIMS
    # =====================================================

    st.subheader("✅ Validation Results")

    for line_no, line in enumerate(lines):

        parts = line.strip().split("|")


        # -------------------------------------------------
        # VALIDATE SEGMENT
        # -------------------------------------------------

        if parts[0] != "CLM":

            validation_errors.append(
                f"Line {line_no+1}: Invalid segment"
            )

            continue


        # -------------------------------------------------
        # VALIDATE FIELD COUNT
        # -------------------------------------------------

        if len(parts) < 6:

            validation_errors.append(
                f"Line {line_no+1}: Missing fields"
            )

            continue


        # -------------------------------------------------
        # EXTRACT CLAIM DATA
        # -------------------------------------------------

        claim_id = parts[1]
        amount = parts[2]
        patient = parts[3]
        provider = parts[4]
        status = parts[5]


        # -------------------------------------------------
        # STRUCTURED DOCUMENT
        # -------------------------------------------------

        structured_text = f"""
Claim ID: {claim_id}
Payment Amount: {amount}
Patient Name: {patient}
Provider: {provider}
Status: {status}
"""

        documents.append(structured_text)


        # -------------------------------------------------
        # DASHBOARD DATA
        # -------------------------------------------------

        structured_data.append({

            "Claim ID": claim_id,
            "Amount": amount,
            "Patient": patient,
            "Provider": provider,
            "Status": status
        })


    # =====================================================
    # SHOW VALIDATION ERRORS
    # =====================================================

    if validation_errors:

        for error in validation_errors:

            st.error(error)

        st.error(
            "835 validation failed. Claims will NOT be stored in Vector DB."
        )

        st.stop()

    else:

        st.success("835 File Validation Passed")


    # =====================================================
    # SHOW DASHBOARD
    # =====================================================

    st.subheader("📋 Parsed Claim Dashboard")

    df = pd.DataFrame(structured_data)

    st.dataframe(
        df,
        use_container_width=True
    )


    # =====================================================
    # GENERATE EMBEDDINGS
    # =====================================================

    st.subheader("🧠 Generating Embeddings")

    embeddings = model.encode(documents)

    st.success("Embeddings created successfully")


    # =====================================================
    # STORE IN CHROMADB
    # =====================================================

    st.subheader("💾 Storing Claims in ChromaDB")

    existing_count = collection.count()

    for i, doc in enumerate(documents):

        unique_id = f"{uploaded_file.name}_{existing_count+i}"

        collection.add(

            ids=[unique_id],

            documents=[doc],

            embeddings=[
                embeddings[i].tolist()
            ]
        )

    st.success("✅ Claims stored successfully in ChromaDB")


    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

    st.subheader("🔍 Semantic Claim Search")

    query = st.text_input(
        "Ask questions about claims"
    )


    # =====================================================
    # PROCESS QUERY
    # =====================================================

    if query:


        # -------------------------------------------------
        # CREATE QUERY EMBEDDING
        # -------------------------------------------------

        query_embedding = model.encode([query])


        # -------------------------------------------------
        # SEARCH VECTOR DB
        # -------------------------------------------------

        results = collection.query(

            query_embeddings=[
                query_embedding[0].tolist()
            ],

            n_results=2
        )


        retrieved_docs = results["documents"][0]


        # -------------------------------------------------
        # CREATE CONTEXT
        # -------------------------------------------------

        context = "\n".join(retrieved_docs)


        # -------------------------------------------------
        # CREATE RAG PROMPT
        # -------------------------------------------------

        rag_prompt = f"""
You are Healthcare Claims AI Assistant.

Use ONLY retrieved claims data.

Retrieved Claims:
{context}

User Question:
{query}

Rules:
- Provide short clean answer
- Show only relevant claims
- Do not repeat unnecessary data
- If user asks pending claims, show pending only
- If user asks denied claims, show denied only
- If user asks paid claims, show paid only
- If user asks claim count, provide count

Final Answer:
"""


        # -------------------------------------------------
        # SEND TO OPEN SOURCE LLM
        # -------------------------------------------------

        llm_response = llm(

            rag_prompt,

            max_new_tokens=120
        )


        final_answer = llm_response[0][
            "generated_text"
        ]


        # -------------------------------------------------
        # CLEAN RESPONSE
        # -------------------------------------------------

        final_answer = final_answer.replace(
            rag_prompt,
            ""
        )


        # =====================================================
        # SHOW FINAL AI RESPONSE
        # =====================================================

        st.subheader("🤖 AI Response")

        st.success(final_answer)


        # =====================================================
        # OPTIONAL DEBUG VIEW
        # =====================================================

        with st.expander("📄 View Retrieved Claims"):

            for doc in retrieved_docs:

                st.info(doc)