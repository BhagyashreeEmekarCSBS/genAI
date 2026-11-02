
import streamlit as st
import fitz
import requests
import numpy as np

OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(
    page_title="TroubleMate AI",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 TroubleMate AI")
st.subheader("AI-powered PDF Troubleshooting Assistant")

st.write(
    "Upload a technical manual, describe your problem, "
    "and receive step-by-step troubleshooting guidance."
)

def get_embeddings(texts):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=180
    )
    response.raise_for_status()
    return np.array(response.json()["embeddings"])

def extract_pdf(pdf_file):
    document = fitz.open(
        stream=pdf_file,
        filetype="pdf"
    )

    chunks = []

    for page_number, page in enumerate(document, start=1):
        words = page.get_text().split()

        for i in range(0, len(words), 180):
            content = " ".join(words[i:i+180])

            if content.strip():
                chunks.append({
                    "text": content,
                    "page": page_number
                })

    return chunks

def find_relevant_chunks(question, chunks, vectors):
    query_vector = get_embeddings([question])[0]

    scores = vectors @ query_vector / (
        np.linalg.norm(vectors, axis=1)
        * np.linalg.norm(query_vector) + 1e-10
    )

    best_indices = np.argsort(scores)[-4:][::-1]

    return [
        chunks[i] for i in best_indices
        if scores[i] > 0
    ]

def generate_answer(question, context):
    prompt = f"""
You are TroubleMate AI, a technical troubleshooting assistant.

Answer using the supplied PDF information only.
Do not invent technical instructions.

If the PDF does not contain enough information,
clearly explain that.

User Problem:
{question}

Relevant PDF Information:
{context}

Provide your response in this format:

1. Problem Explanation
2. Possible Causes
3. Step-by-Step Troubleshooting
4. Expected Result
5. Safety Precautions
6. Relevant PDF Page Numbers
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["message"]["content"]


if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.vectors = None

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader(
    "Upload Troubleshooting PDF",
    type=["pdf"]
)

if uploaded_file:
    if st.button("Process PDF", type="primary"):
        try:
            with st.spinner("Processing PDF..."):
                chunks = extract_pdf(uploaded_file.getvalue())

                if not chunks:
                    st.error("No readable PDF text found.")
                    st.stop()

                vectors = get_embeddings(
                    [chunk["text"] for chunk in chunks]
                )

                st.session_state.chunks = chunks
                st.session_state.vectors = vectors
                st.session_state.messages = []

            st.success(
                f"PDF processed successfully! "
                f"{len(chunks)} sections indexed."
            )

        except Exception as error:
            st.error(f"PDF processing error: {error}")

if st.session_state.chunks is not None:

    st.success("Your PDF knowledge base is ready!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Describe your technical problem..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing your problem..."):

                    relevant = find_relevant_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.vectors
                    )

                    context = "\n\n".join(
                        f"Page {chunk['page']}: "
                        f"{chunk['text']}"
                        for chunk in relevant
                    )

                    answer = generate_answer(
                        question,
                        context
                    )

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

            except Exception as error:
                st.error(f"AI error: {error}")

st.divider()
st.caption(
    "Powered by Ollama | Llama 3.2 | Nomic Embeddings"
)
