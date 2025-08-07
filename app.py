import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="🧠 RAG Chatbot (Local LLM)", layout="wide")
st.title("📄 RAG Chatbot (Local, No API Needed)")

# --- INPUT ---
query = st.text_input("🤖 Ask a question based on the PDF you uploaded earlier:")

# --- Load FAISS vector store ---
@st.cache_resource(show_spinner="Loading FAISS vector store...")
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_path = os.path.join("C:", os.sep, "CodeShell_Core", "GitHub_Repository", "AI Chatbot-with-RAG", "faiss_store")
    return FAISS.load_local(faiss_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

# --- Load Local LLM ---
@st.cache_resource(show_spinner="Loading Local LLM... (e.g., TinyLlama)")
def load_local_llm():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=gen_pipeline)

# Load vector store and retriever
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Prompt template
prompt_template = """You are a helpful AI assistant. Use the following context to answer the question.
If the answer cannot be found in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Load LLM
llm = load_local_llm()

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# --- Run Query ---
if query:
    with st.spinner("Generating answer..."):
        result = qa_chain.invoke({"query": query})

        st.subheader("💬 Answer")
        st.write(result["result"])

        st.subheader("📚 Source Documents")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
            st.info(doc.page_content[:500] + "...")
