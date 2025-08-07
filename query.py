import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Load FAISS vector store
print("🔍 Loading FAISS vector store...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_path = os.path.join("C:", os.sep, "CodeShell_Core", "GitHub_Repository", "AI Chatbot-with-RAG", "faiss_store")
vectorstore = FAISS.load_local(faiss_path, embeddings=embedding_model, allow_dangerous_deserialization=True)

# 2. Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 3. Define prompt template
prompt_template = """You are a helpful AI assistant. Use the following context to answer the question.
If the answer cannot be found in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 4. Load Local LLM (e.g., Mistral, TinyLLama, or any causal model)
print("🧠 Loading local LLM...")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  # You can replace with a lighter model like TinyLlama if needed
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

# Build the pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 5. Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 6. Ask a question
query = input("🤖 Ask a question: ")
response = qa_chain.invoke({"query": query})

# 7. Show answer
print("\n💬 Answer:")
print(response["result"])

# 8. Show sources
print("\n📚 Source Documents:")
for i, doc in enumerate(response["source_documents"]):
    print(f"\nSource {i+1}: {doc.metadata['source']}")
    print(doc.page_content[:300], "...")
