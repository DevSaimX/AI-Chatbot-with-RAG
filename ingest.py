import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. Load all .txt files manually
data_folder = os.path.join("C:", os.sep, "CodeShell_Core", "GitHub_Repository", "AI Chatbot-with-RAG", "custom_data")
documents = []

print(f"📂 Loading .txt files from: {data_folder}\n")

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    doc = Document(page_content=content, metadata={"source": filename})
                    documents.append(doc)
                    print(f"✅ Loaded: {filename} ({len(content)} characters)")
                else:
                    print(f"⚠️ Skipped empty file: {filename}")
        except Exception as e:
            print(f"❌ Error loading {filename}: {str(e)}")

print(f"\n📄 Total documents loaded: {len(documents)}")

# 🔍 Show preview of each document
for i, doc in enumerate(documents):
    preview = doc.page_content.strip().replace("\n", " ")[:100]
    print(f"🔍 Doc {i+1} preview: {preview}...")

# 2. Chunk text into small pieces (force splitting)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # lowered to support shorter docs
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"\n✂️ Total chunks created: {len(chunks)}")

# Exit if still empty
if not chunks:
    raise ValueError("❌ No chunks were created. Check your text content again.")

# 3. Embed and save to FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_path = os.path.join("C:", os.sep, "CodeShell_Core", "GitHub_Repository", "AI Chatbot-with-RAG", "faiss_store")
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local(faiss_path)

print(f"\n✅ Vector store created and saved to: {faiss_path}")
