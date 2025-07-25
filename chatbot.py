import os
from glob import glob
import subprocess
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Load all .txt files from the 'docs/' folder
docs_path = "docs/"
all_files = glob(os.path.join(docs_path, "*.txt"))

documents = []
for file_path in all_files:
    loader = TextLoader(file_path)
    documents.extend(loader.load())

print(f"📄 Total files loaded: {len(all_files)}")
print(f"📚 Total documents: {len(documents)}")

# 2. Split all documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"🧩 Total chunks: {len(texts)}")

if not texts:
    print("⚠️ No text found. Please check the 'docs/' folder.")
    exit()

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store
vectorstore = FAISS.from_documents(texts, embedding_model)

# Local model
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE_NAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}. Starting download...")
    os.makedirs("models", exist_ok=True)
    cmd = [
        "huggingface-cli", "download",
        MODEL_REPO,
        MODEL_FILE_NAME,
        "--local-dir", "models",
        "--local-dir-use-symlinks", "False"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("Download completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading the model: {e}")
        exit(1)
else:
    print("✅ Model found locally.")

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.3,
    max_tokens=200,
    top_p=0.95,
    n_ctx=1024,
    verbose=False
)

# RAG pipeline
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = """
Use the information below to answer the question briefly and clearly. If you don't know, respond with 'I don't know'.

Information:
{context}

Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# Interaction loop
while True:
    question = input("\nQuestion: ")
    if question.lower() in ["exit", "quit", "sair"]:
        break

    answer = qa_chain.invoke({"query": question})
    print(f"\nAnswer: {answer['result']}\n")
