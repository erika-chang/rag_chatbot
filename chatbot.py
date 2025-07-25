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

# 1. Carregar todos os arquivos .txt da pasta docs/
docs_path = "docs/"
all_files = glob(os.path.join(docs_path, "*.txt"))

documents = []
for file_path in all_files:
    loader = TextLoader(file_path)
    documents.extend(loader.load())

print(f"📄 Total de arquivos carregados: {len(all_files)}")
print(f"📚 Total de documentos: {len(documents)}")

# 2. Dividir todos os documentos em chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"🧩 Total de chunks: {len(texts)}")

if not texts:
    print("⚠️ Nenhum texto encontrado. Verifique a pasta 'docs/'")
    exit()

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Banco vetorial
vectorstore = FAISS.from_documents(texts, embedding_model)

# Modelo local
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE_NAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Modelo não encontrado em {MODEL_PATH}. Iniciando download...")
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
        print("Download concluído com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"Erro no download do modelo: {e}")
        exit(1)
else:
    print("✅ Modelo encontrado localmente.")

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.7,
    max_tokens=200,
    top_p=0.95,
    n_ctx=1024,
    verbose=False
)

# RAG
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """
Use as informações abaixo para responder a pergunta de forma breve e clara. Se não souber, responda 'Não sei'.

Informações:
{context}

Pergunta: {question}
Resposta:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# Loop
while True:
    question = input("\nPergunta: ")
    if question.lower() in ["sair", "exit", "quit"]:
        break

    answer = qa_chain.invoke({"query": question})
    print(f"\nResposta: {answer['result']}\n")
