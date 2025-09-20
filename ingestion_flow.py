import os
from prefect import flow, task
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient, models
from typing import List, Dict, Any

# --- Configuración General ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "perfil_personal")

# --- Configuración de Rutas de Datos ---
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CV_DIR = os.path.join(DATA_DIR, "CV")
PROJECTS_DIR = os.path.join(DATA_DIR, "proyectos")
REPOS_DIR = os.path.join(DATA_DIR, "repos")

# Mapeo de extensiones a cargadores
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": UnstructuredWordDocumentLoader,
    # Estos formatos no los eh usado y no se si funcionan bien
    ".py": TextLoader, ".js": TextLoader, ".ts": TextLoader,
    ".html": TextLoader, ".css": TextLoader, ".txt": TextLoader, ".ipynb": TextLoader,
}

@task
def load_documents_from_directory(directory: str) -> List[Dict[str, Any]]:
    all_docs = []
    if not os.path.isdir(directory):
        print(f"ADVERTENCIA: El directorio '{directory}' no existe. Saltando...")
        return []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in LOADER_MAPPING:
                loader_class = LOADER_MAPPING[file_ext]
                try:
                    loader = loader_class(file_path)
                    all_docs.extend(loader.load())
                except Exception as e:
                    print(f"    Error al cargar el archivo {file_path}: {e}")
    print(f"Se cargaron {len(all_docs)} documentos de '{directory}'.")
    return all_docs

@task
def split_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not documents: return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Total de chunks creados: {len(chunks)}.")
    return chunks

@task
def generate_and_store_embeddings(chunks: List[Dict[str, Any]]):
    if not chunks: return
    
    valid_chunks = [c for c in chunks if c.page_content and len(c.page_content.strip()) > 0]
    if not valid_chunks: return
    print(f"Chunks válidos: {len(valid_chunks)}.")

    try:
        print("Inicializando embeddings con Google Gemini...")
        # Asegúrate de que GOOGLE_API_KEY está en tu .env
        # Este modelo de embedding es muy bueno y rápido
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        texts = [chunk.page_content for chunk in valid_chunks]
        metadatas = [chunk.metadata for chunk in valid_chunks]
        
        print(f"Enviando {len(texts)} chunks de texto a la API de Gemini...")
        vectors = embeddings.embed_documents(texts)
        print(f"Se generaron {len(vectors)} vectores de embedding exitosamente.")

    except Exception as e:
        print(f"!!!!!! ERROR al generar embeddings con Gemini: {e}")
        raise

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_size = len(vectors[0])
    
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"La colección '{COLLECTION_NAME}' ya existe. Se agregarán/actualizarán los datos.")
    except Exception:
        print(f"La colección '{COLLECTION_NAME}' no existe. Creándola con vectores de tamaño {vector_size}...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    print("Subiendo puntos a Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=models.Batch(
            ids=list(range(len(valid_chunks))),
            vectors=vectors,
            payloads=[{"page_content": text, "metadata": metadata} for text, metadata in zip(texts, metadatas)]
        ),
        wait=True,
    )
    
    print("¡Ingesta de datos completada exitosamente!")

@flow(name="Flujo de Ingesta de Datos RAG")
def data_ingestion_flow(cv_dir: str, projects_dir: str, repos_dir: str):
    print("Iniciando carga de documentos...")
    all_documents = (
        load_documents_from_directory(cv_dir) + 
        load_documents_from_directory(projects_dir) + 
        load_documents_from_directory(repos_dir)
    )
    print(f"Total de documentos cargados: {len(all_documents)}.")

    if all_documents:
        chunks = split_documents(all_documents)
        generate_and_store_embeddings(chunks)
    else:
        print("No se encontraron documentos para procesar.")

if __name__ == "__main__":
    print("--- Iniciando el flujo de ingesta de datos ---")
    data_ingestion_flow(
        cv_dir=CV_DIR,
        projects_dir=PROJECTS_DIR,
        repos_dir=REPOS_DIR
    )
    print("--- Flujo de ingesta de datos finalizado ---")
