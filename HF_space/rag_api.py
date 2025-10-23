import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate

# --- FastAPI App ---
app = FastAPI(title="API RAG Ale-Uy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["POST", "GET"], # Permitir GET para pre-vuelo
    allow_headers=["*"],
)

# --- Variables Globales ---
qa_chain: Optional[RetrievalQA] = None

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    """Al iniciar, conecta a Qdrant y configura la cadena de QA con Groq."""
    global qa_chain
    print("--- Iniciando la aplicación: Configuración de RAG... ---")

    # 1. Configurar Retriever de Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "carrera_profesional")

    if not QDRANT_URL or not QDRANT_API_KEY:
        print("!!!!!! FATAL: Variables de Qdrant no definidas.")
        return

    try:
        print("Inicializando embeddings con Google Gemini...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        print(f"Conectando a Qdrant en: {QDRANT_URL}")
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print("Retriever de Qdrant inicializado exitosamente.")

    except Exception as e:
        print(f"!!!!!! ERROR al inicializar el retriever de Qdrant: {e}")
        return # Detener el inicio si el retriever falla

    # 2.1. Configurar el LLM (Groq) y la cadena de QA
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b") # Modelo por defecto

    if not GROQ_API_KEY or not SYSTEM_PROMPT:
        print("!!!!!! FATAL: GROQ_API_KEY o SYSTEM_PROMPT no definidas.")
        return

    try:
        print(f"Configurando LLM de Groq: {MODEL_NAME}")
        llm = ChatGroq(model=MODEL_NAME, groq_api_key=GROQ_API_KEY)

        PROMPT = PromptTemplate(
            template=SYSTEM_PROMPT, input_variables=["context", "question"]
        )

    # # 2.2. Configurar el LLM (Gemma) y la cadena de QA
    # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
    # MODEL_NAME = os.getenv("MODEL_NAME", "gemma-3-27b-it") # Modelo por defecto

    # if not GOOGLE_API_KEY or not SYSTEM_PROMPT:
    #     print("!!!!!! FATAL: GOOGLE_API_KEY o SYSTEM_PROMPT no definidas.")
    #     return

    # try:
    #     print(f"Configurando LLM de Google: {MODEL_NAME}")
    #     llm = ChatGoogleGenerativeAI(
    #         model=MODEL_NAME,
    #         google_api_key=GOOGLE_API_KEY,
    #         temperature=0.7,
    #     )

        print("Creando la cadena de QA...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("--- ¡Aplicación lista para recibir preguntas! ---")

    except Exception as e:
        print(f"!!!!!! ERROR al configurar la cadena de QA: {e}")
        qa_chain = None

@app.post("/ask")
async def ask_rag(request: QueryRequest):
    if not qa_chain:
        raise HTTPException(status_code=503, detail="La cadena de QA no está inicializada. Revisa los logs del servidor.")
    
    print(f"Recibida pregunta: {request.query}")
    try:
        response = qa_chain.invoke({"query": request.query})
        return {
            "answer": response["result"],
            "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in response["source_documents"]],
        }
    except Exception as e:
        print(f"!!!!!! ERROR durante la invocación de la cadena: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la pregunta: {e}")

@app.get("/")
def health_check():
    """Endpoint de salud para verificar que la API está viva."""
    return {"status": "ok", "qa_chain_ready": qa_chain is not None}