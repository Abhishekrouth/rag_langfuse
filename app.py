from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from langfuse import Langfuse

load_dotenv()

app = Flask(__name__)

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)
gemini_api_key = os.getenv("GEMINI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=gemini_api_key
)

embedder = GoogleGenerativeAIEmbeddings(model="text-embedding-004",task_type="RETRIEVAL_DOCUMENT")

@app.route('/', methods=['GET'])
def home():
    return "WELCOME to LEGAL RAG SERVICE"

@app.route("/ingest", methods=["POST"])
def ingest():

    loader = PyPDFLoader('Data/agreement.pdf')
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40
    )
    chunks = splitter.split_documents(documents)

    batch_size = 64
    vectors = []

    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.embed_documents(texts)

    for idx, (doc, vec) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{doc.metadata.get('doc_id', 'doc')}{idx}"

        vectors.append((vector_id, vec, {"text": doc.page_content}))

        if len(vectors) >= batch_size:
            index.upsert(vectors)
            vectors = []

    if vectors:
        index.upsert(vectors)
    return jsonify({
        "message": "Document indexed",
        "chunks": len(chunks)
    })

@app.route("/query", methods=["POST"])
def query():
    
    data = request.json
    query = data.get("query")

    with langfuse.start_as_current_observation(as_type="span", name="process-request") as span:
        span.update(output="Query Embedding")

    query_vec = embedder.embed_query(query)

    span.update(output="Query Embedding")
    result = index.query(
        vector=query_vec,
        top_k=3,
        include_metadata=True
    )

    context = "\n\n".join(
        match["metadata"]["text"] for match in result["matches"]
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a Legal assistant of company named SoftwareTree. 
    Answer only using the context below.
    If the answer is not present, say "Not found in documents".

    Context:
    {context}

    Question:
    {query}
""")
    
    chain = prompt | llm | StrOutputParser()

    with langfuse.start_as_current_observation(as_type="generation", name="llm-response", model="gemini-2.5-flash") as generation:
        output = chain.invoke({"context": context, "query": query})
    
        generation.update(output=output)

    langfuse.flush()

    return jsonify({
        "answer": output,
        "context": context
    })

if __name__ == "__main__":
    app.run(debug=True)