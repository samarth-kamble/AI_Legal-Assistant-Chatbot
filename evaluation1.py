import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from transformers import BertTokenizer, BertModel
import torch

# Configuration
CHROMA_DB_PATH = "chroma_db_audio"  # Path to store Chroma DB

# Hardcoded PDF file path (replace with your actual PDF path)
uploaded_pdf = "./data/OnlySections.pdf"  # Full path to the PDF file

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate BERT embeddings
def get_bert_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Function to load and split the PDF into documents
def load_pdf_documents(file_path):
    """Load and split the PDF into documents."""
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

# Function to split documents into smaller chunks
def chunk_documents(raw_documents):
    """Split documents into smaller chunks."""
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1500,  
        chunk_overlap=300,  
        add_start_index=True #Adding some meta data at each index.
    )
    return text_processor.split_documents(raw_documents)

# Function to index documents into Chroma DB with BERT embeddings
def index_documents(document_chunks):
    """Index documents into Chroma DB."""
    # Generate BERT embeddings for each chunk
    embeddings = []
    for chunk in document_chunks:
        embedding = get_bert_embedding(chunk.page_content)
        if embedding:
            embeddings.append(embedding)
        else:
            print(f"Skipping chunk due to missing embedding: {chunk.page_content[:50]}...")

    # Create ChromaDB collection
    chroma_client = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=None)

    # Add documents and embeddings to ChromaDB using `upsert`
    ids = [f"doc_{i}" for i in range(len(document_chunks))]
    documents = [chunk.page_content for chunk in document_chunks]
    metadatas = [chunk.metadata for chunk in document_chunks]

    chroma_client._collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("✅ Documents indexed and embeddings stored in Chroma DB.")

if __name__ == "__main__":
    # Check if the file exists
    if not os.path.exists(uploaded_pdf):
        print(f"Error: File not found at {uploaded_pdf}")
    else:
        # Process the PDF and store embeddings
        raw_docs = load_pdf_documents(uploaded_pdf)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        print("Embeddings done and saved using Chroma DB.")