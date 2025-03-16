import csv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Save text chunks and embeddings to CSV
def save_to_csv(text_chunks, embeddings, file_name="output.csv"):
    # Compute embeddings for each chunk
    chunk_texts = [t.page_content for t in text_chunks]
    chunk_embeddings = embeddings.embed_documents(chunk_texts)

    # Write to CSV
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Chunk Text", "Embedding"])

        for text, embedding in zip(chunk_texts, chunk_embeddings):
            writer.writerow([text, ",".join(map(str, embedding))])

# Main execution
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Save the extracted text chunks and embeddings to a CSV file
save_to_csv(text_chunks, embeddings)
