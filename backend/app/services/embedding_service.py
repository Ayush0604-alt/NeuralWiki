from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embedding(text):

    embedding = model.encode(text)

    return embedding.tolist()


def generate_embeddings(chunks):

    embedded_chunks = []

    for chunk in chunks:

        vector = generate_embedding(chunk["content"])

        embedded_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "length": chunk["length"],
            "embedding": vector
        })

    return embedded_chunks