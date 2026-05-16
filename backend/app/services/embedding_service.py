from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embedding(text):
    embedding = model.encode(text)
    return embedding.tolist()


def generate_embeddings(chunks):
    embedded_chunks = []

    for chunk in chunks:
        vector = generate_embedding(chunk["content"])

        embedded_chunks.append({
            "chunk_id":  chunk["chunk_id"],
            "content":   chunk["content"],
            "length":    chunk["length"],
            "embedding": vector,
            # ← carry forward entity tags set by link_entities_to_chunks()
            # previously this dict was built from scratch and "entities" was silently dropped
            "entities":  chunk.get("entities", []),
        })

    return embedded_chunks