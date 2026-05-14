import chromadb

# Persistent ChromaDB client
client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="neuralwiki"
)


# ==========================================
# Store Embeddings
# ==========================================

def store_embeddings(filename, embedded_chunks):

    for chunk in embedded_chunks:

        collection.add(
            documents=[chunk["content"]],

            embeddings=[chunk["embedding"]],

            metadatas=[{
                "source": filename,
                "chunk_id": chunk["chunk_id"]
            }],

            ids=[
                f"{filename}_{chunk['chunk_id']}"
            ]
        )


# ==========================================
# Advanced Semantic Search
# ==========================================

def semantic_search(
    query_embedding,
    top_k=5,
    similarity_threshold=1.5
):

    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=top_k
    )

    formatted_results = []

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    seen_content = set()


    for doc, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        # ==================================
        # Similarity Filtering
        # ==================================

        if distance > similarity_threshold:
            continue


        # ==================================
        # Duplicate Removal
        # ==================================

        if doc in seen_content:
            continue

        seen_content.add(doc)


        formatted_results.append({

            "content": doc,

            "source": metadata["source"],

            "chunk_id": metadata["chunk_id"],

            "similarity_score": round(
                distance,
                4
            )
        })


    return formatted_results