from langchain_community.embeddings import HuggingFaceEmbeddings


def load_embeddings():
    """
    Load embedding model for semantic search.
    This directly affects retrieval quality.
    """

    print("Loading embedding model (MPNet)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},   # safe for MacBook
        encode_kwargs={"normalize_embeddings": True}  # improves similarity search
    )

    print("Embeddings loaded successfully")

    return embeddings