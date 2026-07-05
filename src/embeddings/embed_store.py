import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   encode_kwargs={"normalize_embeddings": True})


