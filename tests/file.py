import numpy as np
import psycopg2

from langchain_huggingface import HuggingFaceEmbeddings


texts = [
    "The cat is sleeping on the sofa.",
    "My  dog loves playingin the park.",
    "The fish swims quietly in the aquarium.",
    "A small bird landed on the tree branch.",
    "The lizard is warming itself on a rock.",
    "My dad bought a new car last week.",
    "A large truck delivered the furniture this morning.",
    "She rides her bicycle to work every day.",
    "He enjoys riding his motorcycle on weekends.",
    "The airplane landed safely at the airport.",
    "The children walked to school together.",
    "The patient was taken to the hospital for treatment.",
    "She is studying computer science at the university.",
    "I borrowed an interesting book from the library.",
]



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_list = []
for text in texts:
    embedding_list.append(embeddings.embed_query(text))

conn = psycopg2.connect(
    host="localhost",
    database="testembedding",
    user="postgres",
    password="postgres"
)
curr = conn.cursor()

# for i in range(len(texts)):
#     embedding_value = embedding_list[i]
#     content = texts[i]
#     curr.execute("INSERT INTO items (content, embedding) VALUES (%s, %s)",
#                  (content, embedding_value))

new_text = ["The cat is playing with a ball of yarn."]

new_embedding = embeddings.embed_query(new_text[0])
curr.execute("SELECT id, content FROM items ORDER BY embedding <-> %s::vector LIMIT 5", (new_embedding,))

result = curr.fetchall()
for row in result:
    print(f"ID: {row[0]}, Content: {row[1]}")


conn.commit()
curr.close()
conn.close()

