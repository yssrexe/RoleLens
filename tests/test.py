from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = model.predict(
    [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "Berlin is the capital of France."),
        ("What is the capital of France?", "marseille is the capital of France."),
    
    ]
)
print(scores)