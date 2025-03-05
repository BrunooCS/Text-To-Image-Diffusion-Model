from sentence_transformers import SentenceTransformer
import torch

def get_word_embeddings(words):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(words, show_progress_bar=True)
    return torch.tensor(embeddings)
