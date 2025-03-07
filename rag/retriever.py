import os
import faiss
import json
import numpy as np
from langchain_openai import ChatOpenAI
from data_processor import embeddings_model, faiss_index_path, json_data_path


# Initialize LLM (GPT-4o)
llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Load FAISS index and text data from JSON
def load_faiss_and_json():    
    index = faiss.read_index(faiss_index_path)    
    
    with open(json_data_path, "r", encoding="utf-8") as f:
        stored_texts = json.load(f)
    
    return index, stored_texts



# Load questions
def load_questions_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f.readlines() if line.strip()]
    return questions



# Search FAISS index for the most relevant documents
def search_faiss(query, k=5):    
    
    index, stored_texts = load_faiss_and_json()
    
    # Convert query into embedding
    query_embedding = embeddings_model.embed_query(query)
    query_vector = np.array(query_embedding).reshape(1, -1).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_vector, k)

    # Retrieve original text based on FAISS indices
    retrieved_texts = [stored_texts[i] for i in indices[0]]

    return retrieved_texts, distances[0]



# Retrieve information using RAG
def search_with_rag(query, k=5):
    retrieved_texts, distances = search_faiss(query, k)
    
    # If FAISS don't find relevant information, send the question to GPT
    if not retrieved_texts:
        return f"No relevant documents found. GPT will generate an answer based on its knowledge."

    # If FAISS return the results, combine them to the content
    context = "\n".join(retrieved_texts[:3]) # maximum 3 documents
    return f"Content:\n{context}\n\nQuestion: {query}"



# Generate an answer using GPT based on retrieved context
def generate_answer(query):
    print(f"Generating predicted query for the question: {query}")
    context_with_query = search_with_rag(query)
    prompt = (
        "You are an SQL expert. "
        "Based on the retrieved database schema, generate a valid SQL query "
        "that correctly answers the user's question. "
        "Strictly use only the provided schema and do not assume missing columns or tables. "
        "Return only the SQL query without explanation.\n"
        f"{context_with_query}"
    )
    response = llm.invoke(prompt)
    
    return response.content


# Store the results
output_file = "pred_example.txt"

questions = load_questions_from_txt("questions.txt")

with open(output_file, "w", encoding="utf-8") as f:
    for i, question in enumerate(questions, 1):
        answer = generate_answer(question)        
       
        f.write(f"{question}\n")
        f.write(f"Answer: {answer}\n")
        f.write("-" * 50 + "\n")

print(f"Answers stored in '{output_file}'.")
