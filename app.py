from flask import Flask, render_template, jsonify, request
import csv
import numpy as np
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline
from dotenv import load_dotenv
import os
import pandas as pd

app = Flask(__name__)

load_dotenv()

# Load CSV Data
def load_csv_data(csv_file):
    data = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row["Chunk Text"]
            embedding = list(map(float, row["Embedding"].split(',')))
            data.append({"text": text, "embedding": embedding})
    return data

# Simple retriever to find the best matching text chunk
def retrieve_from_csv(data, query_embedding, top_k=2):
    scores = []
    for item in data:
        # Calculate cosine similarity
        score = np.dot(query_embedding, item['embedding']) / (np.linalg.norm(query_embedding) * np.linalg.norm(item['embedding']))
        scores.append((score, item['text']))
    # Sort by similarity score
    scores.sort(reverse=True, key=lambda x: x[0])
    return [text for _, text in scores[:top_k]]

# Load embeddings model
embeddings = download_hugging_face_embeddings()
csv_data = load_csv_data("output.csv")

# LLM model and prompt setup
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Switch to Hugging Face's pipeline for DistilGPT-2
llm = pipeline('text-generation', model='distilgpt2')

# QA chain setup
def qa_chain(query):
    query_embedding = embeddings.embed_query(query)
    retrieved_texts = retrieve_from_csv(csv_data, query_embedding, top_k=2)
    context = " ".join(retrieved_texts)
    
    # Generate response from the LLM
    result = llm(f"Context: {context}\nQuestion: {query}", max_length=450, num_return_sequences=1)
    
    # Access the generated text
    generated_text = result[0]['generated_text']
    
    return generated_text

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)
    result = qa_chain(msg)
    a=len(result)
    for i in range(a):
        if i >= (a/2):
            if result[i]=='.':
                b=result[:i]
                break
   

# Sample data
    data = {
        'Name': [
            'Dr. John Smith', 'Dr. Emily Davis', 'Dr. Michael Brown', 'Dr. Linda Johnson', 
            'Dr. Robert White', 'Dr. Jessica Lee', 'Dr. Daniel Clark', 'Dr. Laura Wilson', 
            'Dr. James Martinez', 'Dr. Sarah Taylor', 'Dr. William Anderson', 'Dr. Nancy Thompson', 
            'Dr. Richard Garcia', 'Dr. Karen Hall', 'Dr. Steven Adams', 'Dr. Elizabeth Wright', 
            'Dr. David Scott', 'Dr. Patricia Young', 'Dr. Charles King', 'Dr. Susan Evans'
        ],
        'Location': [
            'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 
            'Phoenix, AZ', 'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA', 
            'Dallas, TX', 'San Jose, CA', 'Austin, TX', 'Jacksonville, FL', 
            'San Francisco, CA', 'Indianapolis, IN', 'Columbus, OH', 'Fort Worth, TX', 
            'Charlotte, NC', 'Detroit, MI', 'El Paso, TX', 'Seattle, WA'
        ],
        'Contact': [
            '555-1234', '555-5678', '555-8765', '555-4321', 
            '555-9876', '555-6543', '555-3456', '555-7890', 
            '555-2345', '555-6789', '555-3457', '555-8901', 
            '555-4567', '555-8902', '555-6780', '555-1235', 
            '555-9875', '555-4320', '555-8764', '555-5432'
        ]
    }

    # Creating DataFrame
    doctor_df = pd.DataFrame(data)

    # Function to convert a DataFrame row to a string
    def doctor_to_string(row):
        return f"Name: {row['Name']}, Location: {row['Location']}, Contact: {row['Contact']}"

    # Example usage
    random_doctor = doctor_df.sample().iloc[0]  # Get a random row and convert to Series
    doctor_string = doctor_to_string(random_doctor)
    b=b+".In such serious cases, please consult the doctor "+doctor_string
    print("Response:", b)
    return str(b)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)