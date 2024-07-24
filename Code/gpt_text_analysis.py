import openai
import os
import json
from pdfminer.high_level import extract_text
import requests
from bs4 import BeautifulSoup
import tempfile
from dotenv import load_dotenv
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from io import BytesIO
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index_name = "resumebuilder"
namespace = "ns1"

# Ensure the index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust the dimension based on your embeddings model
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

index = pc.Index(index_name)

def extract_text_from_pdf(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name

        text = extract_text(temp_file_path)
        os.unlink(temp_file_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# def embed_text(text):
#     try:
#         response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
#         return np.array(response['data'][0]['embedding'], dtype=np.float32)
#     except openai.error.OpenAIError as e:
#         if 'You exceeded your current quota' in str(e):
#             st.error(f"Error generating embedding: {e}")
#         else:
#             st.error(f"Error generating embedding: {e}")
#         return np.array([])

# def store_resume_in_pinecone(resume_id, resume_text):
#     embedding = embed_text(resume_text)
#     if embedding.size > 0:
#         index.upsert(
#             vectors=[
#                 {
#                     "id": resume_id,
#                     "values": embedding.tolist(),
#                     "metadata": {"text": resume_text}
#                 }
#             ],
#             namespace=namespace
#         )

def parse_resume(resume_text):
    prompt = f"""Please rank the following resume on a scale of 0-3 for each of the following traits: 
    Leadership, Communication, Teamwork, Problem Solving, Creativity, Adaptability, Work Ethic, Time Management, Interpersonal Skills, Attention to Detail, Initiative, Analytical Thinking, Emotional, Intelligence, Integrity, Resilience, Cultural Awareness, Programming Languages, Technical Skills, Office Tools
    return a dictionary of trait to value.

    Resume:
    {resume_text}"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional resume analyzer."},
                {"role": "user", "content": prompt}
            ]
        )
        optimized_resume = response.choices[0].message['content'].strip()
        return optimized_resume
    except Exception as e:
        st.error(f"Error optimizing resume: {e}")
        return ""

def file_to_parsed_res(file):
    text = extract_text_from_pdf(file)
    text = text.replace("'", '"')
    parse = json.loads(parse_resume(text))
    return parse
