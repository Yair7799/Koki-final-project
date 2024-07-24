import streamlit as st
import openai
import os

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


# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone

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

def embed_text(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except openai.error.OpenAIError as e:
        if 'You exceeded your current quota' in str(e):
            st.error(f"Error generating embedding: {e}")
        else:
            st.error(f"Error generating embedding: {e}")
        return np.array([])

def store_resume_in_pinecone(resume_id, resume_text):
    embedding = embed_text(resume_text)
    if embedding.size > 0:
        index.upsert(
            vectors=[
                {
                    "id": resume_id,
                    "values": embedding.tolist(),
                    "metadata": {"text": resume_text}
                }
            ],
            namespace=namespace
        )

def fetch_job_description(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        job_description = soup.get_text()
        return job_description
    except Exception as e:
        st.error(f"Error fetching job description: {e}")
        return ""

def match_resumes_to_job_description(job_description):
    job_embedding = embed_text(job_description)
    if job_embedding.size > 0:
        results = index.query(
            namespace=namespace,
            vector=job_embedding.tolist(),
            top_k=1,
            include_values=True,
            include_metadata=True
        )
        return results
    return []

def optimize_resume(resume_file, job_url):
    resume_text = extract_text_from_pdf(resume_file)
    job_description = fetch_job_description(job_url)

    prompt = f"""Optimize the following resume for the job description. 
    The optimized resume should include the following sections in order: 
    Summary, Skills, Work Experience, Certificates, and Education. 
    Start with the person's name and the profession mentioned in the job description.
    Ensure all work experience is included, but prioritize relevant experience.
    The entire resume must fit on one page, so be concise while preserving key information.
    Use bullet points (•) for listing items, not dashes (-).
    Do not include any additional text or explanations outside of the resume content.

    Job Description: {job_description}

    Resume:
    {resume_text}"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional resume optimizer."},
                {"role": "user", "content": prompt}
            ]
        )
        optimized_resume = response.choices[0].message['content'].strip()
        return optimized_resume
    except Exception as e:
        st.error(f"Error optimizing resume: {e}")
        return ""

def create_pdf(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.5*inch, rightMargin=0.5*inch)
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name='Name', fontSize=14, leading=16, spaceAfter=4, alignment=1))
    styles.add(ParagraphStyle(name='Profession', fontSize=12, leading=14, spaceAfter=8, alignment=1))
    styles.add(ParagraphStyle(name='Section', fontSize=11, leading=13, spaceAfter=4, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Content', fontSize=9, leading=11, spaceAfter=2, bulletIndent=10, leftIndent=20))
    styles.add(ParagraphStyle(name='ContentNoBullet', fontSize=9, leading=11, spaceAfter=2, leftIndent=10))

    content = []
    lines = text.split('\n')

    if len(lines) > 1:
        content.append(Paragraph(lines[0], styles['Name']))
        content.append(Paragraph(lines[1], styles['Profession']))
        content.append(Spacer(1, 8))

        current_section = ''
        for line in lines[2:]:
            line = line.strip()
            if line in ['Summary:', 'Skills:', 'Work Experience:', 'Certificates:', 'Education:']:
                current_section = line
                content.append(Paragraph(f"<b>{current_section}</b>", styles['Section']))
            elif line.startswith('•'):
                content.append(Paragraph(line, styles['Content']))
            elif line:
                content.append(Paragraph(line, styles['ContentNoBullet']))

        doc.build(content)
        buffer.seek(0)
        return buffer
    else:
        st.error("The optimized resume does not have the expected format.")
        return None