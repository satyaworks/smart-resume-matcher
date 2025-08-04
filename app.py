import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import io
import re
import datetime
import torch

st.set_page_config(page_title="Simple Resume Matcher")

# Load model (force CPU for Streamlit Cloud)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Helper function to extract keywords
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(["and", "or", "with", "in", "on", "the", "a", "an", "to", "of", "for", "we", "you", "are", "is", "looking", "need", "have", "has"])
    return set([w for w in words if w not in stopwords and len(w) > 2])

# Streamlit UI
st.title("🧠 Simple Resume Matcher")

uploaded_jd = st.file_uploader("📄 Upload Job Description (PDF)", type=["pdf"])
uploaded_resumes = st.file_uploader("📁 Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("🔍 Match Resumes") and uploaded_jd and uploaded_resumes:
    with pdfplumber.open(io.BytesIO(uploaded_jd.read())) as jd_pdf:
        job_text = ''.join([page.extract_text() or "" for page in jd_pdf.pages])

    jd_keywords = extract_keywords(job_text)
    job_embedding = model.encode([job_text])

    results = []
    for file in uploaded_resumes:
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            resume_text = ''.join([page.extract_text() or "" for page in pdf.pages])

        if not resume_text.strip():
            st.warning(f"{file.name} is empty or unreadable.")
            continue

        resume_embedding = model.encode([resume_text])
        score = np.dot(job_embedding, resume_embedding.T)[0]
        matched_keywords = ", ".join(sorted(jd_keywords.intersection(extract_keywords(resume_text))))

        results.append({
            "Resume": file.name,
            "Match Score": round(float(score), 4),
            "Matched Keywords": matched_keywords,
        })

    df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)
    st.subheader("🎯 Match Results")
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", data=csv_data, file_name="resume_match_results.csv", mime="text/csv")
