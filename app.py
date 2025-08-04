# app.py

import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import io
import re
import datetime
from sqlalchemy import create_engine, Table, Column, String, Float, MetaData

st.set_page_config(page_title="Smart Resume Matcher", layout="wide")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# PostgreSQL DB setup
DATABASE_URL = st.secrets["DATABASE_URL"]  # Define in Streamlit Cloud or .streamlit/secrets.toml
engine = create_engine(DATABASE_URL)
metadata = MetaData()

match_table = Table("resume_matches", metadata,
    Column("timestamp", String),
    Column("resume_name", String),
    Column("match_score", Float),
    Column("matched_keywords", String),
    Column("job_description", String)
)
metadata.create_all(engine)

# Helper function to extract keywords
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set([
        "and", "or", "with", "in", "on", "the", "a", "an", "to", "of", "for", "we", "you",
        "are", "is", "looking", "need", "have", "has"
    ])
    return set([word for word in words if word not in stopwords and len(word) > 2])

st.title("\U0001F9E0 Smart Resume Matcher")

# Upload custom JD or choose from default
uploaded_jd = st.file_uploader("\U0001F4C4 Upload Job Description (PDF)", type=["pdf"])

if uploaded_jd:
    with pdfplumber.open(io.BytesIO(uploaded_jd.read())) as jd_pdf:
        job_text = ""
        for page in jd_pdf.pages:
            job_text += page.extract_text() or ""

    if len(job_text.strip()) == 0:
        st.warning("Job Description file is empty or unreadable.")
        st.stop()

    st.success("✅ Job description uploaded successfully!")
    with st.expander("\U0001F4C4 View Uploaded Job Description"):
        st.write(job_text)
else:
    df_jobs = pd.read_csv("jobs.csv")
    job_texts = df_jobs["description"].tolist()
    job_titles = [f"Job {i+1}" for i in range(len(job_texts))]

    selected_job_index = st.selectbox("\U0001F4CC Or select a predefined job", range(len(job_texts)), format_func=lambda i: job_titles[i])
    job_text = job_texts[selected_job_index]

    with st.expander("\U0001F4C4 View Selected Job Description"):
        st.write(job_text)

jd_keywords = extract_keywords(job_text)

# Upload multiple resumes
uploaded_files = st.file_uploader("\U0001F4C4 Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("\U0001F50D Match Resumes") and uploaded_files:
    job_embedding = model.encode([job_text])
    results = []

    for file in uploaded_files:
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            resume_text = ""
            for page in pdf.pages:
                resume_text += page.extract_text() or ""

        if len(resume_text.strip()) == 0:
            st.warning(f"{file.name} is empty or unreadable.")
            continue

        resume_embedding = model.encode([resume_text])
        score = np.dot(job_embedding, resume_embedding.T)[0]

        resume_keywords = extract_keywords(resume_text)
        overlap = jd_keywords.intersection(resume_keywords)
        overlap_display = ", ".join(sorted(overlap))

        results.append({
            "Resume": file.name,
            "Match Score": round(float(score), 4),
            "Matched Keywords": overlap_display,
            "Preview": resume_text[:300] + "..."
        })

        # Save to DB
        with engine.connect() as conn:
            conn.execute(
                match_table.insert().values(
                    timestamp=str(datetime.datetime.now()),
                    resume_name=file.name,
                    match_score=float(score),
                    matched_keywords=overlap_display,
                    job_description=job_text[:300]
                )
            )

    # Show results
    results_df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False).reset_index(drop=True)
    st.subheader("\U0001F3AF Match Results")
    st.dataframe(results_df[["Resume", "Match Score", "Matched Keywords"]], use_container_width=True)

    # Download
    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("\U0001F4C5 Download Results as CSV", data=csv_data, file_name="resume_match_results.csv", mime="text/csv")

# Match history
with st.expander("\U0001F4CA View Match History (Database)"):
    with engine.connect() as conn:
        history_df = pd.read_sql("SELECT * FROM resume_matches ORDER BY timestamp DESC LIMIT 100", conn)
    st.dataframe(history_df)
