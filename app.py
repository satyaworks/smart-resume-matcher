import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import io
import re

st.set_page_config(page_title="Simple Resume Matcher")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Helper function to extract keywords
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set([
        "and", "or", "with", "in", "on", "the", "a", "an", "to", "of", "for",
        "we", "you", "are", "is", "looking", "need", "have", "has"
    ])
    return set([w for w in words if w not in stopwords and len(w) > 2])

# UI
st.title("🧠 Smart Resume Matcher")

uploaded_jd = st.file_uploader("📄 Upload Job Description (PDF)", type=["pdf"])
uploaded_resumes = st.file_uploader("📁 Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

threshold_str = st.selectbox("Select Minimum Match Score", ["0.2","0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
threshold = float(threshold_str)

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
    filtered_df = df[df["Match Score"] >= threshold]

    st.subheader(f"🎯 Match Results (Filtered by Score ≥ {threshold})")
    st.dataframe(filtered_df, use_container_width=True)

    if not filtered_df.empty:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Results", data=csv_data, file_name="resume_match_results.csv", mime="text/csv")
    else:
        st.info("No resumes matched the selected threshold.")
