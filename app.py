import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import io
import re
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple Resume Matcher")

# Load sentence transformer model (CPU)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Helper function
def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set([
        "and", "or", "with", "in", "on", "the", "a", "an", "to", "of", "for",
        "we", "you", "are", "is", "looking", "need", "have", "has"
    ])
    return set([w for w in words if w not in stopwords and len(w) > 2])

# Load jobs.csv (for dropdown)
@st.cache_data
def load_job_descriptions():
    try:
        df = pd.read_csv("jobs.csv")
        return df
    except:
        return pd.DataFrame(columns=["id", "description"])

job_df = load_job_descriptions()

# Streamlit UI
st.title("🧠 Simple Resume Matcher")

uploaded_jd = st.file_uploader("📄 Upload Job Description (PDF)", type=["pdf"])
uploaded_resumes = st.file_uploader("📁 Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

# Dropdown job description
selected_job = None
if not uploaded_jd and not job_df.empty:
    jd_options = job_df["description"].tolist()
    selected_job = st.selectbox("🧾 Or Select a Job Description", [""] + jd_options)

# Match resumes
if st.button("🔍 Match Resumes") and uploaded_resumes and (uploaded_jd or selected_job):
    # Job description text
    if uploaded_jd:
        with pdfplumber.open(io.BytesIO(uploaded_jd.read())) as jd_pdf:
            job_text = ''.join([page.extract_text() or "" for page in jd_pdf.pages])
    elif selected_job:
        job_text = selected_job
    else:
        st.error("Please provide a job description.")
        st.stop()

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

    # Filter threshold slider
    st.subheader("🎚️ Filter by Match Score")
    threshold = st.slider("Minimum Score", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    filtered_df = df[df["Match Score"] >= threshold]

    st.subheader("🎯 Filtered Match Results")
    st.dataframe(filtered_df, use_container_width=True)

    # Chart for filtered results
    st.subheader("📊 Match Score Chart (Filtered)")
    if not filtered_df.empty:
        fig, ax = plt.subplots()
        ax.barh(filtered_df["Resume"], filtered_df["Match Score"], color="skyblue")
        ax.set_xlabel("Match Score")
        ax.set_ylabel("Resumes")
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.info("No resumes matched the selected threshold.")

    # Download
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Results as CSV", data=csv_data, file_name="filtered_resume_results.csv", mime="text/csv")
