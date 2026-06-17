
import streamlit as st
import pandas as pd
import pickle
import re
try:
    import nltk
    st.write("NLTK imported successfully")
except Exception as e:
    st.error(f"NLTK ERROR: {e}")
    st.stop()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import io
import os

# Configure NLTK to use the locally bundled data
os.environ['NLTK_DATA'] = 'nltk_data'

# Initialize NLTK components globally to avoid re-initialization in functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Load Models and Vectorizer with caching ---
@st.cache_resource
def load_artifacts():
    model_path = 'models/'

    with open(model_path + 'label_encoder.pkl', 'rb') as file:
        le = pickle.load(file)
    with open(model_path + 'tfidf_vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    with open(model_path + 'logistic_regression_model.pkl', 'rb') as file:
        logistic_model = pickle.load(file)

    return le, tfidf_vectorizer, logistic_model

le, tfidf_vectorizer, classification_model = load_artifacts()

# --- Define Common Skills List ---
common_skills = [
    "python", "java", "c++", "c#", "javascript", "html", "css", "sql", "nosql",
    "aws", "azure", "google cloud", "docker", "kubernetes", "git", "github",
    "machine learning", "deep learning", "nlp", "data analysis", "data science",
    "statistics", "excel", "power bi", "tableau", "r", "sas", "matlab",
    "agile", "scrum", "project management", "communication", "leadership",
    "teamwork", "problem solving", "critical thinking", "adaptability",
    "time management", "customer service", "sales", "marketing", "seo", "sem",
    "content creation", "ui/ux", "frontend development", "backend development",
    "full stack development", "database management", "cloud computing", "cybersecurity",
    "network administration", "devops", "software development", "web development",
    "mobile development", "api development", "testing", "quality assurance",
    "technical writing", "research", "financial analysis", "accounting",
    "business intelligence", "strategic planning", "risk management", "supply chain management",
    "logistics", "manufacturing", "autocad", "solidworks", "sap", "oracle erp"
]

# --- Helper Functions ---
def clean_resume_text(text):
    text = text.lower()
    text = re.sub(r'http\S+\s*', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    text = ' '.join([word for word in tokens if word not in stop_words])
    tokens = text.split()
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])
    return text

def predict_resume_category(raw_resume_text):
    cleaned_text = clean_resume_text(raw_resume_text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    predicted_category_encoded = classification_model.predict(vectorized_text)
    predicted_category = le.inverse_transform(predicted_category_encoded)
    return predicted_category[0]

def process_text(text):
    cleaned_text = clean_resume_text(text)
    tfidf_vector = tfidf_vectorizer.transform([cleaned_text])
    return tfidf_vector

def calculate_similarity(job_description_text, resume_text):
    job_desc_vector = process_text(job_description_text)
    resume_vector = process_text(resume_text)
    similarity_score = cosine_similarity(job_desc_vector, resume_vector)[0][0]
    return similarity_score

def extract_skills(cleaned_resume_text, skills_list):
    found_skills = []
    text_lower = cleaned_resume_text.lower()
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill)
    return found_skills

def calculate_skill_gap(candidate_extracted_skills, job_relevant_skills):
    candidate_skills_set = set(candidate_extracted_skills)
    job_skills_set = set(job_relevant_skills)
    missing_skills = list(job_skills_set - candidate_skills_set)
    return missing_skills

def rank_candidate(job_description_text, candidate_raw_resume_text, desired_job_category, job_relevant_skills):
    predicted_category = predict_resume_category(candidate_raw_resume_text)
    cleaned_resume_text = clean_resume_text(candidate_raw_resume_text)
    similarity_score = calculate_similarity(job_description_text, cleaned_resume_text)
    extracted_skills = extract_skills(cleaned_resume_text, common_skills)
    missing_skills = calculate_skill_gap(extracted_skills, job_relevant_skills)
    category_relevance_score = 1 if predicted_category.lower() == desired_job_category.lower() else 0

    matching_relevant_skills_count = len(set(job_relevant_skills).intersection(set(extracted_skills)))
    skill_score = matching_relevant_skills_count / len(job_relevant_skills) if len(job_relevant_skills) > 0 else 0

    weight_category = 0.3
    weight_similarity = 0.5
    weight_skills = 0.2

    ranking_score = (
        weight_category * category_relevance_score +
        weight_similarity * similarity_score +
        weight_skills * skill_score
    )

    return {
        'original_resume': candidate_raw_resume_text,
        'predicted_category': predicted_category,
        'job_description_similarity': similarity_score,
        'extracted_skills': extracted_skills,
        'missing_skills': missing_skills,
        'ranking_score': ranking_score
    }

# --- Text Extraction Function ---
def extract_text_from_upload(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() if page.extract_text() else ""
            return text
    else:
        return ""


# Streamlit App Title and Configuration
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# --- Custom CSS for Professional Styling ---
custom_css = """
<style>
/* General Styling */
body {
    color: #eee;
    background-color: #1e1e1e;
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #1e1e1e;
}

/* Sidebar Styling */
.css-pkajpt {
    background-color: #2b2b2b;
    color: #eee;
}
.css-1lcbmhc {
    background-color: #2b2b2b; /* For collapsed sidebar */
}

/* Header and Titles */
h1, h2, h3, h4, h5, h6 {
    color: #00b0f0; /* Blue accent for headers */
    font-weight: 600;
}

/* Markdown Text */
.stMarkdown {
    color: #ccc;
}

/* Buttons */
.stButton>button {
    background-color: #007bff; /* Primary blue button */
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.2s, color 0.2s;
}

.stButton>button:hover {
    background-color: #0056b3;
    color: #fff;
}

/* Text Areas and Inputs */
.stTextArea>div>div>textarea,
.stTextInput>div>div>input {
    background-color: #333;
    color: #fff;
    border: 1px solid #00b0f0;
    border-radius: 5px;
    padding: 10px;
}

/* Expander Styling */
.streamlit-expanderHeader {
    background-color: #333;
    color: #00b0f0;
    border-radius: 5px;
    padding: 10px;
    margin-top: 10px;
    border: 1px solid #00b0f0;
}
.streamlit-expanderContent {
    background-color: #2b2b2b;
    border-left: 2px solid #00b0f0;
    padding: 10px;
    border-bottom-left-radius: 5px;
    border-bottom-right-radius: 5px;
}

/* Dataframe Styling */
.stDataFrame {
    background-color: #2b2b2b;
    color: #eee;
    border: 1px solid #00b0f0;
    border-radius: 5px;
}
.dataframe thead th {
    background-color: #007bff !important;
    color: white !important;
}
.dataframe tbody tr {
    background-color: #2b2b2b !important;
    color: #eee !important;
}
.dataframe tbody tr:nth-child(odd) {
    background-color: #333 !important;
}

/* Info, Warning, Success messages */
.stAlert > div {
    border-radius: 5px;
}
.stAlert.info {
    background-color: #17a2b8;
    color: white;
}
.stAlert.warning {
    background-color: #ffc107;
    color: black;
}
.stAlert.success {
    background-color: #28a745;
    color: white;
}

/* Multiselect / File Uploader */
.stFileUploader label {
    color: #eee;
}
.stFileUploader > div > div > button {
    background-color: #00b0f0;
    color: white;
}
.stFileUploader > div > div > button:hover {
    background-color: #008cba;
}
.stMultiSelect label {
    color: #eee;
}
.stMultiSelect div[data-baseweb="select"] {
    background-color: #333;
    color: #fff;
    border: 1px solid #00b0f0;
    border-radius: 5px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("AI-Powered Resume Screening and Candidate Ranking System")
st.markdown("""
<p style='font-size: 1.1em; color: #ccc;'>
    This application helps recruiters efficiently screen and rank job applicants based on job descriptions and resumes.
</p>
<hr style='border-top: 1px solid #00b0f0;'>
""")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses NLP and machine learning to categorize resumes, calculate similarity "
    "to job descriptions, extract skills, and rank candidates. "
    "Developed by Google Colab AI Agent. "
)

st.sidebar.header("Deployment Guide")
st.sidebar.markdown("""
### 1. Run Locally
To run this application on your local machine, follow these steps:

1.  **Save the files**: Ensure `app.py`, `requirements.txt`, and the `models/` directory (containing all `.pkl` files) are in the same directory.
2.  **Install dependencies**: Open your terminal or command prompt, navigate to the project directory, and run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the app**: Execute the Streamlit application using:
    ```bash
    streamlit run app.py
    ```
    Your browser will automatically open the application.

### 2. Deploy on Streamlit Cloud
To deploy this application publicly using Streamlit Cloud, follow these steps:

1.  **GitHub Repository**: Create a GitHub repository with `app.py`, `requirements.txt`, and the `models/` directory.
2.  **Connect to Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io/), log in, and click 'New app'.
3.  **Select Repository**: Choose your GitHub repository, the branch (e.g., `main`), and set the main file path to `app.py`.
4.  **Deploy**: Click 'Deploy!' Streamlit Cloud will handle the environment setup and deploy your app.

**Important**: Ensure your `requirements.txt` is accurate and all model `.pkl` files are correctly placed in the `models/` directory within your GitHub repository.
""")




# Main content area for job description input
st.header("1. Enter Job Description")

# Use st.columns for better layout of input fields
cols1, cols2 = st.columns([2, 1])

with cols1:
    job_description = st.text_area(
        "Paste the Job Description here:",
        height=300,
        key="job_description_input",
        help="Enter the full text of the job description for analysis."
    )

with cols2:
    st.write("### ") # Spacer to align inputs
    desired_job_category = st.text_input(
        "Enter the Desired Job Category:",
        key="desired_job_category_input",
        help="e.g., 'Data Science', 'Engineering'. This helps in classifying resumes accurately."
    )
    job_relevant_skills_input = st.text_area(
        "Enter Job Relevant Skills (comma-separated):",
        key="job_relevant_skills_input",
        help="Provide a comma-separated list of essential skills for the job."
    )

# Process Job Description button
if st.button("Process Job Description", key="process_job_desc_button"):
    if job_description and desired_job_category and job_relevant_skills_input:
        st.session_state.job_description = job_description
        st.session_state.desired_job_category = desired_job_category
        st.session_state.job_relevant_skills = [s.strip().lower() for s in job_relevant_skills_input.split(',') if s.strip()]
        st.success("Job Description processed successfully!")
    else:
        st.warning("Please fill in all fields (Job Description, Desired Job Category, and Job Relevant Skills) to process.")


# --- Resume Upload and Processing ---
st.header("3. Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF or TXT) - Single or Multiple:",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="resume_uploader"
)

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} resumes...")

    # Store processed resumes
    processed_resumes = []

    for uploaded_file in uploaded_files:
        resume_text = extract_text_from_upload(uploaded_file)
        if resume_text:
            cleaned_resume = clean_resume_text(resume_text)
            processed_resumes.append({
                'file_name': uploaded_file.name,
                'raw_text': resume_text,
                'cleaned_text': cleaned_resume
            })
        else:
            st.warning(f"Could not extract text from {uploaded_file.name}. Skipping.")

    if processed_resumes:
        st.success(f"Successfully processed {len(processed_resumes)} resumes.")
        # Store processed resumes in session state for later use
        st.session_state.processed_resumes = processed_resumes

# --- Display Candidate Analysis Results ---
st.header("4. Candidate Analysis Results")

if "processed_resumes" in st.session_state and st.session_state.processed_resumes and    "job_description" in st.session_state and st.session_state.job_description and    "desired_job_category" in st.session_state and st.session_state.desired_job_category and    "job_relevant_skills" in st.session_state and st.session_state.job_relevant_skills:

    job_description_for_ranking = st.session_state.job_description
    desired_job_category_for_ranking = st.session_state.desired_job_category
    job_relevant_skills_for_ranking = st.session_state.job_relevant_skills

    all_candidate_rankings = []

    for resume_data in st.session_state.processed_resumes:
        file_name = resume_data['file_name']
        raw_text = resume_data['raw_text']

        ranking_result = rank_candidate(
            job_description_for_ranking,
            raw_text,
            desired_job_category_for_ranking,
            job_relevant_skills_for_ranking
        )

        all_candidate_rankings.append({
            'Resume Name': file_name,
            'Predicted Category': ranking_result['predicted_category'],
            'Job Description Similarity': f"{ranking_result['job_description_similarity']:.2f}",
            'Extracted Skills': ', '.join(ranking_result['extracted_skills']),
            'Missing Skills': ', '.join(ranking_result['missing_skills']) if ranking_result['missing_skills'] else 'None',
            'Ranking Score': f"{ranking_result['ranking_score']:.2f}"
        })

    if all_candidate_rankings:
        st.subheader("Consolidated Candidate Ranking Table")
        df_rankings = pd.DataFrame(all_candidate_rankings)
        df_rankings['Ranking Score (Numeric)'] = df_rankings['Ranking Score'].astype(float)
        df_rankings = df_rankings.sort_values(by='Ranking Score (Numeric)', ascending=False)
        df_rankings = df_rankings.drop(columns=['Ranking Score (Numeric)'])

        st.dataframe(df_rankings, use_container_width=True)

        st.subheader("Detailed Analysis for Each Candidate")
        for resume_data_in_expander in st.session_state.processed_resumes:
            file_name = resume_data_in_expander['file_name']
            raw_text = resume_data_in_expander['raw_text']

            ranking_result = rank_candidate(
                job_description_for_ranking,
                raw_text,
                desired_job_category_for_ranking,
                job_relevant_skills_for_ranking
            )
            with st.expander(f"**{file_name}** - Predicted Category: {ranking_result['predicted_category']}"):
                st.write(f"**Predicted Category:** {ranking_result['predicted_category']}")
                st.write(f"**Job Description Similarity:** {ranking_result['job_description_similarity']:.2f}")
                st.write(f"**Extracted Skills:** {', '.join(ranking_result['extracted_skills'])}")
                if ranking_result['missing_skills']:
                    st.write(f"**Missing Skills (for desired job):** :red[{', '.join(ranking_result['missing_skills'])}]")
                else:
                    st.write("**Missing Skills (for desired job):** None :green[All relevant skills found!] ")
                st.write(f"**Overall Ranking Score:** {ranking_result['ranking_score']:.2f}")
    else:
        st.info("No candidate rankings to display yet. Upload resumes and process the job description.")

else:
    st.info("Please enter job details, upload resumes, and click 'Process Job Description' to see candidate analysis.")

