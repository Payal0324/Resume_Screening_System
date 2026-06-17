# 📊 AI Powered Resume Screening & Candidate Ranking System

A machine learning-powered web application that automatically screens resumes, extracts skills, matches candidates with job descriptions, and ranks them based on relevance using NLP techniques.

👉 Live App: https://resumescreeningsystem-swwx4gqn69zsionamlbnom.streamlit.app/  

---

## 🚀 Features

- Upload multiple resumes (PDF / TXT)
- NLP-based resume text processing
- Automatic resume classification using ML model
- Job description similarity scoring using TF-IDF + Cosine Similarity
- Skill extraction from resumes
- Skill gap analysis (missing skills detection)
- AI-based candidate ranking system
- Interactive Streamlit dashboard UI

---

## 🧠 Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Machine Learning:** Scikit-learn  
- **NLP:** NLTK, TF-IDF Vectorizer  
- **PDF Parsing:** pdfplumber  
- **Models Used:**
  - Logistic Regression
  - Random Forest
  - Naive Bayes (training phase)

---

## 📂 Project Structure

```bash
Resume_Screening_System/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
│
├── models/                       # Trained ML models
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── logistic_regression_model.pkl
│   ├── naive_bayes_model.pkl
│   └── random_forest_model.pkl
│
├── nltk_data/                    # NLTK required datasets
│   ├── stopwords/
│   ├── wordnet/
│   └── omw-1.4/
│
├── Resume.zip                    # Resume dataset (for training/testing)
│
└── .streamlit/                   # Streamlit config (optional)
```

---

## ⚙️ How It Works

### 1. Input Job Description
User provides:
- Job description
- Desired job category
- Required skills

### 2. Upload Resumes
Upload multiple resumes in PDF or TXT format.

### 3. NLP Processing
- Lowercasing
- Stopword removal
- Lemmatization

### 4. Feature Extraction
- TF-IDF vectorization
- Skill extraction using regex matching

### 5. Model Prediction
- Resume classification using trained ML model
- Similarity calculation using cosine similarity

### 6. Candidate Ranking Formula

```

Ranking Score =
0.3 × Category Match +
0.5 × Similarity Score +
0.2 × Skill Match

```

---

## 📊 Example Use Case

- HR uploads multiple resumes  
- Enters job description (e.g., Data Scientist role)  
- System:
  - Extracts skills  
  - Matches resumes with job description  
  - Ranks candidates automatically  
- Output: Sorted list of best candidates  

---

## 🔥 Future Improvements

- BERT-based NLP model for better accuracy  
- Resume database storage system  
- AI interview question generator  
- Advanced analytics dashboard  
- Multi-role job support system  

---

## 👩‍💻 Author

**Payal Jangale**
