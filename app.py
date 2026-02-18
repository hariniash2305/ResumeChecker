import streamlit as st
import pdfplumber
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# You need to set your Gemini API key in an environment variable named GEMINI_API_KEY
# or create a .env file with GEMINI_API_KEY=your_api_key_here
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- Classes ---

class ResumeParser:
    def extract_text(self, uploaded_file):
        """Extracts text from a PDF file."""
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            st.error(f"Error parsing {uploaded_file.name}: {e}")
            return ""

class EmbeddingEngine:
    def __init__(self):
        # Load a lightweight, efficient model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embeddings(self, text_list):
        """Generates embeddings for a list of texts."""
        return self.model.encode(text_list)

class VectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = [] # List to store (filename, text) tuples

    def add_resumes(self, embeddings, metadata_list):
        """Adds resume embeddings and metadata to the store."""
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata.extend(metadata_list)

    def search(self, query_embedding, k=25):
        """Searches for the top k most similar resumes."""
        if self.index.ntotal == 0:
            return [], []
        
        # Ensure k doesn't exceed the number of resumes
        k = min(k, self.index.ntotal)
        
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        return distances[0], indices[0]

class GeminiEvaluator:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def evaluate_candidate(self, resume_text, job_description):
        """Generates an evaluation for a candidate based on the job description."""
        prompt = f"""
        You are an expert technical recruiter. Evaluate the following resume against the job description.
        
        Job Description:
        {job_description}

        Resume Text:
        {resume_text}

        Provide a structured response with the following:
        1. Suitability Summary (2 lines max)
        2. Key Strengths (bullet points)
        3. Missing Areas (bullet points)
        4. Hiring Recommendation (Yes/Maybe/No)
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error triggering Gemini API: {e}"

# --- Main App Logic ---

def main():
    st.set_page_config(page_title="AI Resume Screener", layout="wide")
    st.title("AI-Powered Bulk Resume Screener")

    # Initialize components (cached to avoid reloading on every interaction)
    @st.cache_resource
    def get_embedding_engine():
        return EmbeddingEngine()

    @st.cache_resource
    def get_gemini_evaluator():
        return GeminiEvaluator()

    embedding_engine = get_embedding_engine()
    gemini_evaluator = get_gemini_evaluator()

    # Session State for Vector Store
    if 'vector_store' not in st.session_state:
        # 384 is the dimension for all-MiniLM-L6-v2
        st.session_state.vector_store = VectorStore(dimension=384)
    
    if 'processed_resumes' not in st.session_state:
        st.session_state.processed_resumes = set()

    # UI Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("1. Upload Resumes")
        uploaded_files = st.file_uploader("Upload PDF Resumes", type=['pdf'], accept_multiple_files=True)
        
        if uploaded_files:
            parser = ResumeParser()
            new_resumes = []
            new_metadata = []
            
            for file in uploaded_files:
                if file.name not in st.session_state.processed_resumes:
                    with st.spinner(f"Parsing {file.name}..."):
                        text = parser.extract_text(file)
                        if text:
                            new_resumes.append(text)
                            new_metadata.append({"filename": file.name, "text": text})
                            st.session_state.processed_resumes.add(file.name)
            
            if new_resumes:
                with st.spinner("Generating embeddings and updating vector store..."):
                    embeddings = embedding_engine.generate_embeddings(new_resumes)
                    st.session_state.vector_store.add_resumes(embeddings, new_metadata)
                st.success(f"Processed {len(new_resumes)} new resumes!")
            elif len(uploaded_files) > 0 and len(new_resumes) == 0:
                 st.info("Resumes already processed.")

        st.divider()
        st.header("2. Job Description")
        job_description = st.text_area("Paste Job Description Here", height=300)
        analyze_button = st.button("Analyze Candidates")

    with col2:
        st.header("3. Results")
        if analyze_button and job_description:
            if st.session_state.vector_store.index.ntotal == 0:
                st.warning("Please upload resumes first.")
            else:
                with st.spinner("Searching and ranking candidates..."):
                    # 1. Generate embedding for JD
                    jd_embedding = embedding_engine.generate_embeddings([job_description])[0]
                    
                    # 2. Search Vector Store
                    distances, indices = st.session_state.vector_store.search(jd_embedding, k=25)
                    
                    # 3. Process Results
                    results = []
                    for i, idx in enumerate(indices):
                        if idx != -1: # FAISS returns -1 for empty slots if k > ntotal
                            meta = st.session_state.vector_store.metadata[idx]
                            # Convert L2 distance to a similarity score (approximate)
                            # Lower distance means higher similarity. 
                            # Simple conversion: 1 / (1 + distance) or just normalized
                            score = 100 * (1 / (1 + distances[i])) 
                            results.append({
                                "rank": i + 1,
                                "filename": meta['filename'],
                                "score": score,
                                "text": meta['text']
                            })

                    # 4. Gemini Evaluation for Top 3
                    st.subheader(f"Top {min(3, len(results))} Candidates Analysis")
                    
                    for i, result in enumerate(results[:3]):
                        with st.expander(f"Rank {result['rank']}: {result['filename']} (Score: {result['score']:.2f}%)", expanded=True):
                            with st.spinner(f"Asking Gemini to evaluate {result['filename']}..."):
                                evaluation = gemini_evaluator.evaluate_candidate(result['text'], job_description)
                                st.markdown(evaluation)

                    # 5. Display List of Others
                    if len(results) > 3:
                        st.subheader("Other Candidates")
                        for result in results[3:]:
                             st.write(f"**Rank {result['rank']}**: {result['filename']} - Score: {result['score']:.2f}%")

if __name__ == "__main__":
    main()