import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np

# Load models
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Best quality embeddings

# Preprocess text
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Keyword overlap score (0-100)
def keyword_score(resume, job_desc):
    vectorizer = TfidfVectorizer().fit([resume, job_desc])
    tfidf = vectorizer.transform([resume, job_desc])
    overlap_matrix = (tfidf @ tfidf.T).toarray()  # @ is matrix multiplication
    overlap = overlap_matrix[0,1] * 100
    return round(overlap, 2)

# Combined scoring
def calculate_match(resume_text, job_desc_text):
    # Preprocess
    processed_resume = preprocess(resume_text)
    processed_job_desc = preprocess(job_desc_text)

    # Semantic similarity (70% weight)
    resume_embedding = embedding_model.encode(processed_resume)
    job_embedding = embedding_model.encode(processed_job_desc)
    semantic_score = cosine_similarity([resume_embedding], [job_embedding])[0][0] * 100

    # Keyword score (30% weight)
    kw_score = keyword_score(processed_resume, processed_job_desc)

    # Combined final score
    final_score = 0.7 * semantic_score + 0.3 * kw_score
    return min(round(final_score, 2), 100)  # Cap at 100%

# Gradio UI
with gr.Blocks(title="Optimized Resume-Job Matcher") as demo:
    gr.Markdown("## 🔍 Resume-Job Matcher (Hybrid AI + Keywords)")
    with gr.Row():
        resume_input = gr.TextArea(label="Paste Resume", placeholder="Skills: Python, NLP, Hugging Face...")
        job_desc_input = gr.TextArea(label="Paste Job Description", placeholder="Requirements: ML, Transformers...")

    submit_btn = gr.Button("Calculate Match Score", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Results")
            match_score = gr.Number(label="Match Score (%)")
        with gr.Column():
            gr.Markdown("### Debug Info")
            semantic_debug = gr.Textbox(label="Semantic Score")
            keyword_debug = gr.Textbox(label="Keyword Score")

    # Extended examples
    gr.Examples(
        examples=[
            ["Python developer with Hugging Face and NLP experience.", "Seeking ML engineer with transformer model expertise."],
            ["Data scientist skilled in pandas and sklearn.", "Want Python developer with machine learning knowledge."]
        ],
        inputs=[resume_input, job_desc_input]
    )

    def calculate_with_debug(resume, job_desc):
        processed_resume = preprocess(resume)
        processed_job_desc = preprocess(job_desc)

        # Get embeddings
        resume_embedding = embedding_model.encode(processed_resume)
        job_embedding = embedding_model.encode(processed_job_desc)

        # Calculate scores
        semantic = cosine_similarity([resume_embedding], [job_embedding])[0][0] * 100
        keywords = keyword_score(processed_resume, processed_job_desc)
        final = 0.7 * semantic + 0.3 * keywords

        return {
            match_score: min(round(final, 2), 100),
            semantic_debug: f"{round(semantic, 2)}% (AI)",
            keyword_debug: f"{round(keywords, 2)}% (Keywords)"
        }

    submit_btn.click(
        fn=calculate_with_debug,
        inputs=[resume_input, job_desc_input],
        outputs=[match_score, semantic_debug, keyword_debug]
    )

demo.launch(
    share = True,
    debug=True
)
