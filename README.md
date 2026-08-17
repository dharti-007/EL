import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONTENT FILTER MODULE
# ==========================================
# Inappropriate / Explicit words list
BAD_WORDS_LIST = {"abuse", "badword", "hate", "scam", "spam", "violence"}


def preprocess_text(text):
    """Text normalization: Lowercase aur special characters remove karna."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def filter_inappropriate_content(text):
    """Explicit content check karta hai aur flagged words return karta hai."""
    clean_text = preprocess_text(text)
    words = set(clean_text.split())

    # Find intersection of words and bad words
    flagged_words = list(words.intersection(BAD_WORDS_LIST))
    is_clean = len(flagged_words) == 0

    return is_clean, flagged_words


# ==========================================
# 2. PLAGIARISM DETECTOR MODULE
# ==========================================
def calculate_plagiarism_score(target_text, reference_text):
    """TF-IDF Vectorization & Cosine Similarity se similarity score nikalta hai."""
    # Preprocess both texts
    clean_target = preprocess_text(target_text)
    clean_ref = preprocess_text(reference_text)

    # Convert text into TF-IDF Vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([clean_target, clean_ref])

    # Calculate Cosine Similarity Score
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score_percentage = round(similarity[0][0] * 100, 2)

    return score_percentage


def highlight_common_phrases(target_text, reference_text):
    """Common overlapping words ko identify karta hai."""
    target_words = set(preprocess_text(target_text).split())
    ref_words = set(preprocess_text(reference_text).split())

    common_words = list(target_words.intersection(ref_words))
    return common_words


# ==========================================
# 3. REPORT GENERATOR (MAIN PIPELINE)
# ==========================================
def generate_report(target_text, reference_text):
    print("=" * 50)
    print(" AUTOMATED CONTENT FILTER & PLAGIARISM REPORT")
    print("=" * 50)

    # Step 1: Content Filter Check
    is_clean, flagged_words = filter_inappropriate_content(target_text)

    print("\n[1] CONTENT SAFETY CHECK:")
    if is_clean:
        print(" Status: APPROVED (No inappropriate content detected)")
    else:
        print(" Status: REJECTED / FLAGGED")
        print(f" Inappropriate Words Found: {', '.join(flagged_words)}")

    # Step 2: Plagiarism Detection Check
    similarity_score = calculate_plagiarism_score(target_text, reference_text)
    common_words = highlight_common_phrases(target_text, reference_text)

    print("\n[2] PLAGIARISM ASSESSMENT:")
    print(f" Plagiarism Score: {similarity_score}%")
    print(f" Common Tokens Highlighted: {', '.join(common_words)}")

    # Verdict
    print("\n[3] FINAL VERDICT:")
    if similarity_score > 70:
        print(" Verdict: High Plagiarism Detected!")
    elif similarity_score > 30:
        print(" Verdict: Moderate Similarity (Review Required).")
    else:
        print(" Verdict: Content is Original.")

    print("=" * 50)


# ==========================================
# EXECUTION / DRIVER CODE
# ==========================================
if __name__ == "__main__":
    # Test Input Documents
    sample_target_doc = """
    Python is an amazing language for machine learning and artificial intelligence. 
    It contains hate speech and scam content to test the filter.
    """

    sample_reference_doc = """
    Python is a popular language for machine learning, artificial intelligence, 
    and data science applications.
    """

    # Run the Pipeline
    generate_report(sample_target_doc, sample_reference_doc)
