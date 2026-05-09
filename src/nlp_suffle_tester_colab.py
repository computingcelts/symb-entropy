# ============================================================================
# COMPLETE SHUFFLE TEST SUITE - ALL 7 METHODS WITH 3-WAY COMPARISON
# ============================================================================
# CORRECTED VERSION: Models trained ONCE on original, applied to all conditions
# 
# This code runs all methods comparing:
#   1. ORIGINAL text
#   2. WORD-SHUFFLED text (destroys all structure)
#   3. SENTENCE-SHUFFLED text (preserves local, destroys discourse structure)
#
# METHODOLOGICAL FIX: LDA, BERTopic, and TF-IDF now train on original only,
# then apply the SAME model to all conditions for valid comparison.
# ============================================================================

print("="*80)
print("  SYMBOLIC ENTROPY VALIDATION: 3-WAY SHUFFLE TEST SUITE")
print("  (CORRECTED: Single model trained on original, applied to all)")
print("="*80)
print("\nThis will test 7 different NLP methods on:")
print("  • Original text")
print("  • Word-shuffled text (complete structure destruction)")
print("  • Sentence-shuffled text (local coherence preserved)")
print("\nExpected runtime: ~20-30 minutes total")
print("="*80)

# ============================================================================
# INSTALL ALL LIBRARIES
# ============================================================================

print("\n📦 Installing required libraries...")
!pip install transformers torch python-docx bert-score sentence-transformers -q
!pip install scikit-learn spacy gensim bertopic umap-learn hdbscan -q
!python -m spacy download en_core_web_sm -q

print("✓ All libraries installed")

# ============================================================================
# IMPORTS
# ============================================================================

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from gensim import corpora
from gensim.models import LdaModel
from bertopic import BERTopic
from google.colab import files
from docx import Document
import io
import numpy as np
import re
from collections import Counter

print("✓ All imports successful")

# ============================================================================
# FILE UPLOAD - NOW 3 FILES
# ============================================================================

print("\n" + "="*80)
print("📤 PLEASE UPLOAD YOUR 3 FILES")
print("="*80)

print("\n📤 Upload ORIGINAL file (.docx or .txt):")
uploaded_original = files.upload()

print("\n📤 Upload WORD-SHUFFLED file (.docx or .txt):")
uploaded_word_shuffled = files.upload()

print("\n📤 Upload SENTENCE-SHUFFLED file (.docx or .txt):")
uploaded_sent_shuffled = files.upload()

# ============================================================================
# FILE READING FUNCTION
# ============================================================================

def read_file(file_dict):
    """Read text from uploaded file (handles .docx and .txt with any encoding)"""
    filename = list(file_dict.keys())[0]
    content = file_dict[filename]

    if filename.endswith('.docx'):
        doc = Document(io.BytesIO(content))
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    else:
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                return content.decode(encoding)
            except (UnicodeDecodeError, AttributeError):
                continue
        return content.decode('utf-8', errors='ignore')

# Read files
print("\n📖 Reading uploaded files...")
original_text = read_file(uploaded_original)
word_shuffled_text = read_file(uploaded_word_shuffled)
sent_shuffled_text = read_file(uploaded_sent_shuffled)

print(f"✓ Original text:         {len(original_text)} characters")
print(f"✓ Word-shuffled text:    {len(word_shuffled_text)} characters")
print(f"✓ Sentence-shuffled text: {len(sent_shuffled_text)} characters")

# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def calculate_cohens_d(scores_a, scores_b):
    """Calculate Cohen's d with proper pooled standard deviation"""
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    std_a = np.std(scores_a, ddof=1)
    std_b = np.std(scores_b, ddof=1)

    pooled_sd = np.sqrt((std_a**2 + std_b**2) / 2)

    if pooled_sd > 0:
        cohens_d = abs(mean_a - mean_b) / pooled_sd
    else:
        cohens_d = 0.0

    return cohens_d, mean_a, mean_b, std_a, std_b

def split_into_windows(text, window_size=200, overlap=0.5):
    """Split text into overlapping windows of tokens"""
    words = text.split()
    step_size = int(window_size * (1 - overlap))
    windows = []

    for i in range(0, len(words) - window_size + 1, step_size):
        window = ' '.join(words[i:i+window_size])
        windows.append(window)

    return windows

def split_sentences(text):
    """Split text into sentences"""
    sentences = []
    for sent in text.replace('\n', ' ').split('.'):
        sent = sent.strip()
        if len(sent) > 20:
            sentences.append(sent)
    return sentences

# Store all results - now with 3-way comparison
results = {}

# ============================================================================
# METHOD 1: PERPLEXITY (GPT-2) - SLIDING WINDOW VERSION
# ============================================================================
# STATUS: ✅ VALID - Uses pre-trained model, no fitting on test data

print("\n" + "="*80)
print("METHOD 1/7: GPT-2 Perplexity (Sliding Window)")
print("STATUS: ✅ Pre-trained model - valid comparison")
print("="*80)

print("Loading GPT-2 model...")
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
print("✓ Model loaded")

def calculate_window_perplexity(text, window_size=200):
    """Calculate perplexity for each window"""
    windows = split_into_windows(text, window_size=window_size)
    perplexities = []

    print(f"   Processing {len(windows)} windows...")

    for i, window in enumerate(windows):
        try:
            encodings = gpt2_tokenizer(window, return_tensors='pt', truncation=True, max_length=200)
            with torch.no_grad():
                outputs = gpt2_model(**encodings, labels=encodings.input_ids)
                perplexity = torch.exp(outputs.loss).item()
                perplexities.append(perplexity)
        except:
            continue

        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{len(windows)} windows...")

    return np.array(perplexities)

print("\n⚙️  Calculating perplexity for ORIGINAL text...")
original_ppls = calculate_window_perplexity(original_text)

print("\n⚙️  Calculating perplexity for WORD-SHUFFLED text...")
word_shuf_ppls = calculate_window_perplexity(word_shuffled_text)

print("\n⚙️  Calculating perplexity for SENTENCE-SHUFFLED text...")
sent_shuf_ppls = calculate_window_perplexity(sent_shuffled_text)

d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_ppls, word_shuf_ppls)
d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_ppls, sent_shuf_ppls)

results['Perplexity'] = {
    'd_vs_word': d_word,
    'd_vs_sent': d_sent,
    'original_mean': orig_mean,
    'word_shuf_mean': word_mean,
    'sent_shuf_mean': sent_mean,
    'original_std': orig_std,
    'word_shuf_std': word_std,
    'sent_shuf_std': sent_std,
    'n_observations': len(original_ppls)
}

print(f"\n✓ Perplexity complete:")
print(f"   d(orig vs word-shuf) = {d_word:.2f}")
print(f"   d(orig vs sent-shuf) = {d_sent:.2f}")
print(f"   (n={len(original_ppls)} windows)")

# Clean up memory
del gpt2_model, gpt2_tokenizer
torch.cuda.empty_cache()

# ============================================================================
# METHOD 2: SENTIMENT ANALYSIS
# ============================================================================
# STATUS: ✅ VALID - Uses pre-trained model, no fitting on test data

print("\n" + "="*80)
print("METHOD 2/7: Sentiment Analysis")
print("STATUS: ✅ Pre-trained model - valid comparison")
print("="*80)

print("Loading sentiment model...")
sentiment_analyzer = pipeline("sentiment-analysis",
                              model="distilbert-base-uncased-finetuned-sst-2-english",
                              device=0 if torch.cuda.is_available() else -1)
print("✓ Model loaded")

def analyze_sentiment_chunks(text):
    """Analyze sentiment of text chunks"""
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s.strip()) > 10]

    chunks = []
    for sentence in sentences:
        if len(sentence) > 400:
            words = sentence.split()
            for i in range(0, len(words), 100):
                chunk = ' '.join(words[i:i+100])
                if len(chunk) > 10:
                    chunks.append(chunk)
        else:
            chunks.append(sentence)

    scores = []
    print(f"   Analyzing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks[:50]):
        try:
            result = sentiment_analyzer(chunk)[0]
            score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
            scores.append(score)
        except:
            continue

        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{min(len(chunks), 50)} chunks...")

    return np.array(scores)

print("\n⚙️  Analyzing ORIGINAL text...")
original_sent = analyze_sentiment_chunks(original_text)

print("\n⚙️  Analyzing WORD-SHUFFLED text...")
word_shuf_sent = analyze_sentiment_chunks(word_shuffled_text)

print("\n⚙️  Analyzing SENTENCE-SHUFFLED text...")
sent_shuf_sent = analyze_sentiment_chunks(sent_shuffled_text)

d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_sent, word_shuf_sent)
d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_sent, sent_shuf_sent)

results['Sentiment'] = {
    'd_vs_word': d_word,
    'd_vs_sent': d_sent,
    'original_mean': orig_mean,
    'word_shuf_mean': word_mean,
    'sent_shuf_mean': sent_mean,
    'original_std': orig_std,
    'word_shuf_std': word_std,
    'sent_shuf_std': sent_std,
    'n_observations': len(original_sent)
}

print(f"\n✓ Sentiment complete:")
print(f"   d(orig vs word-shuf) = {d_word:.2f}")
print(f"   d(orig vs sent-shuf) = {d_sent:.2f}")

del sentiment_analyzer
torch.cuda.empty_cache()

# ============================================================================
# METHOD 3: TF-IDF (CORRECTED)
# ============================================================================
# STATUS: 🔧 CORRECTED - Now fits vectorizer on ORIGINAL only, transforms all

print("\n" + "="*80)
print("METHOD 3/7: TF-IDF Coherence")
print("STATUS: 🔧 CORRECTED - Vectorizer fit on original only")
print("="*80)

def calculate_tfidf_coherence_corrected(original_text, word_shuf_text, sent_shuf_text):
    """
    Calculate TF-IDF similarity between consecutive sentences.
    CORRECTED: Fit vectorizer on ORIGINAL sentences only, then transform all conditions.
    """
    orig_sentences = split_sentences(original_text)
    word_sentences = split_sentences(word_shuf_text)
    sent_sentences = split_sentences(sent_shuf_text)

    if len(orig_sentences) < 2:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    # FIT ONCE on original text only
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    try:
        vectorizer.fit(orig_sentences)
    except:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    def get_consecutive_similarities(sentences, vectorizer):
        """Transform with pre-fit vectorizer and calculate consecutive similarities"""
        try:
            tfidf_matrix = vectorizer.transform(sentences)
        except:
            return np.array([0.0])

        similarities = []
        for i in range(tfidf_matrix.shape[0] - 1):
            sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0][0]
            similarities.append(sim)
        return np.array(similarities)

    print("   Transforming original sentences...")
    orig_sims = get_consecutive_similarities(orig_sentences, vectorizer)
    
    print("   Transforming word-shuffled sentences...")
    word_sims = get_consecutive_similarities(word_sentences, vectorizer)
    
    print("   Transforming sentence-shuffled sentences...")
    sent_sims = get_consecutive_similarities(sent_sentences, vectorizer)

    return orig_sims, word_sims, sent_sims

print("\n⚙️  Calculating TF-IDF coherence (single vectorizer)...")
original_tfidf, word_shuf_tfidf, sent_shuf_tfidf = calculate_tfidf_coherence_corrected(
    original_text, word_shuffled_text, sent_shuffled_text
)

d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_tfidf, word_shuf_tfidf)
d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_tfidf, sent_shuf_tfidf)

results['TF-IDF'] = {
    'd_vs_word': d_word,
    'd_vs_sent': d_sent,
    'original_mean': orig_mean,
    'word_shuf_mean': word_mean,
    'sent_shuf_mean': sent_mean,
    'original_std': orig_std,
    'word_shuf_std': word_std,
    'sent_shuf_std': sent_std,
    'n_observations': len(original_tfidf)
}

print(f"✓ TF-IDF complete:")
print(f"   d(orig vs word-shuf) = {d_word:.2f}")
print(f"   d(orig vs sent-shuf) = {d_sent:.2f}")

# ============================================================================
# METHOD 4: NER - SENTENCE-LEVEL VERSION
# ============================================================================
# STATUS: ✅ VALID - Uses pre-trained model, no fitting on test data

print("\n" + "="*80)
print("METHOD 4/7: Named Entity Recognition (Sentence-Level)")
print("STATUS: ✅ Pre-trained model - valid comparison")
print("="*80)

print("Loading spaCy NER model...")
nlp = spacy.load("en_core_web_sm")
print("✓ Model loaded")

def calculate_sentence_ner_density(text):
    """Calculate entity density for each sentence"""
    sentences = split_sentences(text)
    densities = []

    print(f"   Analyzing {len(sentences)} sentences...")

    for i, sent in enumerate(sentences):
        doc = nlp(sent)
        entities = [ent for ent in doc.ents]
        word_count = len(sent.split())
        density = (len(entities) / word_count) * 100 if word_count > 0 else 0
        densities.append(density)

        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{len(sentences)} sentences...")

    return np.array(densities)

print("\n⚙️  Analyzing ORIGINAL text...")
original_ner = calculate_sentence_ner_density(original_text)

print("\n⚙️  Analyzing WORD-SHUFFLED text...")
word_shuf_ner = calculate_sentence_ner_density(word_shuffled_text)

print("\n⚙️  Analyzing SENTENCE-SHUFFLED text...")
sent_shuf_ner = calculate_sentence_ner_density(sent_shuffled_text)

d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_ner, word_shuf_ner)
d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_ner, sent_shuf_ner)

results['NER'] = {
    'd_vs_word': d_word,
    'd_vs_sent': d_sent,
    'original_mean': orig_mean,
    'word_shuf_mean': word_mean,
    'sent_shuf_mean': sent_mean,
    'original_std': orig_std,
    'word_shuf_std': word_std,
    'sent_shuf_std': sent_std,
    'n_observations': len(original_ner)
}

print(f"✓ NER complete:")
print(f"   d(orig vs word-shuf) = {d_word:.2f}")
print(f"   d(orig vs sent-shuf) = {d_sent:.2f}")

# ============================================================================
# METHOD 5: LDA - DOCUMENT-LEVEL TOPIC PROBABILITIES (CORRECTED)
# ============================================================================
# STATUS: 🔧 CORRECTED - Now trains on ORIGINAL only, infers on all conditions

print("\n" + "="*80)
print("METHOD 5/7: LDA Topic Modeling (Document-Level Probabilities)")
print("STATUS: 🔧 CORRECTED - Model trained on original only")
print("="*80)

def preprocess_text(text):
    """Preprocess text for LDA"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()

    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'is', 'was', 'are', 'were', 'be', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'her', 'its', 'our', 'their', 'what', 'which', 'who', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'as', 'by', 'from', 'with'
    ])

    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens

def calculate_lda_corrected(original_text, word_shuf_text, sent_shuf_text, num_topics=5):
    """
    Calculate topic probabilities using LDA.
    CORRECTED: Train on ORIGINAL only, then infer on all conditions.
    """
    orig_sentences = split_sentences(original_text)
    word_sentences = split_sentences(word_shuf_text)
    sent_sentences = split_sentences(sent_shuf_text)

    # Preprocess all
    orig_docs = [preprocess_text(sent) for sent in orig_sentences]
    orig_docs = [doc for doc in orig_docs if len(doc) > 5]

    word_docs = [preprocess_text(sent) for sent in word_sentences]
    word_docs = [doc for doc in word_docs if len(doc) > 5]

    sent_docs = [preprocess_text(sent) for sent in sent_sentences]
    sent_docs = [doc for doc in sent_docs if len(doc) > 5]

    if len(orig_docs) < 10:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    # BUILD DICTIONARY FROM ORIGINAL ONLY
    print("   Building dictionary from original text...")
    dictionary = corpora.Dictionary(orig_docs)
    
    # Create corpora using the SAME dictionary
    orig_corpus = [dictionary.doc2bow(doc) for doc in orig_docs]
    word_corpus = [dictionary.doc2bow(doc) for doc in word_docs]
    sent_corpus = [dictionary.doc2bow(doc) for doc in sent_docs]

    # TRAIN LDA ON ORIGINAL ONLY
    print("   Training LDA on original text...")
    lda = LdaModel(
        corpus=orig_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        per_word_topics=True
    )

    def get_topic_probabilities(corpus, lda_model):
        """Get max topic probability for each document using pre-trained model"""
        probabilities = []
        for doc_bow in corpus:
            doc_topics = lda_model.get_document_topics(doc_bow)
            max_prob = max([prob for _, prob in doc_topics]) if doc_topics else 0
            probabilities.append(max_prob)
        return np.array(probabilities)

    print("   Inferring topics on original...")
    orig_probs = get_topic_probabilities(orig_corpus, lda)
    
    print("   Inferring topics on word-shuffled...")
    word_probs = get_topic_probabilities(word_corpus, lda)
    
    print("   Inferring topics on sentence-shuffled...")
    sent_probs = get_topic_probabilities(sent_corpus, lda)

    return orig_probs, word_probs, sent_probs

print("\n⚙️  Running LDA (single model)...")
original_lda, word_shuf_lda, sent_shuf_lda = calculate_lda_corrected(
    original_text, word_shuffled_text, sent_shuffled_text
)

d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_lda, word_shuf_lda)
d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_lda, sent_shuf_lda)

results['LDA'] = {
    'd_vs_word': d_word,
    'd_vs_sent': d_sent,
    'original_mean': orig_mean,
    'word_shuf_mean': word_mean,
    'sent_shuf_mean': sent_mean,
    'original_std': orig_std,
    'word_shuf_std': word_std,
    'sent_shuf_std': sent_std,
    'n_observations': len(original_lda)
}

print(f"✓ LDA complete:")
print(f"   d(orig vs word-shuf) = {d_word:.2f}")
print(f"   d(orig vs sent-shuf) = {d_sent:.2f}")

# ============================================================================
# METHOD 6: BERTSCORE - SEQUENTIAL COHERENCE (CORRECTED)
# ============================================================================
# STATUS: 🔧 CORRECTED - Uses fixed-size windows instead of period-based splitting
# 
# ISSUE WITH ORIGINAL: Splitting by periods in word-shuffled text creates
# arbitrary chunks (periods land randomly among words), making comparison invalid.
#
# FIX: Use fixed-size token windows. This ensures we're comparing the same
# positional structure across all conditions.

print("\n" + "="*80)
print("METHOD 6/7: BERTScore (Sequential Coherence)")
print("STATUS: 🔧 CORRECTED - Fixed-size windows for valid comparison")
print("="*80)

def calculate_bertscore_window_coherence(text, window_size=50):
    """
    Calculate BERTScore between consecutive FIXED-SIZE windows.
    
    CORRECTED: Uses token-based windows instead of period-based sentences.
    This ensures valid comparison across conditions regardless of punctuation.
    """
    words = text.split()
    
    # Create fixed-size windows (no overlap for consecutive comparison)
    windows = []
    for i in range(0, len(words) - window_size + 1, window_size):
        window = ' '.join(words[i:i+window_size])
        windows.append(window)
    
    if len(windows) < 2:
        return np.array([0.0])

    scores = []
    print(f"   Comparing {len(windows)-1} consecutive window pairs ({window_size} tokens each)...")

    for i in range(len(windows) - 1):
        try:
            P, R, F1 = bertscore([windows[i]], [windows[i+1]],
                                lang='en', model_type='bert-base-uncased',
                                verbose=False)
            scores.append(F1.item())
        except:
            continue

        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{len(windows)-1} pairs...")

    return np.array(scores)

print("\n⚙️  Calculating sequential coherence for ORIGINAL...")
original_bert = calculate_bertscore_window_coherence(original_text)

print("\n⚙️  Calculating sequential coherence for WORD-SHUFFLED...")
word_shuf_bert = calculate_bertscore_window_coherence(word_shuffled_text)

print("\n⚙️  Calculating sequential coherence for SENTENCE-SHUFFLED...")
sent_shuf_bert = calculate_bertscore_window_coherence(sent_shuffled_text)

d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_bert, word_shuf_bert)
d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_bert, sent_shuf_bert)

results['BERTScore'] = {
    'd_vs_word': d_word,
    'd_vs_sent': d_sent,
    'original_mean': orig_mean,
    'word_shuf_mean': word_mean,
    'sent_shuf_mean': sent_mean,
    'original_std': orig_std,
    'word_shuf_std': word_std,
    'sent_shuf_std': sent_std,
    'n_observations': len(original_bert)
}

print(f"✓ BERTScore complete:")
print(f"   d(orig vs word-shuf) = {d_word:.2f}")
print(f"   d(orig vs sent-shuf) = {d_sent:.2f}")

# ============================================================================
# METHOD 7: BERTOPIC (CORRECTED)
# ============================================================================
# STATUS: 🔧 CORRECTED - Now trains on ORIGINAL only, transforms all conditions

print("\n" + "="*80)
print("METHOD 7/7: BERTopic")
print("STATUS: 🔧 CORRECTED - Model trained on original only")
print("="*80)

def calculate_bertopic_corrected(original_text, word_shuf_text, sent_shuf_text):
    """
    Calculate topic assignment probabilities using BERTopic.
    CORRECTED: Train on ORIGINAL only, then transform all conditions.
    """
    orig_docs = split_sentences(original_text)
    word_docs = split_sentences(word_shuf_text)
    sent_docs = split_sentences(sent_shuf_text)

    if len(orig_docs) < 10:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    # TRAIN ON ORIGINAL ONLY
    print("   Training BERTopic on original text...")
    bertopic_model = BERTopic(
        language="english",
        calculate_probabilities=True,
        verbose=False,
        min_topic_size=3,
        nr_topics="auto"
    )

    # fit_transform on original
    orig_topics, orig_probs = bertopic_model.fit_transform(orig_docs)
    orig_max_probs = np.max(orig_probs, axis=1)

    # transform (NOT fit_transform) on shuffled conditions
    print("   Transforming word-shuffled text...")
    word_topics, word_probs = bertopic_model.transform(word_docs)
    word_max_probs = np.max(word_probs, axis=1)

    print("   Transforming sentence-shuffled text...")
    sent_topics, sent_probs = bertopic_model.transform(sent_docs)
    sent_max_probs = np.max(sent_probs, axis=1)

    return orig_max_probs, word_max_probs, sent_max_probs

print("\n⚙️  Running BERTopic (single model)...")
original_bertopic, word_shuf_bertopic, sent_shuf_bertopic = calculate_bertopic_corrected(
    original_text, word_shuffled_text, sent_shuffled_text
)

d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_bertopic, word_shuf_bertopic)
d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_bertopic, sent_shuf_bertopic)

results['BERTopic'] = {
    'd_vs_word': d_word,
    'd_vs_sent': d_sent,
    'original_mean': orig_mean,
    'word_shuf_mean': word_mean,
    'sent_shuf_mean': sent_mean,
    'original_std': orig_std,
    'word_shuf_std': word_std,
    'sent_shuf_std': sent_std,
    'n_observations': len(original_bertopic)
}

print(f"✓ BERTopic complete:")
print(f"   d(orig vs word-shuf) = {d_word:.2f}")
print(f"   d(orig vs sent-shuf) = {d_sent:.2f}")

# ============================================================================
# FINAL RESULTS TABLE - 3-WAY COMPARISON
# ============================================================================

print("\n" + "="*80)
print("  FINAL RESULTS - 3-WAY SHUFFLE COMPARISON")
print("  (CORRECTED METHODOLOGY)")
print("="*80)

print("\n" + "="*100)
print("SUMMARY TABLE: Cohen's d Effect Sizes")
print("="*100)
print(f"\n{'Method':<15} {'d(Word-Shuf)':>12} {'Verdict':>12} {'d(Sent-Shuf)':>12} {'Verdict':>12} {'n':>8}")
print("-" * 75)

method_order = ['Perplexity', 'Sentiment', 'TF-IDF', 'NER', 'LDA', 'BERTScore', 'BERTopic']

def get_verdict(d):
    if d >= 3.0:
        return "✅✅ STRONG"
    elif d >= 2.0:
        return "✅ PASS"
    elif d >= 1.0:
        return "~ BORDER"
    else:
        return "❌ FAIL"

for method_name in method_order:
    r = results[method_name]
    d_w = r['d_vs_word']
    d_s = r['d_vs_sent']
    n = r['n_observations']
    
    print(f"{method_name:<15} {d_w:>12.2f} {get_verdict(d_w):>12} {d_s:>12.2f} {get_verdict(d_s):>12} {n:>8}")

print("="*100)

print("\n" + "="*100)
print("MEAN VALUES BY CONDITION")
print("="*100)
print(f"\n{'Method':<15} {'Original':>12} {'Word-Shuf':>12} {'Sent-Shuf':>12}")
print("-" * 55)

for method_name in method_order:
    r = results[method_name]
    print(f"{method_name:<15} {r['original_mean']:>12.3f} {r['word_shuf_mean']:>12.3f} {r['sent_shuf_mean']:>12.3f}")

print("="*100)

print("\n" + "="*80)
print("DETAILED STATISTICS")
print("="*80)

for method_name in method_order:
    r = results[method_name]
    print(f"\n{method_name}:")
    print(f"  d(orig vs word-shuffled):     {r['d_vs_word']:.3f} - {get_verdict(r['d_vs_word'])}")
    print(f"  d(orig vs sentence-shuffled): {r['d_vs_sent']:.3f} - {get_verdict(r['d_vs_sent'])}")
    print(f"  Original:       {r['original_mean']:.3f} (SD: {r['original_std']:.3f})")
    print(f"  Word-shuffled:  {r['word_shuf_mean']:.3f} (SD: {r['word_shuf_std']:.3f})")
    print(f"  Sent-shuffled:  {r['sent_shuf_mean']:.3f} (SD: {r['sent_shuf_std']:.3f})")
    print(f"  N observations: {r['n_observations']}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Categorize by word-shuffle sensitivity
strong_word = [m for m in results.keys() if results[m]['d_vs_word'] >= 3.0]
pass_word = [m for m in results.keys() if 2.0 <= results[m]['d_vs_word'] < 3.0]
border_word = [m for m in results.keys() if 1.0 <= results[m]['d_vs_word'] < 2.0]
fail_word = [m for m in results.keys() if results[m]['d_vs_word'] < 1.0]

# Categorize by sentence-shuffle sensitivity  
strong_sent = [m for m in results.keys() if results[m]['d_vs_sent'] >= 3.0]
pass_sent = [m for m in results.keys() if 2.0 <= results[m]['d_vs_sent'] < 3.0]
border_sent = [m for m in results.keys() if 1.0 <= results[m]['d_vs_sent'] < 2.0]
fail_sent = [m for m in results.keys() if results[m]['d_vs_sent'] < 1.0]

print(f"\n📊 WORD-SHUFFLE SENSITIVITY (total structure destruction):")
print(f"   ✅✅ STRONG PASS (d ≥ 3.0): {len(strong_word)} methods")
for m in strong_word:
    print(f"      • {m} (d = {results[m]['d_vs_word']:.2f})")
print(f"   ✅ PASS (2.0 ≤ d < 3.0): {len(pass_word)} methods")
for m in pass_word:
    print(f"      • {m} (d = {results[m]['d_vs_word']:.2f})")
print(f"   ~ BORDERLINE (1.0 ≤ d < 2.0): {len(border_word)} methods")
for m in border_word:
    print(f"      • {m} (d = {results[m]['d_vs_word']:.2f})")
print(f"   ❌ FAILING (d < 1.0): {len(fail_word)} methods")
for m in fail_word:
    print(f"      • {m} (d = {results[m]['d_vs_word']:.2f})")

print(f"\n📊 SENTENCE-SHUFFLE SENSITIVITY (discourse structure only):")
print(f"   ✅✅ STRONG PASS (d ≥ 3.0): {len(strong_sent)} methods")
for m in strong_sent:
    print(f"      • {m} (d = {results[m]['d_vs_sent']:.2f})")
print(f"   ✅ PASS (2.0 ≤ d < 3.0): {len(pass_sent)} methods")
for m in pass_sent:
    print(f"      • {m} (d = {results[m]['d_vs_sent']:.2f})")
print(f"   ~ BORDERLINE (1.0 ≤ d < 2.0): {len(border_sent)} methods")
for m in border_sent:
    print(f"      • {m} (d = {results[m]['d_vs_sent']:.2f})")
print(f"   ❌ FAILING (d < 1.0): {len(fail_sent)} methods")
for m in fail_sent:
    print(f"      • {m} (d = {results[m]['d_vs_sent']:.2f})")

# Identify methods sensitive to discourse but not just local coherence
discourse_sensitive = [m for m in results.keys() 
                       if results[m]['d_vs_sent'] >= 1.0 and results[m]['d_vs_word'] > results[m]['d_vs_sent']]

print(f"\n🎯 DISCOURSE-LEVEL SENSITIVE METHODS:")
print("   (Methods that detect both total destruction AND discourse-only disruption)")
for m in discourse_sensitive:
    ratio = results[m]['d_vs_word'] / results[m]['d_vs_sent'] if results[m]['d_vs_sent'] > 0 else 0
    print(f"   • {m}: word d={results[m]['d_vs_word']:.2f}, sent d={results[m]['d_vs_sent']:.2f}, ratio={ratio:.1f}x")

if not discourse_sensitive:
    print("   None of the tested methods show significant discourse-level sensitivity.")
    print("   This is the gap that Symbolic Entropy's Σ component is designed to fill.")

print("\n" + "="*80)
print("METHODOLOGICAL NOTE")
print("="*80)
print("CORRECTED METHODOLOGY:")
print("  • TF-IDF: Vectorizer fit on ORIGINAL, transform applied to all conditions")
print("  • LDA: Dictionary + model trained on ORIGINAL, inference on all conditions")
print("  • BERTopic: fit_transform on ORIGINAL, transform() on shuffled conditions")
print("  • BERTScore: Fixed-size token windows (not period-based sentence splitting)")
print("      → Ensures valid comparison when periods are randomly distributed")
print("  • Perplexity, Sentiment, NER: Pre-trained models (unchanged)")
print("")
print("This 3-way comparison tests semantic sensitivity at two levels:")
print("  • WORD-SHUFFLE: Destroys ALL structure (local + discourse)")
print("  • SENTENCE-SHUFFLE: Preserves local coherence, destroys discourse order")
print("\nMethods sensitive to sentence-shuffle detect discourse-level organization")
print("beyond just local word patterns - this is what SE's Σ component measures.")
print("\nAll Cohen's d values calculated using proper pooled standard deviations")
print("from multiple observations per condition.")
print("="*80)

print("\n✅ 3-WAY SHUFFLE TEST SUITE COMPLETE (CORRECTED METHODOLOGY)!")
