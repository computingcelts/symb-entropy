# ============================================================================
# COMPLETE SHUFFLE TEST SUITE - UP TO 7 METHODS WITH 3-WAY COMPARISON
# ============================================================================
# CORRECTED VERSION: Models trained ONCE on original, applied to all conditions
# 
# This code runs all methods comparing:
#   1. ORIGINAL text
#   2. WORD-SHUFFLED text (destroys all structure)
#   3. SENTENCE-SHUFFLED text (preserves local, destroys discourse structure)
#
# ============================================================================

print("="*80)
print("  SYMBOLIC ENTROPY VALIDATION: 3-WAY SHUFFLE TEST SUITE")
print("  (Single model trained on original, applied to all)")
print("="*80)
print("\nThis will test 7 different NLP methods on:")
print("  • Original text")
print("  • Word-shuffled text (complete structure destruction)")
print("  • Sentence-shuffled text (local coherence preserved)")
print("\nExpected runtime: ~20-30 minutes total but it depends on the corpus size")
print("="*80)

# ============================================================================
# IMPORTS
# ============================================================================

from collections import Counter
import io
import os
import re
from time import time

from bert_score import score as bertscore
from bertopic import BERTopic
from docx import Document
from gensim import corpora
from gensim.models import LdaModel
from inquirer.themes import GreenPassion
import inquirer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline
from umap import UMAP
import torch


# relative path where input texts are located
local_input_path = "./texts"

# change to True those methods to be included in the analysis
list_of_methods = {'Perplexity': True,
                    'Sentiment': True,
                    'TF-IDF': True,
                    'NER': True,
                    'LDA': True,
                    'BERTScore':True,
                    'BERTopic' :True
            }

# counts the number of methods to run
total_methods = len([m for m in list_of_methods.values() if m == True])

class NLPMethod:
    method_name = ''
    completion_time = ''

    results_data = {
        'd_vs_word': 0.0,
        'd_vs_sent': 0.0,
        'original_mean': 0.0,
        'word_shuf_mean': 0.0,
        'sent_shuf_mean': 0.0,
        'original_std': 0.0,
        'word_shuf_std': 0.0,
        'sent_shuf_std': 0.0,
        'n_observations': 0
    }
    
    # override this method with a specific computation
    # expects two texts as input
    def compute_method(self, original_text, comparison_text):
        print('Processing method: ', self.method_name)

    # stores the results of a computation
    def set_results(self, d_word,d_sent,orig_mean,word_mean,sent_mean,orig_std,word_std,sent_std, original_length):
        self.results_data = {
            'd_vs_word': d_word,
            'd_vs_sent': d_sent,
            'original_mean': orig_mean,
            'word_shuf_mean': word_mean,
            'sent_shuf_mean': sent_mean,
            'original_std': orig_std,
            'word_shuf_std': word_std,
            'sent_shuf_std': sent_std,
            'n_observations': original_length
        }

    # prints the results of a computation
    def print_results(self):
        print(f"\n✓ " + self.method_name + " complete:")
        print(f"   d(orig vs word-shuf) = {self.results_data['d_vs_word']:.2f}")
        print(f"   d(orig vs sent-shuf) = {self.results_data['d_vs_sent']:.2f}")
        print(f"   (n={self.results_data['n_observations']} windows)")
        print("   Completion time: ",   self.completion_time)

# class to allow file selection
class SelectFiles():

    def validation_function(self, answers, current):
        print('Current selected value(s): ', current)
        if len(current) == 0:
            raise inquirer.errors.ValidationError(
                '', reason='You have to select at least one option.')
        return True

    def get_txt_doc_files(self, path):
        if not os.path.isdir(path):
            return []
        return sorted([
            filename for filename in os.listdir(path)
            if os.path.isfile(os.path.join(path, filename)) and (filename.endswith('.txt') or filename.endswith('.docx'))
        ])


    def select_files_to_process(self, directory_path):
        files = self.get_txt_doc_files(directory_path)
        if not files:
            print(f"No .txt .docx files found in {directory_path}. Choose a different directory.")
            return []

        choices = ['All'] + files + ['Exit']
        questions = [
            inquirer.Checkbox(
                'selected_files',
                message=f'A total of {len(files)} files found in {directory_path}. Choose one or more:',
                choices=choices,
                validate=self.validation_function
            )
        ]

        answers = inquirer.prompt(questions)
        if not answers or 'selected_files' not in answers:
            return []

        selected = answers['selected_files']
        if 'Exit' in selected:
            exit()
        if 'All' in selected:
            return [os.path.join(directory_path, f) for f in files]
        return [os.path.join(directory_path, f) for f in selected]


# ============================================================================
# FILE READING FUNCTION
# ============================================================================

def read_file(filename):
    """Read text from uploaded file (handles .docx and .txt with any encoding)"""

    with open(filename, "rb") as f:
        content = f.read()

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

def segment_into_units(text, unit_size=50, min_words=10):
    """
    Split text into fixed-size non-overlapping token chunks.

    Args:
        text:      input string
        unit_size: number of tokens per chunk (default 50)
        min_words: discard chunks shorter than this (default 10)

    Returns:
        list of string chunks
    """
    words = text.split()
    segments = []
    for i in range(0, len(words), unit_size):
        seg_words = words[i:i + unit_size]
        if len(seg_words) >= min_words:
            segments.append(' '.join(seg_words))
    
    return segments


def split_sentences(text):
    """Split text into sentences"""
    sentences = []
    for sent in text.replace('\n', ' ').split('.'):
        sent = sent.strip()
        if len(sent) > 20:
            sentences.append(sent)
    return sentences

class ComputePerplexity(NLPMethod):

    def __init__(self):
        self.method_name = 'Perplexity'
        self.completion_time = ''

        print("Loading GPT-2 model...")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_model.eval() 
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        print("✓ Model loaded")

    def calculate_window_perplexity(self, text, window_size=200):
        """Calculate perplexity for each window"""
        windows = split_into_windows(text, window_size=window_size)
        perplexities = []

        print(f"   Processing {len(windows)} windows...")

        for i, window in enumerate(windows):
            try:
                encodings = self.gpt2_tokenizer(window, return_tensors='pt', truncation=True, max_length=200)
                with torch.no_grad():
                    outputs = self.gpt2_model(**encodings, labels=encodings.input_ids)
                    perplexity = torch.exp(outputs.loss).item()
                    perplexities.append(perplexity)
            except:
                continue

            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(windows)} windows...")

        return np.array(perplexities)

    def compute_method(self, original_text, word_shuffled_text, sentence_shuffled_text):

        computation_start = time()

        print("\n⚙️  Calculating " + self.method_name + " for ORIGINAL text...")
        original_analysis = self.calculate_window_perplexity(original_text)

        print("\n⚙️  Calculating " + self.method_name + " for WORD-SHUFFLED text...")
        word_shuffled_analysis = self.calculate_window_perplexity(word_shuffled_text)

        print("\n⚙️  Calculating " + self.method_name + " for SENTENCE-SHUFFLED text...")
        sentence_shuffled_analysis = self.calculate_window_perplexity(sentence_shuffled_text)

        d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_analysis, word_shuffled_analysis)
        d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_analysis, sentence_shuffled_analysis)

        self.completion_time = self.method_name + " took about " + str(round((time()-computation_start),3)) + " secs."

        super().set_results(d_word,d_sent,orig_mean,word_mean,sent_mean,orig_std,word_std,sent_std, len(original_analysis))

        super().print_results()

        # Clean up memory
        del self.gpt2_model, self.gpt2_tokenizer
        torch.cuda.empty_cache()

        

class ComputeSentiment(NLPMethod):

    def __init__(self):
        self.method_name = 'Sentiment'
        print("Loading sentiment model...")

        self.sentiment_analyzer = pipeline("sentiment-analysis",
                                    model="distilbert-base-uncased-finetuned-sst-2-english",
                                    device=0 if torch.cuda.is_available() else -1)
        
        print("✓ Model loaded")   

    def analyze_sentiment_chunks(self, text):
        """Analyze sentiment of text chunks"""
        
        chunks = segment_into_units(text, unit_size=80, min_words=10)
        scores = []
        n_run  = min(len(chunks), 50)

        print(f"   Analyzing {n_run} chunks (of {len(chunks)} total)...")

        for i, chunk in enumerate(chunks[:50]):
            try:
                result = self.sentiment_analyzer(chunk)[0]
                score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
                scores.append(score)
            except:
                continue

            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{n_run} chunks...")

        return np.array(scores)

        
    def compute_method(self, original_text, word_shuffled_text, sentence_shuffled_text):

        computation_start = time()

        print("\n⚙️  Analyzing " + self.method_name + " ORIGINAL text...")
        original_analysis = self.analyze_sentiment_chunks(original_text)

        print("\n⚙️  Analyzing " + self.method_name + " WORD-SHUFFLED text...")
        word_shuffled_analysis = self.analyze_sentiment_chunks(word_shuffled_text)

        print("\n⚙️  Analyzing " + self.method_name + " SENTENCE-SHUFFLED text...")
        sentence_shuffled_analysis = self.analyze_sentiment_chunks(sentence_shuffled_text)

        d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_analysis, word_shuffled_analysis)
        d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_analysis, sentence_shuffled_analysis)

        self.completion_time = self.method_name + " took about " + str(round((time()-computation_start),3)) + " secs."

        super().set_results(d_word,d_sent,orig_mean,word_mean,sent_mean,orig_std,word_std,sent_std, len(original_analysis))

        super().print_results()

        # frees model memory
        del self.sentiment_analyzer
        torch.cuda.empty_cache()

class ComputeTF_IDF(NLPMethod):

    def __init__(self):
        self.method_name = 'TF-IDF'
        
    def calculate_tfidf_coherence_corrected(self, original_text, word_shuffled_text, sentence_shuffled_text):
        """
        Calculate TF-IDF similarity between consecutive sentences.
        CORRECTED: Fit vectorizer on ORIGINAL sentences only, then transform all conditions.
        """
        orig_sentences = segment_into_units(original_text, unit_size=50, min_words=10)
        word_sentences = segment_into_units(word_shuffled_text, unit_size=50, min_words=10)
        sent_sentences = segment_into_units(sentence_shuffled_text, unit_size=50, min_words=10)

        if len(orig_sentences) < 2:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])

        # fit once on original, transform all through shared vocab
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
         
    def compute_method(self, original_text, word_shuffled_text, sentence_shuffled_text):

        computation_start = time()

        print("\n⚙️  Calculating " + self.method_name + " coherence (single vectorizer)...")

        original_analysis, word_shuffled_analysis, sentence_shuffled_analysis = self.calculate_tfidf_coherence_corrected(
            original_text, word_shuffled_text, sentence_shuffled_text
        )

        d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_analysis, word_shuffled_analysis)
        d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_analysis, sentence_shuffled_analysis)

        self.completion_time = self.method_name + " took about " + str(round((time()-computation_start),3)) + " secs."

        super().set_results(d_word,d_sent,orig_mean,word_mean,sent_mean,orig_std,word_std,sent_std, len(original_analysis))

        super().print_results()

class ComputeNER(NLPMethod):
    """
    Entity density (entities per 100 words) per text unit.
    Pre-trained spaCy model — no fitting on test data.
    """

    def __init__(self):
        self.method_name = 'NER'

        print("Loading spaCy NER model...")
        self.nlp = spacy.load("en_core_web_sm")
        print("✓ Model loaded") 

    def calculate_sentence_ner_density(self, text):
        """Calculate entity density for each sentence"""
        sentences = segment_into_units(text, unit_size=50, min_words=10)
        densities = []

        print(f"   Analyzing {len(sentences)} sentences...")

        for i, sent in enumerate(sentences):
            doc = self.nlp(sent)
            word_count = len(sent.split())
            density = (len(doc.ents) / word_count) * 100 if word_count > 0 else 0
            densities.append(density)

            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(sentences)} sentences...")

        return np.array(densities)
    
    def compute_method(self, original_text, word_shuffled_text, sentence_shuffled_text):

        computation_start = time()
        
        print("\n⚙️  Analyzing " + self.method_name + " ORIGINAL text...")
        original_analysis = self.calculate_sentence_ner_density(original_text)

        print("\n⚙️  Analyzing " + self.method_name + " WORD-SHUFFLED text...")
        word_shuffled_analysis = self.calculate_sentence_ner_density(word_shuffled_text)

        print("\n⚙️  Analyzing " + self.method_name + " SENTENCE-SHUFFLED text...")
        sentence_shuffled_analysis = self.calculate_sentence_ner_density(sentence_shuffled_text)

        d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_analysis, word_shuffled_analysis)
        d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_analysis, sentence_shuffled_analysis)

        self.completion_time = self.method_name + " took about " + str(round((time()-computation_start),3)) + " secs."

        super().set_results(d_word,d_sent,orig_mean,word_mean,sent_mean,orig_std,word_std,sent_std, len(original_analysis))

        super().print_results()   


class ComputeLDA(NLPMethod):

    def __init__(self):
        self.method_name = 'LDA'
        print("\n⚙️  Running " + self.method_name + " (single model)...")


    def preprocess_text(self, text):
        """Preprocess text for LDA"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

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

        tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]
        return tokens

    def calculate_lda_corrected(self, original_text, word_shuf_text, sent_shuf_text, num_topics=5):
        """
        Calculate topic probabilities using LDA.
        CORRECTED: Train on ORIGINAL only, then infer on all conditions.
        """
        orig_sentences = segment_into_units(original_text, unit_size=50, min_words=10)
        word_sentences = segment_into_units(word_shuf_text, unit_size=50, min_words=10)
        sent_sentences = segment_into_units(sent_shuf_text, unit_size=50, min_words=10)

        # Preprocess all
        orig_docs = [self.preprocess_text(sent) for sent in orig_sentences]
        orig_docs = [doc for doc in orig_docs if len(doc) > 5]

        word_docs = [self.preprocess_text(sent) for sent in word_sentences]
        word_docs = [doc for doc in word_docs if len(doc) > 5]

        sent_docs = [self.preprocess_text(sent) for sent in sent_sentences]
        sent_docs = [doc for doc in sent_docs if len(doc) > 5]

        if len(orig_docs) < 10:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])

        # Build dictionary from original only — shared vocab for all conditions
        print("   Building dictionary from original text...")
        dictionary = corpora.Dictionary(orig_docs)
        
        # Create corpora using the SAME dictionary
        orig_corpus = [dictionary.doc2bow(doc) for doc in orig_docs]
        word_corpus = [dictionary.doc2bow(doc) for doc in word_docs]
        sent_corpus = [dictionary.doc2bow(doc) for doc in sent_docs]

        # Train LDA on original only
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
    
    def compute_method(self, original_text, word_shuffled_text, sentence_shuffled_text):

        computation_start = time()
        
        original_analysis, word_shuffled_analysis, sentence_shuffled_analysis = self.calculate_lda_corrected(
            original_text, word_shuffled_text, sentence_shuffled_text
        )

        d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_analysis, word_shuffled_analysis)
        d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_analysis, sentence_shuffled_analysis)

        self.completion_time = self.method_name + " took about " + str(round((time()-computation_start),3)) + " secs."

        super().set_results(d_word,d_sent,orig_mean,word_mean,sent_mean,orig_std,word_std,sent_std, len(original_analysis))

        super().print_results()


class ComputeBERTSCORE(NLPMethod):

    def __init__(self):
        self.method_name = 'BERTScore'
        print("Running " + self.method_name + " (Sequential Coherence)")


    def calculate_bertscore_window_coherence(self, text, window_size=50):
        """
        BERTScore F1 between consecutive fixed-size 50-token windows.
        Pre-trained BERT model — no fitting on test data.

        Window strategy: fixed 50-token non-overlapping windows (no overlap
        required — we measure consecutive window similarity, not density).
        """

        words = text.split()        
        windows = []

        for i in range(0, len(words) - window_size + 1, window_size):
            window = ' '.join(words[i:i+window_size])
            windows.append(window)
        
        if len(windows) < 2:
            return np.array([0.0])
        
        cands = windows[1:]
        refs  = windows[:-1]
        print(f"   Scoring {len(cands)} consecutive window pairs ({window_size} tokens each)...")

        try:
            _, _, F1 = bertscore(
                cands, refs,
                lang='en',
                model_type='bert-base-uncased',
                verbose=False
            )
            return F1.cpu().numpy()
        except Exception as e:
            print(f"   BERTScore error: {e}")
            return np.array([0.0])        
    
    def compute_method(self, original_text, word_shuffled_text, sentence_shuffled_text):

        computation_start = time()

        print("\n⚙️  Calculating sequential coherence for ORIGINAL...")
        original_analysis = self.calculate_bertscore_window_coherence(original_text)

        print("\n⚙️  Calculating sequential coherence for WORD-SHUFFLED...")
        word_shuffled_analysis = self.calculate_bertscore_window_coherence(word_shuffled_text)

        print("\n⚙️  Calculating sequential coherence for SENTENCE-SHUFFLED...")
        sentence_shuffled_analysis = self.calculate_bertscore_window_coherence(sentence_shuffled_text)

        d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_analysis, word_shuffled_analysis)
        d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_analysis, sentence_shuffled_analysis)    

        self.completion_time = self.method_name + " took about " + str(round((time()-computation_start),3)) + " secs."

        super().set_results(d_word,d_sent,orig_mean,word_mean,sent_mean,orig_std,word_std,sent_std, len(original_analysis))

        super().print_results()

    

class COMPUTE_BERTopic(NLPMethod):

    def __init__(self):
        self.method_name = 'BERTopic'

        print(self.method_name)

    def calculate_bertopic_corrected(self, original_text, word_shuf_text, sent_shuf_text):
        """
        Calculate topic assignment probabilities using BERTopic.
        Train on ORIGINAL only.
        """
        orig_docs = segment_into_units(original_text, unit_size=50, min_words=10)
        word_docs = segment_into_units(word_shuf_text, unit_size=50, min_words=10)
        sent_docs = segment_into_units(sent_shuf_text, unit_size=50, min_words=10)

        if len(orig_docs) < 10:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])
        

        # Set the a seed to replicate same results for different runs
        umap_model = UMAP(
            random_state=42
        )

        # fit_transform on original only
        print("   Training BERTopic on original text...")
        bertopic_model = BERTopic(
            umap_model=umap_model,
            language="english",
            calculate_probabilities=True,
            verbose=False,
            min_topic_size=3,
            nr_topics="auto"
        )

        # fit_transform on original
        orig_topics, orig_probs = bertopic_model.fit_transform(orig_docs)
        orig_max_probs = np.max(orig_probs, axis=1)

        # transform() only on shuffled conditions — model never sees scrambled text
        print("   Transforming word-shuffled text...")
        word_topics, word_probs = bertopic_model.transform(word_docs)
        word_max_probs = np.max(word_probs, axis=1)

        print("   Transforming sentence-shuffled text...")
        sent_topics, sent_probs = bertopic_model.transform(sent_docs)
        sent_max_probs = np.max(sent_probs, axis=1)

        return orig_max_probs, word_max_probs, sent_max_probs
    
    def compute_method(self, original_text, word_shuffled_text, sentence_shuffled_text):

        computation_start = time()

        print("\n⚙️  Running BERTopic (single model)...")
        original_analysis, word_shuffled_analysis, sentence_shuffled_analysis  = self.calculate_bertopic_corrected(original_text, word_shuffled_text, sentence_shuffled_text)

        d_word, orig_mean, word_mean, orig_std, word_std = calculate_cohens_d(original_analysis, word_shuffled_analysis)
        d_sent, _, sent_mean, _, sent_std = calculate_cohens_d(original_analysis, sentence_shuffled_analysis)    

        self.completion_time = self.method_name + " took about " + str(round((time()-computation_start),3)) + " secs."

        super().set_results(d_word,d_sent,orig_mean,word_mean,sent_mean,orig_std,word_std,sent_std, len(original_analysis))

        super().print_results()



def get_verdict(d):
    if d >= 3.0:
        return "✅✅ STRONG PASS"
    elif d >= 2.0:
        return "✅ PASS"
    elif d >= 1.0:
        return "~ BORDERLINE"
    else:
        return "❌ FAIL"
    

# ============================================================================
# RESULTS REPORTING
# ============================================================================

def print_results(executed_methods):

    print("\n" + "="*100)
    print("  FINAL RESULTS — 3-WAY SHUFFLE COMPARISON")
    print("="*100)
    print(f"\n{'Method':<14} {'d(Word-Shuf)':>13} {'Verdict':<25} "
          f"{'d(Sent-Shuf)':>13} {'Verdict':<25} {'n':>6}")
    print("-" * 90)

    for what_method in executed_methods:
        r = what_method.results_data
        print(f"{what_method.method_name:<14} "
              f"{r['d_vs_word']:>13.3f} {get_verdict(r['d_vs_word']):<18} "
              f"{r['d_vs_sent']:>13.3f} {get_verdict(r['d_vs_sent']):<18} "
              f"{r['n_observations']:>6}")
    print("="*100)

    # ── Mean values ────────────────────────────────────────────────────────────
    print(f"\n{'Method':<14} {'Original':>12} {'Word-Shuf':>12} {'Sent-Shuf':>12}")
    print("-" * 55)
    for what_method in executed_methods:
        r = what_method.results_data
        print(f"{what_method.method_name:<14} "
              f"{r['original_mean']:>12.4f} "
              f"{r['word_shuf_mean']:>12.4f} "
              f"{r['sent_shuf_mean']:>12.4f}")

    # ── Detailed statistics ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    for what_method in executed_methods:
        r = what_method.results_data
        print(f"\n{what_method.method_name}:")
        print(f"  d(orig vs word-shuffled):     {r['d_vs_word']:.3f}  {get_verdict(r['d_vs_word'])}")
        print(f"  d(orig vs sentence-shuffled): {r['d_vs_sent']:.3f}  {get_verdict(r['d_vs_sent'])}")
        print(f"  Original:        mean={r['original_mean']:.4f}  SD={r['original_std']:.4f}")
        print(f"  Word-shuffled:   mean={r['word_shuf_mean']:.4f}  SD={r['word_shuf_std']:.4f}")
        print(f"  Sent-shuffled:   mean={r['sent_shuf_mean']:.4f}  SD={r['sent_shuf_std']:.4f}")
        print(f"  N observations:  {r['n_observations']}")
        print("  Completion time: ", what_method.completion_time)


    # ── Key insights ────────────────────────────────────────────────
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    tiers_word = {
        'STRONG PASS': [m for m in executed_methods if m.results_data['d_vs_word'] >= 3.0],
        'PASS':        [m for m in executed_methods if 2.0 <= m.results_data['d_vs_word'] < 3.0],
        'BORDERLINE':  [m for m in executed_methods if 1.0 <= m.results_data['d_vs_word'] < 2.0],
        'FAIL':        [m for m in executed_methods if m.results_data['d_vs_word'] < 1.0],
    }
    tiers_sent = {
        'STRONG PASS': [m for m in executed_methods if m.results_data['d_vs_sent'] >= 3.0],
        'PASS':        [m for m in executed_methods if 2.0 <= m.results_data['d_vs_sent'] < 3.0],
        'BORDERLINE':  [m for m in executed_methods if 1.0 <= m.results_data['d_vs_sent'] < 2.0],
        'FAIL':        [m for m in executed_methods if m.results_data['d_vs_sent'] < 1.0],
    }

    tier_labels = [
        ('✅✅ STRONG PASS (d ≥ 3.0)',    'STRONG PASS'),
        ('✅  PASS (2.0 ≤ d < 3.0)',      'PASS'),
        ('~  BORDERLINE (1.0 ≤ d < 2.0)', 'BORDERLINE'),
        ('❌  FAIL (d < 1.0)',             'FAIL'),
    ]

    print(f"\n📊 WORD-SHUFFLE SENSITIVITY (total structure destruction):")
    for label, key in tier_labels:
        print(f"   {label}: {len(tiers_word[key])} method(s)")
        for m in tiers_word[key]:
            print(f"      • {m.method_name}  (d = {m.results_data['d_vs_word']:.3f})")

    print(f"\n📊 SENTENCE-SHUFFLE SENSITIVITY (discourse structure only):")
    for label, key in tier_labels:
        print(f"   {label}: {len(tiers_sent[key])} method(s)")
        for m in tiers_sent[key]:
            print(f"      • {m.method_name}  (d = {m.results_data['d_vs_sent']:.3f})")


    discourse_sensitive = [
        m for m in executed_methods
        if m.results_data['d_vs_sent'] >= 1.0
        and m.results_data['d_vs_word'] > m.results_data['d_vs_sent']
    ]
    print(f"\n🎯 DISCOURSE-LEVEL SENSITIVE METHODS:")
    print("   (Detect both total destruction AND discourse-only disruption)")
    if discourse_sensitive:
        for m in discourse_sensitive:
            d_w   = m.results_data['d_vs_word']
            d_s   = m.results_data['d_vs_sent']
            ratio = d_w / d_s if d_s > 0 else 0.0
            print(f"   • {m.method_name}:  word d={d_w:.3f},  sent d={d_s:.3f},  ratio={ratio:.1f}x")
    else:
        print("   None of the tested methods show significant discourse-level sensitivity.")
        print("   This is the gap that Symbolic Entropy's Σ component is designed to fill.")

    print("\n" + "="*80)
    print("\n✅ 3-WAY SHUFFLE TEST SUITE COMPLETE!")


def main():

    # ============================================================================
    # FILE UPLOAD
    # ============================================================================

    print("\n" + "="*80)
    print("📤 PLEASE UPLOAD YOUR 3 FILES")
    print("="*80)

    prompt_select_files = SelectFiles()
    selected_files = prompt_select_files.select_files_to_process(local_input_path)
    input_files = {}

    if selected_files:
        print(f"Selected files ({len(selected_files)}):")
        for path in selected_files:
            print(f"  - {path}")

            if "original" in path:
                print("\n📤 Upload ORIGINAL file (.docx or .txt):")
                input_files['original'] = path

            if "word_randomized" in path:
                print("\n📤 Upload WORD-SHUFFLED file (.docx or .txt):")
                input_files['word_randomized'] = path

            if "sentence_shuffled" in path:
                print("\n📤 Upload SENTENCE-SHUFFLED file (.docx or .txt):")
                input_files['sentence_shuffled'] = path
    else:
        print("No files selected.")


    # Read files
    print("\n📖 Reading uploaded files...")
    original_text = read_file(input_files['original'])
    word_shuffled_text = read_file(input_files['word_randomized'])
    sent_shuffled_text = read_file(input_files['sentence_shuffled'])

    print(f"✓ Original text:         {len(original_text)} characters")
    print(f"✓ Word-shuffled text:    {len(word_shuffled_text)} characters")
    print(f"✓ Sentence-shuffled text: {len(sent_shuffled_text)} characters")

    methods_run = 0
    executed_methods = []

    for method, run in list_of_methods.items():    

        if method == 'Perplexity' and run == True :
            methods_run += 1

            # ============================================================================
            # METHOD 1: PERPLEXITY (GPT-2) - SLIDING WINDOW VERSION
            # ============================================================================
            # Uses pre-trained model, no fitting on test data

            print("\n" + "="*80)
            print("METHOD " + str(methods_run) + "/" + str(total_methods) + ": GPT-2 Perplexity (Sliding Window)")
            print("="*80)

            compute_perplexity = ComputePerplexity()
            compute_perplexity.compute_method(original_text, word_shuffled_text, sent_shuffled_text)
            executed_methods.append(compute_perplexity)

        if method == 'Sentiment' and run == True :
            methods_run += 1
            
            print("\n" + "="*80)
            print("METHOD " + str(methods_run) + "/" + str(total_methods) + " Sentiment Analysis")
            print("="*80)

            compute_sentiment = ComputeSentiment()
            compute_sentiment.compute_method(original_text, word_shuffled_text, sent_shuffled_text)
            executed_methods.append(compute_sentiment)

        if method == 'TF-IDF' and run == True :
            methods_run += 1

            # ============================================================================
            # METHOD 3: TF-IDF
            # ============================================================================
            # Fits vectorizer on ORIGINAL only, transforms all

            print("\n" + "="*80)
            print("METHOD " + str(methods_run) + "/" + str(total_methods) + " TF-IDF Coherence")
            print("="*80)

            print("\n⚙️  Calculating TF-IDF coherence (single vectorizer)...")

            compute_tf_idf = ComputeTF_IDF()
            compute_tf_idf.compute_method(original_text, word_shuffled_text, sent_shuffled_text)
            executed_methods.append(compute_tf_idf)

        if method == 'NER' and run == True :
            methods_run += 1


            # ============================================================================
            # METHOD 4: NER - SENTENCE-LEVEL VERSION
            # ============================================================================
            # Uses pre-trained model, no fitting on test data

            print("\n" + "="*80)
            print("METHOD " + str(methods_run) + "/" + str(total_methods) + " Named Entity Recognition (Sentence-Level)")
            print("="*80)

            compute_ner = ComputeNER()
            compute_ner.compute_method(original_text, word_shuffled_text, sent_shuffled_text)
            executed_methods.append(compute_ner)

        if method == 'LDA' and run == True :
            methods_run += 1


            # # ============================================================================
            # # METHOD 5: LDA - DOCUMENT-LEVEL TOPIC PROBABILITIES (CORRECTED)
            # # ============================================================================
            # # Trains on ORIGINAL only, infers on all conditions

            print("\n" + "="*80)
            print("METHOD " + str(methods_run) + "/" + str(total_methods) + " LDA Topic Modeling (Document-Level Probabilities)")
            print("="*80)

            # print("\n⚙️  Running LDA (single model)...")

            compute_lda = ComputeLDA()
            compute_lda.compute_method(original_text, word_shuffled_text, sent_shuffled_text)
            executed_methods.append(compute_lda)

        if method == 'BERTScore' and run == True :
            methods_run += 1


            # ============================================================================
            # METHOD 6: BERTSCORE - SEQUENTIAL COHERENCE (CORRECTED)
            # ============================================================================
            # Uses fixed-size windows instead of period-based splitting

            print("\n" + "="*80)
            print("METHOD " + str(methods_run) + "/" + str(total_methods) + " BERTScore (Sequential Coherence)")
            print("="*80)

            compute_bertscore = ComputeBERTSCORE()
            compute_bertscore.compute_method(original_text, word_shuffled_text, sent_shuffled_text)
            executed_methods.append(compute_bertscore)

        if method == 'BERTopic' and run == True :
            methods_run += 1

            # ============================================================================
            # METHOD 7: BERTOPIC
            # ============================================================================
            # Trains on ORIGINAL only, transforms all conditions

            print("\n" + "="*80)
            print("METHOD " + str(methods_run) + "/" + str(total_methods) + " BERTopic")
            print("="*80)        

            compute_bertopic = COMPUTE_BERTopic()
            compute_bertopic.compute_method(original_text, word_shuffled_text, sent_shuffled_text)
            executed_methods.append(compute_bertopic)

    if len(executed_methods) > 0:
        print_results(executed_methods)
    else:
        print("\nNo results to display.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print('CUDA is available')
    else:
        print('NO CUDA available')

    main()
