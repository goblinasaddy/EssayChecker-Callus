"""Surface and stylometric feature extractor for admissions essays."""
import math
import os
import zlib
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np
import nltk

from src.features.base import BaseFeatureExtractor, FeatureMetadata
from src.segmentation.segmenter import HierarchicalSegmenter, EssaySegmentation

# Configure deterministic bundled NLTK data path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUNDLED_NLTK_DATA = os.path.join(REPO_ROOT, "data", "nltk_data")
if os.path.exists(BUNDLED_NLTK_DATA) and BUNDLED_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, BUNDLED_NLTK_DATA)


class SurfaceFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts surface, stylometric, rhythmic, syntactic, and predictability features.
    Fully self-contained: uses bundled lexicons and POS models without runtime downloads.
    """

    # 198 standard English stopwords bundled directly for zero disk/network latency
    ENGLISH_STOPWORDS = {
        'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any',
        'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
        'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't",
        'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few',
        'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven',
        "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', 'hers', 'herself', 'him',
        'himself', 'his', 'how', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', 'isn',
        "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me',
        'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't",
        'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our',
        'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she',
        "she'd", "she'll", "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some',
        'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves',
        'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those',
        'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we',
        "we'd", "we'll", "we're", "we've", 'were', 'weren', "weren't", 'what', 'when', 'where',
        'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't",
        'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
    }

    def __init__(self, segmenter: Optional[HierarchicalSegmenter] = None):
        self.segmenter = segmenter or HierarchicalSegmenter()
        self.stop_words = self.ENGLISH_STOPWORDS

    def _tag_pos(self, words: List[str]) -> List[Tuple[str, str]]:
        """Tags words with POS tags using bundled model or deterministic fallback."""
        if not words:
            return []
        try:
            return nltk.pos_tag(words)
        except Exception:
            return self._fallback_pos_tag(words)

    def _fallback_pos_tag(self, words: List[str]) -> List[Tuple[str, str]]:
        """Fast deterministic rule-based POS tagger fallback."""
        pronouns = {"i", "me", "my", "myself", "we", "us", "our", "ours", "you", "your", "he", "him", "his", "she", "her", "it", "its", "they", "them", "their"}
        prepositions = {"in", "on", "at", "to", "for", "with", "from", "by", "about", "into", "through", "during", "before", "after", "above", "below"}
        be_verbs = {"is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should"}
        
        tags = []
        for w in words:
            lw = w.lower()
            if lw in pronouns:
                tags.append((w, "PRP"))
            elif lw in prepositions:
                tags.append((w, "IN"))
            elif lw in be_verbs:
                tags.append((w, "VB"))
            elif lw.endswith("ly"):
                tags.append((w, "RB"))
            elif lw.endswith("ing"):
                tags.append((w, "VBG"))
            elif lw.endswith("ed"):
                tags.append((w, "VBD"))
            elif lw.endswith("tion") or lw.endswith("ment") or lw.endswith("ness"):
                tags.append((w, "NN"))
            elif lw.endswith("ous") or lw.endswith("ful") or lw.endswith("ive") or lw.endswith("al"):
                tags.append((w, "JJ"))
            else:
                tags.append((w, "NN"))
        return tags

    def extract_features(self, text: str, segmentation: Optional[EssaySegmentation] = None) -> Dict[str, float]:
        if segmentation is None:
            segmentation = self.segmenter.segment(text)

        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return {m.name: 0.0 for m in self.get_metadata()}

        # 1. Information-Theoretic Predictability & Entropy
        char_counts = Counter(text)
        total_chars = len(text)
        char_entropy = -sum((cnt / total_chars) * math.log2(cnt / total_chars) for cnt in char_counts.values() if cnt > 0)
        
        word_counts = Counter(tokens)
        word_entropy = -sum((cnt / total_tokens) * math.log2(cnt / total_tokens) for cnt in word_counts.values() if cnt > 0)
        
        # Word Bigram Entropy
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        if bigrams:
            bigram_counts = Counter(bigrams)
            total_bigrams = len(bigrams)
            bigram_entropy = -sum((cnt / total_bigrams) * math.log2(cnt / total_bigrams) for cnt in bigram_counts.values() if cnt > 0)
        else:
            bigram_entropy = 0.0

        # Compression ratio (zlib bytes / raw utf8 bytes)
        raw_bytes = text.encode("utf-8")
        compressed_bytes = zlib.compress(raw_bytes)
        compression_ratio = len(compressed_bytes) / max(1, len(raw_bytes))

        # 2. Sentence Rhythm & Burstiness
        sentence_lengths = [len(re.findall(r"\b\w+\b", s.text)) for s in segmentation.sentences]
        if not sentence_lengths:
            sentence_lengths = [total_tokens]

        sent_len_arr = np.array(sentence_lengths, dtype=float)
        sent_len_mean = float(np.mean(sent_len_arr))
        sent_len_median = float(np.median(sent_len_arr))
        sent_len_std = float(np.std(sent_len_arr))
        sent_len_cv = float(sent_len_std / max(1e-5, sent_len_mean))
        
        # Sentence-to-sentence length delta variance (Burstiness metric)
        if len(sent_len_arr) > 1:
            sent_deltas = np.diff(sent_len_arr)
            sent_delta_variance = float(np.var(sent_deltas))
            sent_delta_mean_abs = float(np.mean(np.abs(sent_deltas)))
        else:
            sent_delta_variance = 0.0
            sent_delta_mean_abs = 0.0

        # 3. Lexical Richness & Repetition
        unique_tokens = len(word_counts)
        ttr = unique_tokens / total_tokens
        root_ttr = unique_tokens / math.sqrt(total_tokens)
        hapax_count = sum(1 for cnt in word_counts.values() if cnt == 1)
        hapax_ratio = hapax_count / max(1, unique_tokens)
        
        stopword_count = sum(1 for t in tokens if t in self.stop_words)
        stopword_ratio = stopword_count / total_tokens

        # Top 10 word concentration ratio
        top_10_freq_sum = sum(cnt for _, cnt in word_counts.most_common(10))
        top_10_concentration = top_10_freq_sum / total_tokens

        # 4. Syntax & POS Statistics
        tagged = self._tag_pos(re.findall(r"\b\w+\b", text))

        pos_tags = [t[1] for t in tagged]
        pos_counts = Counter(pos_tags)
        total_tagged = max(1, len(pos_tags))

        noun_ratio = sum(pos_counts[t] for t in ["NN", "NNS", "NNP", "NNPS"]) / total_tagged
        verb_ratio = sum(pos_counts[t] for t in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]) / total_tagged
        adj_ratio = sum(pos_counts[t] for t in ["JJ", "JJR", "JJS"]) / total_tagged
        adv_ratio = sum(pos_counts[t] for t in ["RB", "RBR", "RBS"]) / total_tagged
        pronoun_ratio = sum(pos_counts[t] for t in ["PRP", "PRP$"]) / total_tagged
        preposition_ratio = sum(pos_counts[t] for t in ["IN", "TO"]) / total_tagged

        # Passive voice indicator: forms of 'be' followed by VBN
        be_lemmas = {"be", "is", "am", "are", "was", "were", "been", "being"}
        passive_count = 0
        for i in range(len(tagged) - 1):
            if tagged[i][0].lower() in be_lemmas and tagged[i+1][1] == "VBN":
                passive_count += 1
            elif i < len(tagged) - 2 and tagged[i][0].lower() in be_lemmas and tagged[i+1][1].startswith("RB") and tagged[i+2][1] == "VBN":
                passive_count += 1
        passive_ratio = passive_count / max(1, len(segmentation.sentences))

        # 5. Punctuation & Paragraph Profile
        punc_counts = Counter(re.findall(r"[,;:\—–\-\"\'\(\)\?\!]", text))
        comma_rate = (punc_counts[","] / total_tokens) * 100
        semicolon_rate = (punc_counts[";"] / total_tokens) * 100
        colon_rate = (punc_counts[":"] / total_tokens) * 100
        emdash_rate = ((punc_counts["—"] + punc_counts["–"] + punc_counts["-"]) / total_tokens) * 100
        quote_rate = ((punc_counts['"'] + punc_counts["'"]) / total_tokens) * 100
        parentheses_rate = ((punc_counts["("] + punc_counts[")"]) / total_tokens) * 100

        para_lengths = [len(re.findall(r"\b\w+\b", p.text)) for p in segmentation.paragraphs]
        if para_lengths:
            para_len_mean = float(np.mean(para_lengths))
            para_len_std = float(np.std(para_lengths))
        else:
            para_len_mean = float(total_tokens)
            para_len_std = 0.0

        return {
            "surface_char_entropy": round(char_entropy, 4),
            "surface_word_entropy": round(word_entropy, 4),
            "surface_bigram_entropy": round(bigram_entropy, 4),
            "surface_compression_ratio": round(compression_ratio, 4),
            "surface_sent_len_mean": round(sent_len_mean, 4),
            "surface_sent_len_median": round(sent_len_median, 4),
            "surface_sent_len_std": round(sent_len_std, 4),
            "surface_sent_len_cv": round(sent_len_cv, 4),
            "surface_sent_delta_variance": round(sent_delta_variance, 4),
            "surface_sent_delta_mean_abs": round(sent_delta_mean_abs, 4),
            "surface_ttr": round(ttr, 4),
            "surface_root_ttr": round(root_ttr, 4),
            "surface_hapax_ratio": round(hapax_ratio, 4),
            "surface_stopword_ratio": round(stopword_ratio, 4),
            "surface_top10_concentration": round(top_10_concentration, 4),
            "surface_pos_noun_ratio": round(noun_ratio, 4),
            "surface_pos_verb_ratio": round(verb_ratio, 4),
            "surface_pos_adj_ratio": round(adj_ratio, 4),
            "surface_pos_adv_ratio": round(adv_ratio, 4),
            "surface_pos_pronoun_ratio": round(pronoun_ratio, 4),
            "surface_pos_prep_ratio": round(preposition_ratio, 4),
            "surface_passive_ratio": round(passive_ratio, 4),
            "surface_comma_rate": round(comma_rate, 4),
            "surface_semicolon_rate": round(semicolon_rate, 4),
            "surface_colon_rate": round(colon_rate, 4),
            "surface_emdash_rate": round(emdash_rate, 4),
            "surface_quote_rate": round(quote_rate, 4),
            "surface_parentheses_rate": round(parentheses_rate, 4),
            "surface_para_len_mean": round(para_len_mean, 4),
            "surface_para_len_std": round(para_len_std, 4),
        }

    def get_metadata(self) -> List[FeatureMetadata]:
        return [
            FeatureMetadata("surface_char_entropy", "surface", "Character-level Shannon entropy in bits per character", "[0.0, 8.0]", "Measures character-level unpredictability; higher indicates diverse orthography", "Sensitive to non-ASCII encoding artifacts"),
            FeatureMetadata("surface_word_entropy", "surface", "Unigram token Shannon entropy in bits per token", "[0.0, 15.0]", "Vocabulary information density; AI text often exhibits elevated, uniform entropy", "Correlates with essay length below 150 words"),
            FeatureMetadata("surface_bigram_entropy", "surface", "Adjacent word pair transition Shannon entropy", "[0.0, 15.0]", "Syntactic sequence diversity; lower indicates formulaic transitions", "Sparse for very short essays"),
            FeatureMetadata("surface_compression_ratio", "surface", "Normalized zlib compression ratio (compressed_bytes / raw_bytes)", "[0.0, 1.0]", "Text compressibility; lower indicates repetitive/predictable phrasing", "Sensitive to character encoding and short text overhead"),
            FeatureMetadata("surface_sent_len_mean", "surface", "Mean token count per sentence", "[0.0, 150.0]", "Average pacing; AI text clusters near 18-24 tokens", "Skewed by run-on sentences"),
            FeatureMetadata("surface_sent_len_median", "surface", "Median token count per sentence", "[0.0, 150.0]", "Central tendency of sentence pace", "Robust to outliers"),
            FeatureMetadata("surface_sent_len_std", "surface", "Standard deviation of sentence lengths", "[0.0, 100.0]", "Pacing variation across essay", "Zero if single sentence"),
            FeatureMetadata("surface_sent_len_cv", "surface", "Coefficient of variation of sentence length", "[0.0, 5.0]", "Normalized rhythmic variance", "Undefined if length=0"),
            FeatureMetadata("surface_sent_delta_variance", "surface", "Variance of consecutive sentence length differences", "[0.0, 500.0]", "Sentence burstiness rhythm", "Requires >= 2 sentences"),
            FeatureMetadata("surface_sent_delta_mean_abs", "surface", "Mean absolute difference between adjacent sentence lengths", "[0.0, 100.0]", "Local pacing modulation", "Requires >= 2 sentences"),
            FeatureMetadata("surface_ttr", "surface", "Type-Token Ratio (V / N)", "[0.0, 1.0]", "Lexical richness", "Decays with text length"),
            FeatureMetadata("surface_root_ttr", "surface", "Root Type-Token Ratio (V / sqrt(N))", "[0.0, 50.0]", "Length-stabilized lexical richness", "Empirical approximation"),
            FeatureMetadata("surface_hapax_ratio", "surface", "Ratio of hapax legomena to vocabulary", "[0.0, 1.0]", "Proportion of single-use words", "Sensitive to rare spelling"),
            FeatureMetadata("surface_stopword_ratio", "surface", "Proportion of function/stop words", "[0.0, 1.0]", "Grammatical glue density", "Fixed English stopword list"),
            FeatureMetadata("surface_top10_concentration", "surface", "Frequency share of top 10 most common words", "[0.0, 1.0]", "Lexical concentration", "Higher in repetitive text"),
            FeatureMetadata("surface_pos_noun_ratio", "surface", "Ratio of nouns to all tagged words", "[0.0, 1.0]", "Nominal style density", "POS tagger accuracy dependent"),
            FeatureMetadata("surface_pos_verb_ratio", "surface", "Ratio of verbs to all tagged words", "[0.0, 1.0]", "Action/verbal style density", "POS tagger accuracy dependent"),
            FeatureMetadata("surface_pos_adj_ratio", "surface", "Ratio of adjectives to all tagged words", "[0.0, 1.0]", "Descriptive modifier density", "POS tagger accuracy dependent"),
            FeatureMetadata("surface_pos_adv_ratio", "surface", "Ratio of adverbs to all tagged words", "[0.0, 1.0]", "Adverbial qualification density", "POS tagger accuracy dependent"),
            FeatureMetadata("surface_pos_pronoun_ratio", "surface", "Ratio of pronouns to all tagged words", "[0.0, 1.0]", "Personal stance density", "POS tagger accuracy dependent"),
            FeatureMetadata("surface_pos_prep_ratio", "surface", "Ratio of prepositions to all tagged words", "[0.0, 1.0]", "Relational complexity", "POS tagger accuracy dependent"),
            FeatureMetadata("surface_passive_ratio", "surface", "Passive verb phrases per sentence", "[0.0, 5.0]", "Grammatical passivity", "Heuristic rule-based detection"),
            FeatureMetadata("surface_comma_rate", "surface", "Commas per 100 words", "[0.0, 20.0]", "Clause complexity and pacing", "Stylistic preference"),
            FeatureMetadata("surface_semicolon_rate", "surface", "Semicolons per 100 words", "[0.0, 10.0]", "Complex coordination frequency", "Stylistic preference"),
            FeatureMetadata("surface_colon_rate", "surface", "Colons per 100 words", "[0.0, 10.0]", "Apposition and list frequency", "Stylistic preference"),
            FeatureMetadata("surface_emdash_rate", "surface", "Em-dashes/hyphens per 100 words", "[0.0, 10.0]", "Parenthetical interruption style", "Formatting variation"),
            FeatureMetadata("surface_quote_rate", "surface", "Quotation marks per 100 words", "[0.0, 15.0]", "Dialogue or quotation frequency", "Dialogue vs narrative variation"),
            FeatureMetadata("surface_parentheses_rate", "surface", "Parentheses per 100 words", "[0.0, 10.0]", "Asides and qualifying remarks", "Stylistic preference"),
            FeatureMetadata("surface_para_len_mean", "surface", "Mean words per paragraph", "[0.0, 500.0]", "Paragraph chunk size", "Formatting dependent"),
            FeatureMetadata("surface_para_len_std", "surface", "Standard deviation of paragraph word counts", "[0.0, 200.0]", "Structural symmetry", "Formatting dependent"),
        ]
