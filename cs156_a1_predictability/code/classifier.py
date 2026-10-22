"""Compare early and recent essays using classifiers and compression.

The script evaluates function-word and TF-IDF classifiers with leave-one-essay-
out validation, then cross-checks the results with compression-based measures.
Run with ``python3 -I classifier.py ESSAY_DIR OUT_DIR``.
"""
import bz2
import gzip
import importlib.util
import itertools
import json
import lzma
import os
import random
import sys

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "pred", os.path.join(here, "predictability.py")
)
pred = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pred)

NAMES = pred.EARLY + pred.RECENT
LABEL = {n: 0 for n in pred.EARLY} | {n: 1 for n in pred.RECENT}
CHUNK = 300
FWORDS = sorted(pred.FUNCTION_WORDS)

def tokenize_essay(name):
    """Return an essay's tokens without sentence-boundary markers."""
    sentences = pred.sentences(pred.read(name))
    return [
        word
        for sentence in sentences
        for word in pred.tokens(sentence, False)
        if word not in ("<s>", "</s>")
    ]

chunks, chunk_owners = [], []
for essay in NAMES:
    tokens = tokenize_essay(essay)
    for start in range(0, len(tokens) - CHUNK // 2, CHUNK):
        chunk = tokens[start:start + CHUNK]
        if len(chunk) >= CHUNK // 2:
            chunks.append(" ".join(chunk))
            chunk_owners.append(essay)
chunk_owners = np.array(chunk_owners)
print(f"{len(chunks)} chunks of ~{CHUNK} words from {len(NAMES)} essays")


def build_function_word_model():
    """Create a classifier using normalized function-word frequencies."""
    vectorizer = CountVectorizer(
        vocabulary=FWORDS, token_pattern=r"[a-z']+", lowercase=False
    )
    relative_frequencies = FunctionTransformer(
        lambda X: X.toarray() / np.maximum(X.sum(1).A, 1)
    )
    classifier = LogisticRegression(
        C=0.3, class_weight="balanced", max_iter=2000
    )
    return make_pipeline(
        vectorizer, relative_frequencies, StandardScaler(), classifier
    )


def build_tfidf_model():
    """Create a classifier using TF-IDF features from all words."""
    vectorizer = TfidfVectorizer(
        token_pattern=r"[a-z']+|<num>", min_df=2, max_features=2000
    )
    classifier = LogisticRegression(
        C=3.0, class_weight="balanced", max_iter=2000
    )
    return make_pipeline(vectorizer, classifier)


def leave_one_essay_out(make_model, labels):
    """Measure balanced essay accuracy while holding out each essay in turn.

    Returns the mean recall across the two classes and each essay's mean
    predicted probability of belonging to the recent-writing class.
    """
    chunk_labels = np.array([labels[essay] for essay in chunk_owners])
    probabilities = {}
    for held in NAMES:
        training_mask = chunk_owners != held
        test_mask = chunk_owners == held
        if len(set(chunk_labels[training_mask])) < 2:
            continue
        model = make_model().fit(
            [chunk for chunk, is_training in zip(chunks, training_mask) if is_training],
            chunk_labels[training_mask],
        )
        test_chunks = [
            chunk for chunk, is_test in zip(chunks, test_mask) if is_test
        ]
        probabilities[held] = model.predict_proba(test_chunks)[:, 1].mean()

    predicted_labels = {
        name: int(probability > 0.5)
        for name, probability in probabilities.items()
    }
    recalls = [
        np.mean([
            predicted_labels[name] == labels[name]
            for name in NAMES
            if labels[name] == class_label and name in predicted_labels
        ])
        for class_label in (0, 1)
    ]
    return float(np.mean(recalls)), probabilities


rng = random.Random(156)
results = {}
for name, make_model in (
    ("function_words", build_function_word_model),
    ("tfidf_all_words", build_tfidf_model),
):
    acc, probs = leave_one_essay_out(make_model, LABEL)
    null = []
    for _ in range(300):
        permuted_names = NAMES[:]
        rng.shuffle(permuted_names)
        permuted_labels = {
            essay: (1 if index < len(pred.RECENT) else 0)
            for index, essay in enumerate(permuted_names)
        }
        null.append(leave_one_essay_out(make_model, permuted_labels)[0])
    p = (1 + sum(x >= acc - 1e-12 for x in null)) / (1 + len(null))
    misclassified = [
        essay[:28]
        for essay in NAMES
        if essay in probs and int(probs[essay] > 0.5) != LABEL[essay]
    ]
    print(
        f"{name}: leave-one-essay-out balanced accuracy {acc:.2f} "
        f"(chance 0.50), permutation p = {p:.4f}; "
        f"null mean {np.mean(null):.2f}; misclassified: {misclassified}"
    )
    results[name] = dict(
        balanced_accuracy=acc,
        p=p,
        null_mean=float(np.mean(null)),
        misclassified=misclassified,
        prob_recent={
            essay: round(probability, 3)
            for essay, probability in probs.items()
        },
    )
    if name == "function_words":
        labels_by_chunk = np.array([LABEL[essay] for essay in chunk_owners])
        model = make_model().fit(chunks, labels_by_chunk)
        coefficients = model[-1].coef_[0]
        coefficient_order = np.argsort(coefficients)
        early_words = [FWORDS[index] for index in coefficient_order[:12]]
        recent_words = [FWORDS[index] for index in coefficient_order[::-1][:12]]
        print("  function words pointing to EARLY:", early_words)
        print("  function words pointing to RECENT:", recent_words)
        results[name]["early_words"] = early_words
        results[name]["recent_words"] = recent_words

# ---- compression: redundancy of each essay on its own, equal-length samples

def prepare_compression_text(name, mask_function_words):
    """Return normalized essay text, optionally masking non-function words."""
    words = tokenize_essay(name)
    if mask_function_words:
        words = [
            word if word in pred.FUNCTION_WORDS else "w"
            for word in words
        ]
    return " ".join(words)


def compressed_size(data, algorithm):
    """Return the compressed byte length using the named compression method."""
    compressors = {
        "gzip": lambda value: gzip.compress(value, 9),
        "bz2": lambda value: bz2.compress(value, 9),
        "lzma": lambda value: lzma.compress(
            value, preset=9 | lzma.PRESET_EXTREME
        ),
    }
    return len(compressors[algorithm](data))

compression_results = {}
for mask_function_words in (False, True):
    texts = {
        name: prepare_compression_text(name, mask_function_words)
        for name in NAMES
    }
    sample_length = min(len(text) for text in texts.values())
    for algorithm in ("gzip", "bz2", "lzma"):
        bits_per_char = {
            name: (
                8 * compressed_size(
                    texts[name][:sample_length].encode(), algorithm
                ) / sample_length
            )
            for name in NAMES
        }
        early_values = [bits_per_char[name] for name in pred.EARLY]
        recent_values = [bits_per_char[name] for name in pred.RECENT]
        delta, p_value, _ = pred.perm_test(early_values, recent_values)
        key = (
            f"{'function_words' if mask_function_words else 'all_words'}_"
            f"{algorithm}"
        )
        compression_results[key] = dict(
            early=float(np.mean(early_values)),
            recent=float(np.mean(recent_values)),
            delta=float(delta),
            p=p_value,
            chars=sample_length,
        )
        print(
            f"compression {key:24s} first {sample_length} chars: "
            f"early {np.mean(early_values):.3f} bits/char, "
            f"recent {np.mean(recent_values):.3f}, "
            f"delta {delta:+.3f}, p = {p_value:.4f}"
        )
    # drift: normalised compression distance, same period vs across periods
    def normalized_compression_distance(first, second, algorithm="lzma"):
        """Measure the compression distance between two equal-length texts."""
        first_text = texts[first][:sample_length]
        second_text = texts[second][:sample_length]
        first_size, second_size, combined_size = (
            compressed_size(value.encode(), algorithm)
            for value in (
                first_text,
                second_text,
                first_text + second_text,
            )
        )
        return (
            combined_size - min(first_size, second_size)
        ) / max(first_size, second_size)

    early_early = np.mean([
        normalized_compression_distance(first, second)
        for first, second in itertools.combinations(pred.EARLY, 2)
    ])
    recent_recent = np.mean([
        normalized_compression_distance(first, second)
        for first, second in itertools.combinations(pred.RECENT, 2)
    ])
    early_recent = np.mean([
        normalized_compression_distance(early, recent)
        for early in pred.EARLY
        for recent in pred.RECENT
    ])
    result_key = (
        f"ncd_{'function_words' if mask_function_words else 'all_words'}"
    )
    compression_results[result_key] = dict(
        early_early=float(early_early),
        recent_recent=float(recent_recent),
        early_recent=float(early_recent),
    )
    print(
        f"NCD ({'function words' if mask_function_words else 'all words'}, lzma): "
        f"early-early {early_early:.3f}, recent-recent {recent_recent:.3f}, "
        f"early-recent {early_recent:.3f}"
    )
results["compression"] = compression_results
output_path = os.path.join(sys.argv[2], "classifier_compression_results.json")
with open(output_path, "w") as output_file:
    json.dump(results, output_file, indent=1)
