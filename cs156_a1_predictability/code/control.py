"""Negative control for the early-vs-recent classifier: same pipeline, arbitrary labels.
Usage: python3 -I control.py ESSAY_DIR OUT_DIR"""
import importlib.util, json, os, random, sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pred", os.path.join(here, "predictability.py"))
pred = importlib.util.module_from_spec(spec); spec.loader.exec_module(pred)
pred.ESSAYS = sys.argv[1]
CHUNK = 300
FW = sorted(pred.FUNCTION_WORDS)
NAMES = pred.EARLY + pred.RECENT

def toks(n):
    return [w for s in (pred.tokens(x, False) for x in pred.sentences(pred.read(n))) for w in s if w not in ("<s>", "</s>")]
chunks, owner = [], []
for n in NAMES:
    t = toks(n)
    for i in range(0, len(t) - CHUNK // 2, CHUNK):
        c = t[i:i + CHUNK]
        if len(c) >= CHUNK // 2:
            chunks.append(" ".join(c)); owner.append(n)
owner = np.array(owner)

def fw_model():
    cv = CountVectorizer(vocabulary=FW, token_pattern=r"[a-z']+", lowercase=False)
    rel = FunctionTransformer(lambda X: X.toarray() / np.maximum(X.sum(1).A, 1))
    return make_pipeline(cv, rel, StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
def tfidf_model():
    return make_pipeline(TfidfVectorizer(token_pattern=r"[a-z']+|<num>", min_df=2, max_features=2000),
                         LogisticRegression(C=3.0, class_weight="balanced", max_iter=2000))

def auc(labels, probs):
    pos = [probs[n] for n in probs if labels[n] == 1]; neg = [probs[n] for n in probs if labels[n] == 0]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))

def loeo(make, labels):
    names = list(labels)
    y = np.array([labels.get(o, -1) for o in owner])
    probs = {}
    for held in names:
        tr = (owner != held) & (y >= 0); te = owner == held
        m = make().fit([c for c, k in zip(chunks, tr) if k], y[tr])
        probs[held] = m.predict_proba([c for c, k in zip(chunks, te) if k])[:, 1].mean()
    return auc(labels, probs), probs

FALL23 = [n for n in pred.EARLY if n.endswith("fall2023")]; SPR24 = [n for n in pred.EARLY if n.endswith("spring2024")]
FALL25 = [n for n in pred.RECENT if n.endswith("fall2025")]; SPR26 = [n for n in pred.RECENT if n.endswith("spring2026")]
controls = {
    "early_only: Fall 2023 vs Spring 2024": ({n: 0 for n in FALL23} | {n: 1 for n in SPR24}),
    "recent_only: Fall 2025 vs Spring 2026": ({n: 0 for n in FALL25} | {n: 1 for n in SPR26}),
    "real: early vs recent": ({n: 0 for n in pred.EARLY} | {n: 1 for n in pred.RECENT}),
}
out = {}
for mname, make in (("function_words", fw_model), ("tfidf_all_words", tfidf_model)):
    for cname, lab in controls.items():
        a, _ = loeo(make, lab)
        sizes = f"{sum(v==0 for v in lab.values())} vs {sum(v==1 for v in lab.values())}"
        print(f"{mname:16s} {cname:40s} ({sizes}) AUC = {a:.3f}", flush=True)
        out[f"{mname} | {cname}"] = dict(auc=a, sizes=sizes)
    # null: random 9/6 labelling of all 15 essays
    rng = random.Random(156); nulls = []
    for _ in range(100):
        perm = NAMES[:]; rng.shuffle(perm)
        nulls.append(loeo(make, {n: int(i < 6) for i, n in enumerate(perm)})[0])
    real = out[f"{mname} | real: early vs recent"]["auc"]
    out[f"{mname} | shuffled 9/6 null"] = dict(mean=float(np.mean(nulls)), p95=float(np.percentile(nulls, 95)), max=float(max(nulls)),
                                               frac_ge_real=float(np.mean([x >= real - 1e-12 for x in nulls])))
    print(f"{mname:16s} shuffled-label AUC (100 shuffles): mean {np.mean(nulls):.3f}, 95th pct {np.percentile(nulls,95):.3f}, max {max(nulls):.3f}", flush=True)
json.dump(out, open(os.path.join(sys.argv[2], "control_results.json"), "w"), indent=1)
