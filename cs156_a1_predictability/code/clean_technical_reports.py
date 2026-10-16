"""Prose-only extraction from LaTeX-style technical reports (equations, tables and figure panels dropped).
Usage: python3 -I clean_tech.py RAW_DIR OUT_DIR
"""
import os, re, sys
RAW, OUT = sys.argv[1], sys.argv[2]
CFG = {
    "CS166_Final_Project_2": dict(out="essay_brain_criticality_fall2025.txt", start=r"^Introduction\s*$", end=r"^References\s*$"),
    "Statistical_Inference_Assignment_Template": dict(out="essay_udemy_statistical_inference_fall2023.txt", start=r"^Introduction\s*$", end=r"^\s*AI STATEMENT"),
    "Final_Assignment": dict(out="essay_pm25_resolution_review_spring2026.txt", start=r"^Introduction\s*$", end=r"^\s*Word count"),
    "NS125_Research_Project_1_1": dict(out="essay_pulse_oximetry_fall2025.txt", start=r"^Introduction\s*$", end=r"^\s*Word count"),
    "Neuroscience_in_the_News_Assignment_1": dict(out="essay_huntingtons_news_spring2026.txt", start=r"^Introduction\s*$", end=r"^\s*Word count"),
    "Minerva_Assignment_Grader_Airport_Queue": dict(out="essay_airport_queue_fall2025_GROUP.txt", start=r"^Introduction\s*$", end=r"^Word count"),
}
MATH = set("∑∈≈≤≥≠×∫√∂∆⟨⟩αβγδεθλμρσχωΩΣ=<>^_{}|∝→⇒⇐⌈⌉⟺∼±·−")
LONE_NUM = re.compile(r"^\s*\d{1,2}\s*$")
FOOTNOTE_START = re.compile(r"^\s*(\d{1,2}\s+)?#\w+")
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

def is_prose(l, prev=False):
    s = l.strip()
    if not s: return False
    words = [w for w in WORD.findall(s) if len(w) >= 3]
    nonsp = re.sub(r"\s", "", s)
    alpha = sum(c.isalpha() for c in nonsp)
    mathc = sum(c in MATH for c in nonsp)
    ratio = alpha / max(1, len(nonsp))
    if len(words) >= 5 and mathc <= 1 and ratio >= 0.8:
        return True
    # short tail of a prose paragraph, e.g. "structural maturation (da Silva, 2003)."
    return prev and len(words) >= 2 and mathc == 0 and ratio >= 0.6

for stem, cfg in CFG.items():
    raw = open(os.path.join(RAW, stem + ".txt"), encoding="utf-8").read()
    st, en = re.compile(cfg["start"]), re.compile(cfg["end"])
    started = finished = False
    chunks, cur = [], []
    for page in raw.split("\f"):
        plines = page.split("\n")
        for i, l in enumerate(plines):  # footnote block (course-outcome tags) runs to the end of the page
            if FOOTNOTE_START.match(l):
                plines = plines[:i]
                break
        for l in plines:
            if LONE_NUM.match(l) or re.match(r"^\s*STATISTICAL INFERENCE\b.*\d+\s*$", l): continue
            if not started:
                started = bool(st.match(l.strip())); continue
            if en.match(l.strip()): finished = True; break
            if is_prose(l, bool(cur)): cur.append(l.strip())
            elif l.strip() and cur: chunks.append(cur); cur = []  # blank lines don't end a chunk
        if finished: break
    if cur: chunks.append(cur)
    # join hyphen-broken words: join only if the glued form is a word seen elsewhere
    broken = re.compile(r"[A-Za-z]+-\s+[a-z]+")  # line-end hyphen pairs: their fragments are not vocabulary
    allwords = set(w.lower() for c in chunks for w in WORD.findall(broken.sub(" ", " ".join(c))))
    for fn in os.listdir(OUT):
        if fn.endswith(".txt") and fn not in [c["out"] for c in CFG.values()]:  # don't learn from our own output
            allwords |= set(w.lower() for w in WORD.findall(open(os.path.join(OUT, fn), encoding="utf-8").read()))
    notes = []
    paras = []
    for c in chunks:
        out = ""
        for l in c:
            if out.endswith("-") and l[:1].islower():
                left = re.search(r"([A-Za-z]+)-$", out)
                right = re.match(r"[A-Za-z]+", l)
                glued = (left.group(1) + right.group(0)).lower() if left and right else ""
                lf = left.group(1) if left else ""
                rt = right.group(0) if right else ""
                if glued in allwords or (lf and not lf.isupper() and (lf.lower() not in allwords or rt.lower() not in allwords)):
                    out = out[:-1] + l; notes.append("joined " + glued)
                else:
                    out = out + l; notes.append("kept hyphen " + (left.group(1) if left else "") + "-" + (right.group(0) if right else ""))
            else:
                out = (out + " " + l).strip()
        paras.append(out)
    # keep only complete prose paragraphs (equation lead-ins and fragments are dropped),
    # and drop sentences that carry inline math symbols or variable names
    BAD = re.compile(r"\b[A-Za-z]{1,2}ij\b|\b[A-Za-z]\s?\(t\)|\(t\)|\b[A-Za-z]\s?\([a-zA-Z]\)")
    def tidy(q):
        q = re.sub(r"(?<=[.)])\s\d{1,2}(?=\s)", "", q)       # superscript after a citation or full stop
        q = re.sub(r"(?<=[a-z]\.)\d{1,2}(?=\s|$)", "", q)     # glued: "peak.3"
        q = re.sub(r"(?<=\))\d{1,2}(?=[\s.,;])", "", q)       # glued: "error)7."
        q = re.sub(r"\s\d\.(?=\s[A-Z])", "", q)              # list numbers "1. Randomness"
        return q
    paras = [tidy(q) for q in paras]
    kept_paras = []
    for para in paras:
        prot = re.sub(r"\b(al|e\.g|i\.e|vs|Fig|Eq|etc|cf|approx)\.\s", lambda m: m.group(0).replace(". ", ".\u0001"), para)
        sents = [x.replace("\u0001", " ") for x in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"])|(?<=\d\d\d\d\))\s+(?=[A-Z])", prot)]
        good = []
        for x in sents:
            x = x.strip()
            if (x[:1].isupper() or x[:1] in "“\"(") and re.search(r"[.!?][\)\"”]?$|\d\d\d\d[a-z]?\)$", x) \
               and len(WORD.findall(x)) >= 4 and not any(c in MATH for c in x) and not BAD.search(x):
                good.append(x)
        if good:
            kept_paras.append(" ".join(good))
    paras = kept_paras
    text = "\n\n".join(paras) + "\n"
    text = text.replace("‐", "-")
    open(os.path.join(OUT, cfg["out"]), "w", encoding="utf-8").write(text)
    print(f"{cfg['out']}: {len(WORD.findall(text))} words, started={started}, found_end={finished}, chunks={len(paras)}")
    print("  hyphen decisions:", sorted(set(notes)))
