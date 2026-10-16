"""Clean extracted essay text into plain prose for the n-gram project.

Usage: python3 -I clean_essays.py RAW_DIR OUT_DIR
Each essay gets a config describing where the body starts and ends and which
pages or lines to drop. Everything else is generic (running headers, footnote
blocks, footnote markers, stray page numbers).
"""
import os
import re
import sys

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

HEADER = re.compile(
    r"^\s*(BIASES|COLLEGE PERSISTENCE|SF-MARINA FOOD BANK|THESIS DRIVEN ESSAY|FOOD INSECURITY IN SAN FRANCISCO)\s*\d*\s*$"
)
LONE_NUM = re.compile(r"^\s*\d{1,2}\s*$")
FOOTNOTE_START = re.compile(r"^\s*(\d{1,2}\s+)?#\w+")

ESSAYS = {
    "Biases_1_1": dict(
        out="essay_biases_spring2024.txt",
        start=r"^LLMs like ChatGPT",
        end=r"^\s*WORD COUNT",
        drop_pages=[3],  # page 3 is the Bard prompt and Bard's AI-written answer
        drop_lines=[r"^\s*PART A\s*$", r"^\s*\d\.\s+(Cognitive|Prompt|Differences)"],
    ),
    "Research_Design_Assignment": dict(
        out="essay_water_fleas_spring2024.txt",
        start=r"^Urbanization has led",
        end=r"^\s*Word Count",
        drop_pages=[],
        drop_lines=[],
    ),
    "Survey_Assignment_The_Rock_Scientist_Group": dict(
        out="essay_sf_transit_survey_spring2024_GROUP.txt",
        start=r"^Our chosen Supporting Action",
        end=r"^\s*Word count",
        drop_pages=[],
        drop_lines=[
            r"^\s*Mayor London Breed",
            r"^\s*Office of the Mayor",
            r"^\s*City Hall",
            r"^\s*1 Dr\. Carlton",
            r"^\s*San Francisco, CA 94102",
            r"^\s*Dear Mayor",
            r"^\s*Sincerely",
            r"^\s*Team United",
        ],
        post=[("tons of CO annually", "tons of CO2 annually")],  # the marker rule ate a chemical subscript
    ),
    "Complex_Systems_Analysis_Case_Study": dict(
        out="essay_college_persistence_fall2023.txt",
        start=r"^Though blessed with many",
        end=r"^\s*WORD COUNT",
        drop_pages=[],
        drop_lines=[],
        post=[(" 2For example", " For example")],
    ),
    "Irrational_Behavior_Research_Paper": dict(
        out="essay_procrastination_fall2023_GROUP.txt",
        start=r"^\s*Procrastination, defined as",
        end=r"^\s*Word count",
        drop_pages=[],
        drop_lines=[r"^\s*C\s*-\s*k\s*t\s*\^\s*2\s*>\s*D\s*\+\s*a\s*t\s*\^\s*2\s*$"],
    ),
    "Thesis_Driven_Essay": dict(
        out="essay_kenya_democratic_machine_fall2023.txt",
        start=r"^In the modern day, people hail democracy",
        end=r"^\s*WORD COUNT",
        drop_pages=[],
        drop_lines=[],
    ),
    "Social_Change_1_1": dict(
        out="essay_sf_marina_food_bank_fall2023.txt",
        start=r"^The SF-Marina Food Bank aims",
        end=r"^\s*WORD COUNT",
        drop_pages=[],
        drop_lines=[],
        post=[("agency, 4 being", "agency, being")],
    ),
    "Problem_Solving_Part_1_Assignment": dict(
        out="essay_food_insecurity_part1_fall2023.txt",
        start=r"^The San Francisco Food Security Task Force report",
        end=r"^\s*WORD COUNT",
        drop_pages=[],
        drop_lines=[],
        post=[("average 1(Winters", "average (Winters")],
    ),
    "Problem_Solving_Part_2": dict(
        out="essay_food_security_part2_fall2023.txt",
        start=r"^The San Francisco Food Security Task Force report",
        end=r"^\s*WORD COUNT",
        drop_pages=[],
        drop_lines=[],
        post=[("average 1(Winters", "average (Winters")],
    ),
    "Scientific_Proposal_Part_2": dict(
        out="essay_scientific_proposal_part2_fall2023_GROUP.txt",
        start=r"^\s*McQuillan and James",
        end=r"^\s*Word count",
        drop_pages=[],
        drop_lines=[],
        post=[("online learning 7.", "online learning.")],
    ),
    "Data-driven_Research_Assignment": dict(
        out="essay_climate_experience_fall2025.txt",
        start=r"^Public understanding of climate change",
        end=r"^\s*Word Count",
        drop_pages=[],
        drop_lines=[],
        post=[("Summaries - V, 2025", "Summaries - V2, 2025")],  # marker rule ate the 2 in 'v2'
    ),
    "Ethical_Controversies_Assignment": dict(
        out="essay_surveillance_ethics_spring2024.txt",
        start=r"^\s*The attacks on 9/11",
        end=r"^\s*Word Count",
        drop_pages=[],
        drop_lines=[],
    ),
    "Self_Reflection_Purpose_Leadership_Assignment": dict(
        out="essay_freshman_reflection_spring2024.txt",
        start=r"^\s*I have always been scared of being average",
        end=r"^\s*Word Count",
        drop_pages=[],
        drop_lines=[],
        post=[("assignment.” 4 This", "assignment.” This")],
    ),
}


def clean_page(lines):
    """Drop running header, stray page number, and the footnote block."""
    lines = [l.rstrip().replace("\u200c", "").replace("\u200b", "") for l in lines]
    # running header and a lone page number at the top of the page
    while lines and (not lines[0].strip() or HEADER.match(lines[0]) or LONE_NUM.match(lines[0])):
        lines.pop(0)
    # footnote block runs from the first '#tag' line (plus its lone digit) to page end
    for i, l in enumerate(lines):
        if FOOTNOTE_START.match(l):
            j = i
            k = i - 1
            while k >= 0 and not lines[k].strip():
                k -= 1
            if k >= 0 and LONE_NUM.match(lines[k]):
                j = k
            lines = lines[:j]
            break
    return lines


REMOVED = []


def inline_markers(text):
    """Strip footnote numbers without touching decimals, years, or P450-style names."""
    rules = [
        # digit glued to a letter, bracket, or closing quote, then a space, punctuation, or quote
        r"(?<=[A-Za-z\u201d\"\u2019\)])\d(?=[\s\.,;:\u201d\"\u2019]|$)",
        # sentence-final period then the digit (letter or bracket before the period)
        r"(?<=[A-Za-z\u201d\"\u2019\)]\.)\d(?=\s|$)",
        # a space, then the digit, after a sentence-final period
        r"(?<=[A-Za-z\)]\.)\s\d(?=\s|$)",
    ]
    for r in rules:
        for m in re.finditer(r, text):
            REMOVED.append(text[max(0, m.start() - 25):m.end() + 12].replace("\n", " "))
        text = re.sub(r, "", text)
    return text


for stem, cfg in ESSAYS.items():
    raw = open(os.path.join(RAW, stem + ".txt"), encoding="utf-8").read()
    pages = raw.split("\f")
    start = re.compile(cfg["start"])
    end = re.compile(cfg["end"])
    drops = [re.compile(p) for p in cfg["drop_lines"]]
    kept, started, finished = [], False, False
    for pno, page in enumerate(pages, 1):
        if pno in cfg["drop_pages"]:
            continue
        lines = clean_page(page.split("\n"))
        for l in lines:
            if not started:
                if start.match(l.strip()) or start.match(l):
                    started = True
                else:
                    continue
            if end.match(l):
                finished = True
                break
            if any(d.match(l) for d in drops):
                continue
            kept.append(l)
        if finished:
            break
    # rebuild paragraphs: a new paragraph starts at an indented line after a blank
    paras, cur, prev_blank = [], [], True
    for l in kept:
        if not l.strip():
            prev_blank = True
            continue
        indented = len(l) - len(l.lstrip()) >= 3
        if cur and prev_blank and indented:
            paras.append(" ".join(cur))
            cur = []
        cur.append(re.sub(r"\s+", " ", l.strip()))
        prev_blank = False
    if cur:
        paras.append(" ".join(cur))
    text = "\n\n".join(inline_markers(p) for p in paras) + "\n"
    text = text.translate(str.maketrans({"\u0430": "a", "\u043e": "o", "\u041e": "O"}))  # Cyrillic look-alikes
    for a, b in cfg.get("post", []):
        text = text.replace(a, b)
    open(os.path.join(OUT, cfg["out"]), "w", encoding="utf-8").write(text)
    words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    print(f"{cfg['out']}: {words} words, started={started}, found_end={finished}")

print("\n--- digits removed as footnote markers (check none is a real number) ---")
for r in REMOVED:
    print("   ..." + r + "...")
