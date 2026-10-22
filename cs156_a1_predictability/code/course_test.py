"""Does sharing a course explain vocabulary overlap, and does it explain the early/recent coverage gap?
Usage: python3 -I course_test.py ESSAY_DIR OUT_DIR"""
import importlib.util, itertools, json, os, random, sys
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pred", os.path.join(here, "predictability.py"))
pred = importlib.util.module_from_spec(spec); spec.loader.exec_module(pred)

COURSE = {  # from the cover pages
    "college_persistence_fall2023": "CX50", "sf_marina_food_bank_fall2023": "CX50", "kenya_democratic_machine_fall2023": "MC50",
    "food_security_part2_fall2023": "EA50", "udemy_statistical_inference_fall2023": "FA50", "biases_spring2024": "EA51",
    "water_fleas_spring2024": "EA51", "surveillance_ethics_spring2024": "CX51", "freshman_reflection_spring2024": "CX51",
    "climate_experience_fall2025": "NS125", "pulse_oximetry_fall2025": "NS125", "brain_criticality_fall2025": "CS166",
    "pm25_resolution_review_spring2026": "NS156", "huntingtons_news_spring2026": "SS152", "pica_playgrounds_lba_spring2026": "SS152"}
E_, R_ = pred.EARLY, pred.RECENT
pred.ESS = {n: pred.essay_sents(n, False) for n in E_ + R_}
toks = {n: [w for s in pred.ESS[n] for w in s if w not in ("<s>", "</s>")] for n in pred.ESS}
m = pred.mean
print("courses: early", len({COURSE[n] for n in E_}), "for", len(E_), "essays; recent", len({COURSE[n] for n in R_}), "for", len(R_), "essays")

# ---- A. pairwise overlap, same course vs different course (equal-length samples)
M = 740
fw = pred.FUNCTION_WORDS
cont = {n: [w for w in toks[n][:M] if w not in fw and w != "<num>"] for n in toks}
types = {n: set(cont[n]) for n in toks}
# Jaccard overlap of two essays' content-word types
def jac(a, b): return len(types[a] & types[b]) / len(types[a] | types[b])
# average share of each essay's content-word tokens that also appear in the other essay
def cov(a, b): return m([sum(w in types[y] for w in cont[x]) / len(cont[x]) for x, y in ((a, b), (b, a))])
# all unordered pairs of essays
def pairs_of(names):
    return list(itertools.combinations(names, 2))
res = {}
for label, names in (("early", E_), ("recent", R_)):
    same = [p for p in pairs_of(names) if COURSE[p[0]] == COURSE[p[1]]]
    diff = [p for p in pairs_of(names) if COURSE[p[0]] != COURSE[p[1]]]
    res[label] = dict(same=[(a[:18], b[:18], round(jac(a, b), 3), round(cov(a, b), 3)) for a, b in same],
                      same_jaccard=m([jac(*p) for p in same]), diff_jaccard=m([jac(*p) for p in diff]),
                      same_cov=m([cov(*p) for p in same]), diff_cov=m([cov(*p) for p in diff]), n_same=len(same), n_diff=len(diff))
    r = res[label]
    print(f"{label}: {r['n_same']} same-course pairs vs {r['n_diff']} different-course pairs")
    for s in r["same"]: print("    same course:", s)
    print(f"    Jaccard same {r['same_jaccard']:.3f} vs diff {r['diff_jaccard']:.3f};   coverage same {r['same_cov']:.3f} vs diff {r['diff_cov']:.3f}")
# pooled test: shuffle course labels within each period, keep course sizes
def stat(course):
    """Test statistic: mean Jaccard of same-course pairs minus different-course pairs, pooled over both periods.
    `course` maps each essay to a course code, so shuffled labels can be passed in for the permutation test."""
    out = []
    for names in (E_, R_):
        for a, b in pairs_of(names):
            out.append((course[a] == course[b], jac(a, b)))
    s = [v for k, v in out if k]; d = [v for k, v in out if not k]
    return m(s) - m(d)
obs = stat(COURSE)
rng = random.Random(156); ge = 0; N = 5000
for _ in range(N):
    c = dict(COURSE)
    for names in (E_, R_):
        labs = [COURSE[n] for n in names]; rng.shuffle(labs)
        c.update(dict(zip(names, labs)))
    ge += stat(c) >= obs - 1e-12
print(f"pooled same-course minus different-course Jaccard = {obs:+.3f}; permutation p = {(1+ge)/(1+N):.4f} ({N} shuffles of course labels within period)")
res["pooled_same_minus_diff_jaccard"] = obs; res["pooled_p"] = (1 + ge) / (1 + N)

# ---- B. does the early/recent coverage gap survive if no essay is covered by a same-course essay?
# training pool for essay t: the other essays of its period that come from a different course
def pool_for(t, names):
    return [x for x in names if x != t and COURSE[x] != COURSE[t]]
sizes = [sum(pred.n_predicted(pred.ESS[x]) for x in pool_for(t, names)) for names in (E_, R_) for t in names]
Ntr = int(min(sizes) // 100 * 100)
rng = random.Random(pred.SEED)
def oov(t, pool):
    """Share of essay t's word tokens never seen in training, averaged over pred.R random training draws from `pool`."""
    r = []
    for _ in range(pred.R):
        parts = pred.draw_training(pool, Ntr, rng)
        seen = {w for p in parts for s in p for w in s}
        r.append(sum(w not in seen for w in toks[t]) / len(toks[t]))
    return m(r)
oE = {t: oov(t, pool_for(t, E_)) for t in E_}
oR = {t: oov(t, pool_for(t, R_)) for t in R_}
d, p, k = pred.perm_test(list(oE.values()), list(oR.values()))
print(f"coverage when training only on DIFFERENT-course essays (N_train={Ntr}): unseen words early {100*m(list(oE.values())):.1f}%, recent {100*m(list(oR.values())):.1f}%, gap {100*d:+.1f} points, exact p = {p:.4f}")
# same comparison using only SAME-course essays is not possible at this size, so also show the matched all-different gap in the original design
res["different_course_only"] = dict(N_train=Ntr, early=m(list(oE.values())), recent=m(list(oR.values())), gap=d, p=p)
json.dump(res, open(os.path.join(sys.argv[2], "course_test_results.json"), "w"), indent=1)
