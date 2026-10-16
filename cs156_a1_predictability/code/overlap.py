"""Do recent essays share less vocabulary with each other than early essays do?
Usage: python3 -I overlap.py ESSAY_DIR OUT_DIR   (reuses predictability.py)"""
import importlib.util, itertools, json, os, random, sys
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pred", os.path.join(here, "predictability.py"))
pred = importlib.util.module_from_spec(spec); spec.loader.exec_module(pred)

pred.ESS = {n: pred.essay_sents(n, False) for n in pred.EARLY + pred.RECENT}
ESS = pred.ESS
def words(name):
    return [w for s in ESS[name] for w in s if w not in ("<s>", "</s>")]
cnt = {n: words(n) for n in ESS}

# training size exactly as in the study
pools = []
for names in (pred.EARLY, pred.RECENT):
    pools += [sum(pred.n_predicted(ESS[x]) for x in names if x != t) for t in names]
N = int(min(pools) // 100 * 100)
rng = random.Random(pred.SEED)

def oov_rate(test, pool):
    rates = []
    for _ in range(pred.R):
        parts = pred.draw_training(pool, N, rng)
        seen = {w for p in parts for s in p for w in s}
        toks = cnt[test]
        rates.append(sum(w not in seen for w in toks) / len(toks))
    return sum(rates) / len(rates)

res = {"E->E": {}, "R->R": {}, "E->R": {}, "R->E": {}}
for t in pred.EARLY:
    res["E->E"][t] = oov_rate(t, [x for x in pred.EARLY if x != t])
    res["R->E"][t] = oov_rate(t, pred.RECENT)
for t in pred.RECENT:
    res["R->R"][t] = oov_rate(t, [x for x in pred.RECENT if x != t])
    res["E->R"][t] = oov_rate(t, pred.EARLY)

m = pred.mean
print(f"training size N={N} tokens; share of a held-out essay's word tokens never seen in training")
for k in ("E->E", "R->R", "E->R", "R->E"):
    print(f"  train {k[0]} -> test {k[3]}: mean {100*m(list(res[k].values())):.1f}%")
d, p, _ = pred.perm_test(list(res["E->E"].values()), list(res["R->R"].values()))
print(f"  own-period OOV, recent minus early: {100*d:+.1f} points, exact p = {p:.4f}")

# pairwise content-word overlap on equal-length samples (first 740 tokens of every essay)
M = 740
fw = pred.FUNCTION_WORDS
def vocab(name):
    return {w for w in cnt[name][:M] if w not in fw and w != "<num>"}
V = {n: vocab(n) for n in ESS}
def jac(a, b):
    return len(V[a] & V[b]) / len(V[a] | V[b])
def avg(pairs):
    return m([jac(a, b) for a, b in pairs])
E_pairs = list(itertools.combinations(pred.EARLY, 2))
R_pairs = list(itertools.combinations(pred.RECENT, 2))
X_pairs = list(itertools.product(pred.EARLY, pred.RECENT))
print(f"content-word Jaccard overlap (first {M} tokens per essay)")
print(f"  early-early {avg(E_pairs):.3f} ({len(E_pairs)} pairs)   recent-recent {avg(R_pairs):.3f} ({len(R_pairs)} pairs)   early-recent {avg(X_pairs):.3f} ({len(X_pairs)} pairs)")

# does coverage track the perplexity gap?  correlate per-essay OOV with per-essay bits (own-period models)
main = json.load(open(os.path.join(sys.argv[2], "results.json")))["main"]["per_essay_bits"]
xs, ys = [], []
for k in ("E->E", "R->R"):
    for t in res[k]:
        xs.append(res[k][t]); ys.append(main[k][t])
mx, my = m(xs), m(ys)
r = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
print(f"correlation across all 15 essays between OOV rate and bits/token: r = {r:.2f}")
json.dump(dict(N=N, oov=res, jaccard=dict(early=avg(E_pairs), recent=avg(R_pairs), cross=avg(X_pairs)), r=r),
          open(os.path.join(sys.argv[2], "overlap_results.json"), "w"), indent=1)
