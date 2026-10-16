"""Has my writing become more predictable?  Early vs recent, add-alpha bigram models.
Usage: python3 -I predictability.py ESSAY_DIR OUT_DIR
"""
import itertools, json, math, os, random, re, sys
from collections import Counter

ESSAYS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

EARLY = ["college_persistence_fall2023", "sf_marina_food_bank_fall2023", "kenya_democratic_machine_fall2023",
         "food_security_part2_fall2023", "udemy_statistical_inference_fall2023", "biases_spring2024",
         "water_fleas_spring2024", "surveillance_ethics_spring2024", "freshman_reflection_spring2024"]
RECENT = ["climate_experience_fall2025", "brain_criticality_fall2025", "pulse_oximetry_fall2025",
          "pm25_resolution_review_spring2026", "huntingtons_news_spring2026", "pica_playgrounds_lba_spring2026"]
GROUP_EARLY = ["procrastination_fall2023_GROUP", "scientific_proposal_part2_fall2023_GROUP", "sf_transit_survey_spring2024_GROUP"]
GROUP_RECENT = ["airport_queue_fall2025_GROUP"]

ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0]
R = 20          # resamples of the training essays
MIN_COUNT = 2   # vocabulary threshold, pooled over the solo essays
SEED = 156

FUNCTION_WORDS = set("""a about above across after again against all almost along also although always am among an and another any are
aren't around as at be because been before being below between both but by can cannot could did do does doing done down during each either
enough even ever every few for from further had has have having he her here hers herself him himself his how however i if in into is isn't
it its itself just least less like many may me might more most much must my myself neither no nor not of off often on once one only onto or
other our ours ourselves out over own per rather same shall she should since so some such than that the their theirs them themselves then
there therefore these they this those through thus to too toward towards under until up upon us very was we were what when where whether
which while who whom whose why will with within without would yet you your yours yourself yourselves
don't doesn't didn't won't wouldn't can't couldn't shouldn't it's that's there's what's""".split())


def read(name):
    return open(os.path.join(ESSAYS, f"essay_{name}.txt"), encoding="utf-8").read()


CITE = re.compile(r"\([^()]*\b(?:19|20)\d{2}[a-z]?[^()]*\)")


def sentences(text):
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = CITE.sub(" ", text)                     # parenthetical citations are not my voice
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def tokens(sentence, mask):
    sentence = re.sub(r"\d+(?:[.,]\d+)*", " <num> ", sentence)
    words = re.findall(r"<num>|[a-z]+(?:'[a-z]+)?", sentence)
    if mask:
        words = [w if (w in FUNCTION_WORDS or w == "<num>") else "<w>" for w in words]
    return ["<s>"] + words + ["</s>"]


def essay_sents(name, mask):
    return [tokens(s, mask) for s in sentences(read(name))]


def n_predicted(sents):
    return sum(len(s) - 1 for s in sents)


def counts(sents, vocab):
    """bigram, context and unigram counts over unk-mapped sentences"""
    big, ctx, uni = Counter(), Counter(), Counter()
    for s in sents:
        s = [w if w in vocab else "<unk>" for w in s]
        for v, w in zip(s, s[1:]):
            big[(v, w)] += 1
            ctx[v] += 1
            uni[w] += 1
    return big, ctx, uni


def bits(test, big, ctx, uni, n, V, alpha, order):
    """mean surprisal in bits per predicted token of `test` (a counts triple) under an add-alpha model"""
    tot = k = 0
    for (v, w), c in test[0].items():
        if order == 2:
            p = (big.get((v, w), 0) + alpha) / (ctx.get(v, 0) + alpha * V)
        else:
            p = (uni.get(w, 0) + alpha) / (n + alpha * V)
        tot -= c * math.log2(p)
        k += c
    return tot / k


def minus(total, part):
    return {k: total[k] - part.get(k, 0) for k in total}


def fit_and_alpha(parts, vocab, order):
    """alpha chosen by leave-one-essay-out over the essays in the sample; returns (alpha, count tables)"""
    V = len(vocab)
    pc = [counts(p, vocab) for p in parts]
    tb = sum((c[0] for c in pc), Counter()); tc = sum((c[1] for c in pc), Counter()); tu = sum((c[2] for c in pc), Counter())
    n = sum(tb.values())
    best = None
    for a in ALPHAS:
        tot = k = 0.0
        for c in pc:
            m = sum(c[0].values())
            tot += bits(c, minus(tb, c[0]), minus(tc, c[1]), minus(tu, c[2]), n - m, V, a, order) * m
            k += m
        if best is None or tot / k < best[0]:
            best = (tot / k, a)
    return best[1], (tb, tc, tu, n)


def draw_training(pool, n_train, rng):
    """random essays, in random order, until n_train predicted tokens; last essay cut at a sentence boundary"""
    order = pool[:]
    rng.shuffle(order)
    parts, have = [], 0
    for name in order:
        sents = ESS[name]
        take = []
        for s in sents:
            if have + len(s) - 1 > n_train:
                break
            take.append(s)
            have += len(s) - 1
        if take:
            parts.append(take)
        if have >= n_train - 15:
            break
    return parts


def run(early, recent, mask, label):
    global ESS
    ESS = {n: essay_sents(n, mask) for n in early + recent}
    solo = [n for n in EARLY + RECENT if n in ESS]
    cnt = Counter(w for n in solo for s in ESS[n] for w in s)   # shared vocabulary from the solo essays only
    vocab = {w for w, c in cnt.items() if c >= MIN_COUNT} | {"<s>", "</s>", "<unk>"}
    V = len(vocab)
    pools = []
    for names in (early, recent):
        pools += [sum(n_predicted(ESS[x]) for x in names if x != t) for t in names]
    n_train = int(min(pools) // 100 * 100)
    print(f"[{label}] |V|={V}  N_train={n_train}  early essays={len(early)}  recent essays={len(recent)}", flush=True)
    test_counts = {n: counts(ESS[n], vocab) for n in ESS}
    info = dict(label=label, vocab=V, n_train=n_train, tokens={n: n_predicted(ESS[n]) for n in ESS})
    rng = random.Random(SEED)
    res, alphas_used = {}, Counter()
    for tr, trn in (("E", early), ("R", recent)):
        te_out, ten_out = ("R", recent) if tr == "E" else ("E", early)
        for order in (1, 2):
            res[(tr, tr, order)], res[(tr, te_out, order)] = {}, {}
        shared = [draw_training(trn, n_train, rng) for _ in range(R)]   # draws for essays outside the training period
        for order in (1, 2):
            acc = {t: [] for t in ten_out}
            for parts in shared:
                a, (b_, c_, u_, n_) = fit_and_alpha(parts, vocab, order)
                alphas_used[(tr, order, a)] += 1
                for t in ten_out:
                    acc[t].append(bits(test_counts[t], b_, c_, u_, n_, V, a, order))
            for t in ten_out:
                res[(tr, te_out, order)][t] = sum(acc[t]) / R
        for t in trn:                                                   # own-period essays are left out of their own pool
            draws = [draw_training([x for x in trn if x != t], n_train, rng) for _ in range(R)]
            for order in (1, 2):
                hs = []
                for parts in draws:
                    a, (b_, c_, u_, n_) = fit_and_alpha(parts, vocab, order)
                    alphas_used[(tr, order, a)] += 1
                    hs.append(bits(test_counts[t], b_, c_, u_, n_, V, a, order))
                res[(tr, tr, order)][t] = sum(hs) / R
    return info, res, alphas_used


def mean(x):
    return sum(x) / len(x)


def perm_test(hE, hR):
    """exact two-sided permutation test on mean(H_R) - mean(H_E)"""
    vals = list(hE) + list(hR)
    n, k = len(vals), len(hR)
    obs = mean(hR) - mean(hE)
    tot = sum(vals)
    ge = cnt = 0
    for idx in itertools.combinations(range(n), k):
        sR = sum(vals[i] for i in idx)
        d = sR / k - (tot - sR) / (n - k)
        cnt += 1
        if abs(d) >= abs(obs) - 1e-12:
            ge += 1
    return obs, ge / cnt, cnt


def summarize(info, res, alphas_used):
    out = dict(info=info)
    pp = lambda H: 2 ** H
    table = {}
    for tr in "ER":
        for te in "ER":
            h = res[(tr, te, 2)]
            table[f"{tr}->{te}"] = dict(mean_bits=mean(list(h.values())),
                                        PP_geo=pp(mean(list(h.values()))),
                                        PP_arith=mean([pp(v) for v in h.values()]))
    out["table"] = table
    hE, hR = res[("E", "E", 2)], res[("R", "R", 2)]
    d, p, ncomb = perm_test(list(hE.values()), list(hR.values()))
    out["own_period_bigram"] = dict(delta_bits=d, p=p, n_perm=ncomb)
    # bigram gain over unigram (estimates mutual information between neighbouring words)
    gE = {t: res[("E", "E", 1)][t] - res[("E", "E", 2)][t] for t in hE}
    gR = {t: res[("R", "R", 1)][t] - res[("R", "R", 2)][t] for t in hR}
    dg, pg, _ = perm_test(list(gE.values()), list(gR.values()))
    out["bigram_gain"] = dict(early=mean(list(gE.values())), recent=mean(list(gR.values())), delta=dg, p=pg)
    # drift: does a model trained on the other period score worse?
    out["drift_on_recent"] = mean(list(res[("E", "R", 2)].values())) - mean(list(res[("R", "R", 2)].values()))
    out["drift_on_early"] = mean(list(res[("R", "E", 2)].values())) - mean(list(res[("E", "E", 2)].values()))
    out["per_essay_bits"] = {f"{tr}->{te}": res[(tr, te, 2)] for tr in "ER" for te in "ER"}
    out["alpha_chosen"] = {f"{k[0]}-order{k[1]}-a{k[2]}": v for k, v in sorted(alphas_used.items())}
    return out


if __name__ == "__main__":
    allres = {}
    for label, early, recent, mask in [
        ("main", EARLY, RECENT, False),
        ("function_words", EARLY, RECENT, True),
        ("with_group", EARLY + GROUP_EARLY, RECENT + GROUP_RECENT, False),
    ]:
        info, res, au = run(early, recent, mask, label)
        allres[label] = summarize(info, res, au)
        s = allres[label]
        t = s["table"]
        print("  PP (geometric mean over test essays, bigram)")
        print("          test E   test R")
        print(f"  train E {t['E->E']['PP_geo']:7.1f}  {t['E->R']['PP_geo']:7.1f}")
        print(f"  train R {t['R->E']['PP_geo']:7.1f}  {t['R->R']['PP_geo']:7.1f}")
        o = s["own_period_bigram"]
        print(f"  own-period delta (R minus E) = {o['delta_bits']:+.3f} bits/token, exact p = {o['p']:.4f} over {o['n_perm']} splits")
        g = s["bigram_gain"]
        print(f"  bigram gain over unigram: early {g['early']:.3f}, recent {g['recent']:.3f}, delta {g['delta']:+.3f}, p = {g['p']:.4f}")
        print(f"  drift: early model on recent essays {s['drift_on_recent']:+.3f} bits, recent model on early essays {s['drift_on_early']:+.3f} bits")
    json.dump(allres, open(os.path.join(OUT, "results.json"), "w"), indent=1)
