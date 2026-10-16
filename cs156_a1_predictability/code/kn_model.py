"""Interpolated Kneser-Ney bigram dropped into the same study loop.  Usage: python3 -I kn_model.py ESSAY_DIR OUT_DIR"""
import importlib.util, itertools, json, math, os, sys
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pred", os.path.join(here, "predictability.py"))
pred = importlib.util.module_from_spec(spec); spec.loader.exec_module(pred)
from collections import Counter

D_GRID = [0.5, 0.8, 0.95]
DELTA_GRID = [0.1, 0.5]
UNI_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
orig_bits, orig_fit = pred.bits, pred.fit_and_alpha


def kn_tables(big):
    cont, n1p = Counter(), Counter()
    for (v, w), c in big.items():
        if c > 0:
            cont[w] += 1
            n1p[v] += 1
    return n1p, cont, sum(cont.values())


def kn_bits(test, tabs, V, D, delta):
    big, ctx, n1p, cont, ntypes = tabs
    tot = k = 0
    for (v, w), c in test[0].items():
        pc = (cont.get(w, 0) + delta) / (ntypes + delta * V)
        cv = ctx.get(v, 0)
        p = pc if cv == 0 else max(big.get((v, w), 0) - D, 0) / cv + D * n1p.get(v, 0) / cv * pc
        tot -= c * math.log2(p)
        k += c
    return tot / k


def my_fit(parts, vocab, order):
    if order == 1:
        pred.ALPHAS = UNI_ALPHAS
        return orig_fit(parts, vocab, 1)
    V = len(vocab)
    pc = [pred.counts(p, vocab) for p in parts]
    tb = sum((c[0] for c in pc), Counter()); tc = sum((c[1] for c in pc), Counter())
    best = None
    for D in D_GRID:
        for delta in DELTA_GRID:
            tot = k = 0.0
            for c in pc:
                rb, rc = pred.minus(tb, c[0]), pred.minus(tc, c[1])
                n1p, cont, nt = kn_tables(rb)
                m = sum(c[0].values())
                tot += kn_bits(c, (rb, rc, n1p, cont, nt), V, D, delta) * m
                k += m
            if best is None or tot / k < best[0]:
                best = (tot / k, (D, delta))
    n1p, cont, nt = kn_tables(tb)
    return best[1], (tb, tc, (n1p, cont, nt), sum(tb.values()))


def my_bits(test, big, ctx, extra, n, V, alpha, order):
    if order == 1:
        return orig_bits(test, big, ctx, extra, n, V, alpha, 1)
    n1p, cont, nt = extra
    D, delta = alpha
    return kn_bits(test, (big, ctx, n1p, cont, nt), V, D, delta)


pred.fit_and_alpha, pred.bits = my_fit, my_bits
if __name__ == "__main__":
    pred.ESSAYS_DIR = sys.argv[1]
    out = {}
    for label, early, recent, mask in [("main", pred.EARLY, pred.RECENT, False),
                                       ("function_words", pred.EARLY, pred.RECENT, True),
                                       ("with_group", pred.EARLY + pred.GROUP_EARLY, pred.RECENT + pred.GROUP_RECENT, False)]:
        info, res, au = pred.run(early, recent, mask, label)
        s = pred.summarize(info, res, au)
        out[label] = s
        t = s["table"]
        print("  Kneser-Ney bigram PP (geometric mean over test essays)")
        print("          test E   test R")
        print(f"  train E {t['E->E']['PP_geo']:7.1f}  {t['E->R']['PP_geo']:7.1f}")
        print(f"  train R {t['R->E']['PP_geo']:7.1f}  {t['R->R']['PP_geo']:7.1f}")
        o = s["own_period_bigram"]
        print(f"  own-period delta (R minus E) = {o['delta_bits']:+.3f} bits/token, exact p = {o['p']:.4f} over {o['n_perm']} splits")
        g = s["bigram_gain"]
        print(f"  KN bigram gain over unigram: early {g['early']:+.3f}, recent {g['recent']:+.3f} bits, delta {g['delta']:+.3f}, p = {g['p']:.4f}")
        print(f"  drift: early model on recent {s['drift_on_recent']:+.3f} bits, recent model on early {s['drift_on_early']:+.3f} bits", flush=True)
    json.dump(out, open(os.path.join(sys.argv[2], "kn_results.json"), "w"), indent=1)
