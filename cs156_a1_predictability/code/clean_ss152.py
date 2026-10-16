import re,sys
raw=open(sys.argv[1],encoding="utf-8").read().split("\n")
blocks=[(r"^\s*Our group will explore",r"^\s*Words:"),
        (r"^\s*This source gives",r"^\s*Words:"),
        (r"^\s*This provides strong",r"^\s*Words:"),
        (r"^\s*This is more of a review",r"^\s*Words:"),
        (r"^\s*A major challenge in pica",r"^\s*Word count:")]
paras=[]
for st,en in blocks:
    cur=None
    for l in raw:
        if cur is None:
            if re.match(st,l): cur=[l.strip()]
        else:
            if re.match(en,l): break
            if l.strip(): cur.append(l.strip())
    assert cur, st
    t=" ".join(cur)
    t=re.sub(r"(\w)- (\w)", lambda m: m.group(0), t)
    paras.append(t)
text="\n\n".join(paras)+"\n"
open(sys.argv[2],"w",encoding="utf-8").write(text)
print(len(re.findall(r"[A-Za-z]+",text)),"words")
