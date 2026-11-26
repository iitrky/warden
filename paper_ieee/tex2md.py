"""Convert main.tex (IEEE IoT-J version) into pandoc Markdown for the Word manuscript.

All cross-reference numbers (equations, figures, tables, sections) are read from main.aux,
and citation numbers from the order of refs.tex, so the Word file matches the PDF exactly.
usage: python3 tex2md.py  -> writes ieee.md
"""
import re

tex = open("main.tex", encoding="utf-8").read()
aux = open("main.aux", encoding="utf-8").read()
refs = open("refs.tex", encoding="utf-8").read()

# ---------------------------------------------------------------- numbers
labels = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux)}
bibkeys = re.findall(r"\\bibitem\{(\w+)\}", refs)
citenum = {k: i + 1 for i, k in enumerate(bibkeys)}


def compress(ns):
    ns = sorted(set(ns)); out = []; i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[i]) if j == i else (f"{ns[i]}, {ns[j]}" if j == i + 1 else f"{ns[i]}–{ns[j]}"))
        i = j + 1
    return ", ".join(out)


def cite(m):
    return "[" + compress([citenum[k.strip()] for k in m.group(1).split(",")]) + "]"


# ---------------------------------------------------------------- text-level LaTeX -> Markdown
ACC = {r"\'o": "ó", r"\'e": "é", r"\'a": "á", r'\"o': "ö", r'\"u': "ü", r"\`e": "è", r"{\o}": "ø",
       r"{\l}": "ł", r"{\i}": "ı", r"\"O": "Ö", r'\"O': "Ö"}


def inline(s):
    """Convert LaTeX text (outside display math) to Markdown, leaving $...$ math intact."""
    parts = re.split(r"(\$[^$]+\$)", s)
    out = []
    for k, p in enumerate(parts):
        if k % 2 == 1:  # math
            p = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + labels[m.group(1)] + ")", p)
            out.append(p); continue
        p = re.sub(r"\\cite\{([^}]*)\}", cite, p)
        p = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + labels[m.group(1)] + ")", p)
        p = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels[m.group(1)], p)
        p = re.sub(r"\\IEEEPARstart\{(\w)\}\{(\w+)\}", lambda m: m.group(1) + m.group(2), p)
        p = re.sub(r"\\url\{([^}]*)\}", r"<\1>", p)
        for _ in range(2):
            p = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", p)
            p = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", p)
        for a, b in ACC.items():
            p = p.replace(a, b)
        p = (p.replace("``", "“").replace("''", "”").replace("\\%", "%").replace("\\,", "\u2009")
               .replace("vs.\\ ", "vs. ").replace("\\hfill", " ").replace("\\checkmark", "✓")
               .replace("---", "—").replace("--", "–").replace("~", "\u00a0").replace("\\&", "&"))
        p = p.replace(r"\ ", " ")
        out.append(p)
    return "".join(out)


body = tex.split(r"\maketitle", 1)[1].split(r"\input{refs.tex}", 1)[0]


def move_block(src, start_pat, before_pat=None, after_para_pat=None):
    i = src.index(start_pat); end = r"\end{figure*}" if start_pat.startswith(r"\begin{figure*}") else r"\end{figure}"
    j = src.index(end, i) + len(end); blk = src[i:j]; src = src[:i] + src[j:]
    if before_pat:
        k = src.index(before_pat)
    else:
        k = src.index(after_para_pat); k = src.index("\n\n", k)
    return src[:k] + "\n\n" + blk + "\n\n" + src[k:]


body = move_block(body, r"\begin{figure}[!t]" + "\n" + r"\centering\includegraphics[width=\columnwidth]{figs/fig_tradeoff.pdf}",
                  after_para_pat=r"Fig.~\ref{fig:tradeoff} varies")
body = move_block(body, r"\begin{figure*}[!t]" + "\n" + r"\centering\includegraphics[width=\textwidth]{figs/fig_results.pdf}",
                  before_pat=r"\subsection{Physics Prior")
title = re.search(r"\\title\{(.*?)\}\n", tex).group(1)
def balanced(src, start):
    """Return the content of the {...} group that opens at src[start]."""
    depth = 0
    for i in range(start, len(src)):
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        if depth == 0:
            return src[start + 1:i]
thanks = [inline(balanced(tex, m.end() - 1)) for m in re.finditer(r"\\thanks\{", tex)]

md = []
md.append("---\nlang: en-US\n---\n")
# math macros used in the paper (pandoc latex_macros extension expands them inside math)
md.append(r"\newcommand{\Hs}{H_s}" + "\n" + r"\newcommand{\Hhat}{\hat H_s}" + "\n" + r"\newcommand{\defer}{\varnothing}" + "\n")
md.append(f'::: {{custom-style="IEEE Title"}}\n{title}\n:::\n')
fn = " ".join(thanks)
md.append(f'::: {{custom-style="IEEE Authors"}}\nG. Arul Dalton, *Member, IEEE*, and S. V. J. Mani^[{fn}]\n:::\n')
md.append("@@SEC1END@@\n")

sec_n = 0; sub_n = 0
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
eq_counter = 0


def table_to_md(tab_src, caption, label):
    head = re.search(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\}", tab_src, re.S).group(1)
    head = head.replace("\\toprule", "").replace("\\bottomrule", "").replace("\\midrule", "\n\\\\MIDRULE\\\\")
    rows = [r.strip() for r in re.split(r"\\\\", head) if r.strip() and r.strip() != "MIDRULE"]
    cells = []
    for r in rows:
        r = re.sub(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", r"\1", r)
        cells.append([inline(c.strip()).replace("|", "\\|") for c in r.split("&")])
    ncol = max(len(c) for c in cells)
    cells = [c + [""] * (ncol - len(c)) for c in cells]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "|".join(["---"] * ncol) + "|"]
    lines += ["| " + " | ".join(c) + " |" for c in cells[1:]]
    num = labels[label]
    return (f'::: {{custom-style="IEEE Table Number"}}\nTABLE {num}\n:::\n\n'
            f'::: {{custom-style="IEEE Table Title"}}\n{inline(caption)}\n:::\n\n' + "\n".join(lines) + "\n")


def figure_to_md(fsrc, wide):
    cap = re.search(r"\\caption\{(.*)\}\\label\{([^}]*)\}", fsrc, re.S)
    caption, label = cap.group(1), cap.group(2)
    m = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", fsrc)
    img = m.group(1).replace(".pdf", ".png") if m else "figs/fig_pipeline.png"
    width = "7.1in" if wide else "3.45in"
    num = labels[label]
    out = f"![]({img}){{width={width}}}\n\n::: {{custom-style=\"IEEE Figure Caption\"}}\nFig. {num}. {inline(caption)}\n:::\n"
    if wide:
        out = "@@WIDESTART@@\n\n" + out + "\n@@WIDEEND@@\n"
    return out


def algorithm_to_md(asrc):
    title = re.search(r"\\caption\{(.*?)\}\\label", asrc).group(1)
    lines = []
    req = re.search(r"\\Require (.*?)\n", asrc).group(1)
    lines.append(f"**Require:** {inline(req)}")
    n = 0; depth = 0
    for raw in re.findall(r"\\(State|For|EndFor)(.*?)\n", asrc):
        kind, txt = raw
        txt = txt.strip()
        if kind == "For":
            txt = txt[1:-1]
        if kind == "EndFor":
            n += 1; depth -= 1; lines.append(f"{n}:{'  ' * depth} **end for**"); continue
        n += 1
        if kind == "For":
            lines.append(f"{n}: **for** {inline(txt)} **do**"); depth += 1; continue
        t = inline(txt).replace("\\textbf{if}", "**if**").replace("\\textbf{else}", "**else**")
        t = re.sub(r"\*\*if\*\*|\*\*else\*\*", lambda m: m.group(0), t)
        lines.append(f"{n}:{'\u2003' * depth} {t}")
    blk = f'::: {{custom-style="IEEE Algorithm Title"}}\n**Algorithm 1** {inline(title)}\n:::\n\n'
    blk += "".join(f'::: {{custom-style="IEEE Algorithm Line"}}\n{l}\n:::\n\n' for l in lines)
    return blk


# split body into blocks we handle
pos = 0
tokens = re.split(r"(\\begin\{(?:abstract|IEEEkeywords|enumerate|table|figure\*?|algorithm|equation)\}.*?\\end\{(?:abstract|IEEEkeywords|enumerate|table|figure\*?|algorithm|equation)\})", body, flags=re.S)
for tok in tokens:
    if tok.startswith(r"\begin{abstract}"):
        t = inline(tok.split("}", 1)[1].rsplit(r"\end{abstract}", 1)[0].strip())
        md.append(f'::: {{custom-style="IEEE Abstract"}}\n***Abstract*—{t}**\n:::\n'); continue
    if tok.startswith(r"\begin{IEEEkeywords}"):
        t = inline(tok.split("}", 1)[1].rsplit(r"\end{IEEEkeywords}", 1)[0].strip())
        md.append(f'::: {{custom-style="IEEE Abstract"}}\n***Index Terms*—{t}**\n:::\n'); continue
    if tok.startswith(r"\begin{enumerate}"):
        items = re.findall(r"\\item (.*?)(?=\\item|\\end\{enumerate\})", tok, re.S)
        md.append("".join(f'::: {{custom-style="IEEE List"}}\n{i+1}\\)\t{inline(x.strip())}\n:::\n\n' for i, x in enumerate(items))); continue
    if tok.startswith(r"\begin{table}"):
        cap = re.search(r"\\caption\{(.*?)\}\\label\{([^}]*)\}", tok)
        md.append(table_to_md(tok, cap.group(1), cap.group(2))); continue
    if tok.startswith(r"\begin{figure"):
        md.append(figure_to_md(tok, tok.startswith(r"\begin{figure*}"))); continue
    if tok.startswith(r"\begin{algorithm}"):
        md.append(algorithm_to_md(tok)); continue
    if tok.startswith(r"\begin{equation}"):
        eq = tok[len(r"\begin{equation}"):-len(r"\end{equation}")]
        lab = re.search(r"\\label\{([^}]*)\}", eq).group(1)
        eq = re.sub(r"\\label\{[^}]*\}", "", eq).strip()
        eq = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + labels[m.group(1)] + ")", eq)
        eq = eq.replace("\\textstyle", "")
        md.append(f"$$ {eq} \\tag{{{labels[lab]}}} $$\n"); continue
    # plain text with headings
    for para in re.split(r"\n\s*\n", tok):
        para = para.strip()
        if not para:
            continue
        hs = re.match(r"\\section\{(.*?)\}(\\label\{[^}]*\})?\s*(.*)", para, re.S)
        hss = re.match(r"\\subsection\{(.*?)\}(\\label\{[^}]*\})?\s*(.*)", para, re.S)
        if hs:
            sec_n += 1; sub_n = 0
            md.append(f'::: {{custom-style="IEEE Heading 1"}}\n{ROMAN[sec_n]}\\. {inline(hs.group(1))}\n:::\n')
            para = hs.group(3).strip()
        elif hss:
            sub_n += 1
            md.append(f'::: {{custom-style="IEEE Heading 2"}}\n{chr(64 + sub_n)}\\. {inline(hss.group(1))}\n:::\n')
            para = hss.group(3).strip()
        if para:
            md.append(inline(" ".join(para.split("\n"))) + "\n")

# ---------------------------------------------------------------- references
md.append('::: {custom-style="IEEE Heading 1"}\nReferences\n:::\n')
items = re.findall(r"\\bibitem\{(\w+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})", refs, re.S)
for k, txt in items:
    t = inline(" ".join(txt.split()))
    t = t.replace("\\footnotesize", "").strip()
    md.append(f'::: {{custom-style="IEEE Reference"}}\n[{citenum[k]}]\t{t}\n:::\n')

out = "\n".join(md)
assert "\\cite" not in out and "\\ref{" not in out and "\\eqref" not in out, "unresolved reference"
open("ieee.md", "w", encoding="utf-8").write(out)
print("wrote ieee.md:", len(out.split()), "words;", out.count("\\tag{"), "equations;", len(items), "references")
