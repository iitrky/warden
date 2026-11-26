"""Order bibitems by first citation in main.tex (IEEE numbering rule) and write refs.tex."""
import re
tex = open('main.tex').read().split(r'\begin{document}')[1]
src = open('refs_src.tex').read()
items = {m.group(1): m.group(0).strip() for m in re.finditer(r'\\bibitem\{(\w+)\}.*?(?=\\bibitem|\Z)', src, re.S)}
order = []
for grp in re.findall(r'\\cite\{([^}]*)\}', tex):
    for k in grp.split(','):
        k = k.strip()
        if k not in order: order.append(k)
missing = [k for k in items if k not in order]; undefined = [k for k in order if k not in items]
assert not undefined, undefined
print('cited', len(order), 'uncited', missing)
with open('refs.tex', 'w') as f:
    f.write('\\begin{thebibliography}{99}\n\\footnotesize\n')
    for k in order + missing: f.write(items[k] + '\n\n')
    f.write('\\end{thebibliography}\n')
