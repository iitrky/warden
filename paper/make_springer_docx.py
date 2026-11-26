"""Build the Springer Nature style Word manuscript from manuscript.md.

pandoc converts Markdown and LaTeX math to Word with native (editable) equations.
This script then applies the Springer Nature manuscript layout:
  * A4, 2.5 cm margins, Times New Roman 12 pt, 1.5 line spacing, justified text
  * title page: title, authors with affiliation numbers, affiliations, corresponding author, ORCID
  * numbered display equations, with the equation centred and its number right-aligned
  * figure captions below figures, table captions above tables, three-line tables
  * numbered reference list with hanging indent, page numbers in the footer, no line numbers
usage: python make_springer_docx.py manuscript.md manuscript.docx
"""
import re, subprocess, sys, copy
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt, Cm, RGBColor

src, dst = sys.argv[1], sys.argv[2]
md = open(src, encoding="utf-8").read()
tags = re.findall(r"\\tag\{(\d+)\}", "".join(re.findall(r"\$\$(.*?)\$\$", md, re.S)))
raw = dst.replace(".docx", "_raw.docx")
subprocess.run(["pandoc", src, "-o", raw, "--resource-path=.:.."], check=True)
doc = Document(raw)

FONT = "Times New Roman"
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Cm(2.5))
sec.header_distance, sec.footer_distance, sec.gutter = Cm(1.25), Cm(1.25), Cm(0)
TEXT_W = sec.page_width - sec.left_margin - sec.right_margin


def font_of(style_or_run, size=None, bold=None, italic=None, color=True):
    f = style_or_run.font
    f.name = FONT
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color:
        f.color.rgb = RGBColor(0, 0, 0)
    el = style_or_run.element
    rpr = el.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)


STY = {s.name.lower(): s for s in doc.styles}
for name, size, bold in [("normal", 12, None), ("body text", 12, None), ("first paragraph", 12, None),
                         ("compact", 12, None), ("title", 16, True), ("author", 12, None),
                         ("heading 1", 13, True), ("heading 2", 12, True), ("heading 3", 12, True),
                         ("image caption", 10, None), ("caption", 10, None), ("table caption", 10, None),
                         ("abstract", 12, None), ("block text", 12, None)]:
    if name in STY:
        font_of(STY[name], size, bold)
if "source code" in STY:
    STY["source code"].font.name = "Courier New"; STY["source code"].font.size = Pt(9)
if "heading 2" in STY:
    STY["heading 2"].font.italic = False


def set_spacing(p, before=0, after=6, line=1.5):
    pf = p.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = line


# ---------------- numbered display equations ----------------
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
eq_paras = [p for p in doc.paragraphs if p._element.find(f".//{M}oMathPara") is not None]
assert len(eq_paras) == len(tags), (len(eq_paras), len(tags))
for p, tag in zip(eq_paras, tags):
    el = p._element
    mpara = el.find(f".//{M}oMathPara")
    omath = mpara.find(f"{M}oMath")
    parent = mpara.getparent()
    idx = list(parent).index(mpara)
    parent.remove(mpara)
    r1 = OxmlElement("w:r"); r1.append(OxmlElement("w:tab"))
    r2 = OxmlElement("w:r"); r2.append(OxmlElement("w:tab"))
    t = OxmlElement("w:t"); t.text = f"({tag})"; r2.append(t)
    for i, node in enumerate([r1, omath, r2]):
        parent.insert(idx + i, node)
    pf = p.paragraph_format
    pf.tab_stops.add_tab_stop(int(TEXT_W / 2), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(int(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_spacing(p, 6, 6, 1.0)
    for r in p.runs:
        font_of(r, 12)

# ---------------- paragraphs ----------------
in_refs = False
author_lines = [p for p in doc.paragraphs if p.style.name == "Author"]
for k, p in enumerate(author_lines):
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_spacing(p, 0, 2, 1.15)
    for r in p.runs:
        font_of(r, 12 if k == 0 else 10.5, bold=(k == 0), italic=(k > 0 and "E-mail" not in p.text and "ORCID" not in p.text and "Corresponding" not in p.text))
for p in doc.paragraphs:
    sty = p.style.name
    txt = p.text.strip()
    if p in eq_paras or sty == "Author":
        continue
    has_img = p._element.find(".//" + qn("w:drawing")) is not None
    if sty == "Title":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER; set_spacing(p, 0, 12, 1.15)
        for r in p.runs:
            font_of(r, 16, bold=True)
        continue
    if sty.lower().startswith("heading"):
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        set_spacing(p, 12, 6, 1.15); p.paragraph_format.keep_with_next = True
        for r in p.runs:
            font_of(r, 13 if sty.endswith("1") else 12, bold=True, italic=False)
        in_refs = txt.lower().startswith("references")
        continue
    if has_img:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER; set_spacing(p, 6, 2, 1.0)
        p.paragraph_format.keep_with_next = True
        continue
    if sty in ("Image Caption", "Caption"):
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; set_spacing(p, 0, 12, 1.15)
        for r in p.runs:
            font_of(r, 10)
        continue
    if re.match(r"^Table \d+", txt) and p.runs and p.runs[0].bold:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT; set_spacing(p, 10, 4, 1.15)
        p.paragraph_format.keep_with_next = True
        for r in p.runs:
            font_of(r, 10)
        continue
    if sty == "Source Code":
        set_spacing(p, 0, 0, 1.0)
        continue
    if txt.startswith("Algorithm ") and p.runs and p.runs[0].bold:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT; set_spacing(p, 8, 2, 1.15); p.paragraph_format.keep_with_next = True
        continue
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if in_refs else WD_ALIGN_PARAGRAPH.JUSTIFY
    if in_refs:
        set_spacing(p, 0, 3, 1.15)
        p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.8)
        for r in p.runs:
            font_of(r, 10.5)
    else:
        set_spacing(p, 0, 6, 1.5)
        for r in p.runs:
            if r.font.name not in ("Courier New",):
                font_of(r, color=False)

# ---------------- three-line tables ----------------
def tbl_borders(tbl, edges):
    tblPr = tbl._tbl.tblPr
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    b = OxmlElement("w:tblBorders")
    for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{e}")
        if e in edges:
            el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(edges[e])); el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")
        b.append(el)
    nxt = None
    for tagname in ("w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription"):
        nxt = tblPr.find(qn(tagname))
        if nxt is not None:
            break
    if nxt is not None:
        nxt.addprevious(b)
    else:
        tblPr.append(b)


for t in doc.tables:
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_borders(t, {"top": 12, "bottom": 12})
    for i, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        for j, cell in enumerate(row.cells):
            if i == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                tb = OxmlElement("w:tcBorders"); bt = OxmlElement("w:bottom")
                bt.set(qn("w:val"), "single"); bt.set(qn("w:sz"), "6"); bt.set(qn("w:color"), "000000")
                tb.append(bt)
                # schema order inside tcPr: tcW, gridSpan, vMerge, tcBorders, shd, ...
                anchor = None
                for tagname in ("w:vMerge", "w:gridSpan", "w:tcW"):
                    anchor = tcPr.find(qn(tagname))
                    if anchor is not None:
                        break
                if anchor is not None:
                    anchor.addnext(tb)
                else:
                    tcPr.insert(0, tb)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                set_spacing(p, 1, 1, 1.0)
                p.paragraph_format.keep_with_next = i < len(t.rows) - 1
                for r in p.runs:
                    font_of(r, 9, bold=True if i == 0 else None)

# ---------------- footer page numbers ----------------
fp = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run()
for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    if kind:
        fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); run._r.append(fc)
    else:
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; run._r.append(it)
font_of(run, 10)

# page break after the title page (before the Abstract heading) is not used: Springer accepts a running layout.
doc.core_properties.title = "Wave Aware Risk Controlled Link Selection for Hybrid LoRa and LEO Satellite Ocean Monitoring IoT"
doc.core_properties.author = "G. Arul Dalton; S. V. J. Mani"
doc.core_properties.keywords = "Maritime IoT; LoRa; LR-FHSS; direct-to-satellite IoT; conformal risk control; physics-informed learning"
doc.save(dst)
import os
os.remove(raw)
print("saved", dst, "equations numbered:", len(tags))


# ---------------- schema clean-up of pandoc's raw XML (Word-strict ordering) ----------------
import zipfile, shutil
def _fix_xml(name, x):
    if name == "word/document.xml":
        x = re.sub(r"<m:mcPr>(<m:mcJc [^>]*/>)(<m:count [^>]*/>)</m:mcPr>", r"<m:mcPr>\2\1</m:mcPr>", x)
        x = re.sub(r"<m:rPr>(<m:sty [^>]*/>)(<m:scr [^>]*/>)</m:rPr>", r"<m:rPr>\2\1</m:rPr>", x)
        x = re.sub(r"<w:pPr>(<w:numPr>.*?</w:numPr>)(<w:pStyle [^>]*/>)", r"<w:pPr>\2\1", x)
        x = re.sub(r"<m:nor/><m:sty [^>]*/>", "<m:nor/>", x)
        x = re.sub(r"(<m:begChr [^>]*/>)(<m:endChr [^>]*/>)(<m:sepChr [^>]*/>)", r"\1\3\2", x)
    if name == "word/numbering.xml":
        x = re.sub(r'(<w:nsid w:val=")([0-9A-Fa-f]{1,7})(")', lambda m: m.group(1) + m.group(2).rjust(8, "0") + m.group(3), x)
    if name == "word/styles.xml":
        x = re.sub(r"(<w:color [^>]*/>)(?:&gt;|>)\s*", r"\1", x)
        # qFormat must precede pPr/rPr inside a style definition
        x = re.sub(r"(<w:pPr>(?:(?!</w:pPr>).)*</w:pPr>)(<w:qFormat/>)", r"\2\1", x, flags=re.S)
        x = re.sub(r"(<w:jc [^>]*/>)(<w:spacing [^>]*/>)", r"\2\1", x)
        x = re.sub(r"<w:tcPr>(<w:vAlign [^>]*/>)(<w:tcBorders>.*?</w:tcBorders>)</w:tcPr>", r"<w:tcPr>\2\1</w:tcPr>", x)
    if name == "word/settings.xml":
        x = re.sub(r"<w:embedSystemFonts\s*/>", "", x)
        from lxml import etree
        ORDER = ["writeProtection", "view", "zoom", "embedSystemFonts", "proofState", "stylePaneFormatFilter",
                 "doNotTrackMoves", "defaultTabStop", "drawingGridHorizontalSpacing", "drawingGridVerticalSpacing",
                 "displayHorizontalDrawingGridEvery", "displayVerticalDrawingGridEvery", "characterSpacingControl",
                 "savePreviewPicture", "footnotePr", "endnotePr", "compat", "docVars", "rsids", "mathPr",
                 "themeFontLang", "clrSchemeMapping", "shapeDefaults", "decimalSymbol", "listSeparator"]
        root = etree.fromstring(x.encode("utf-8"))
        kids = list(root)
        key = lambda e: ORDER.index(etree.QName(e).localname) if etree.QName(e).localname in ORDER else len(ORDER)
        for k in kids:
            root.remove(k)
        for k in sorted(kids, key=key):
            root.append(k)
        x = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True).decode("utf-8")
    return x


tmp = dst + ".tmp"
with zipfile.ZipFile(dst) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        if item.filename.endswith("/"):
            continue
        data = zin.read(item.filename)
        if item.filename in ("word/document.xml", "word/numbering.xml", "word/settings.xml", "word/styles.xml"):
            data = _fix_xml(item.filename, data.decode("utf-8")).encode("utf-8")
        zout.writestr(item, data)
shutil.move(tmp, dst)
print("schema clean-up done")
