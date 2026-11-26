"""Build the IEEE Internet of Things Journal Word manuscript from ieee.md (made by tex2md.py).

Layout (IEEE Transactions/Journals Word template conventions):
  * US Letter, top 0.75 in, bottom 1 in, left/right 0.625 in
  * title (24 pt) and author line single column; body in two columns (0.2-in gap)
  * Times New Roman 10 pt body, single spacing, 0.125-in first-line indent
  * section headings: Roman numerals, small caps, centred; subsections: letter, italic
  * equations as native Word equations, numbered at the right column edge
  * figure captions "Fig. n." 8 pt below figures; "TABLE n" + small-caps title above tables
  * full-width figures placed in single-column continuous sections
  * references 8 pt with hanging numbers; author affiliations in an unmarked first-page footnote
usage: python3 make_ieee_docx.py ieee.md WARDEN_IEEE_IoTJ.docx
"""
import copy, os, re, shutil, subprocess, sys, zipfile
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

src, dst = sys.argv[1], sys.argv[2]
raw = dst.replace(".docx", "_raw.docx")
subprocess.run(["pandoc", src, "-f", "markdown", "-o", raw, "--resource-path=."], check=True)
doc = Document(raw)
FONT = "Times New Roman"
GAP = Inches(0.2)

sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(0.75), Inches(1.0)
sec.left_margin = sec.right_margin = Inches(0.625)
sec.header_distance, sec.footer_distance, sec.gutter = Inches(0.4), Inches(0.5), Inches(0)
TEXT_W = sec.page_width - sec.left_margin - sec.right_margin
COL_W = int((TEXT_W - GAP) / 2)


def setfont(obj, size=None, bold=None, italic=None, smallcaps=None):
    f = obj.font
    f.name = FONT
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if smallcaps is not None:
        f.small_caps = smallcaps
    f.color.rgb = RGBColor(0, 0, 0)
    rpr = obj.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)


def fmt(p, align=None, before=0, after=0, first=None, left=None, keep=None):
    pf = p.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), 1.0
    if align is not None:
        p.alignment = align
    if first is not None:
        pf.first_line_indent = first
    if left is not None:
        pf.left_indent = left
    if keep is not None:
        pf.keep_with_next = keep


STY = {s.name: s for s in doc.styles}
for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Footnote Text", "Image Caption", "Caption",
             "Table Caption", "Captioned Figure", "Figure"):
    if name in STY:
        setfont(STY[name], 8 if name == "Footnote Text" else 10)
if "Footnote Text" in STY:
    STY["Footnote Text"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

# ---------------------------------------------------------------- equations: centred, number at column edge
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
tags = re.findall(r"\\tag\{([^}]*)\}", open(src, encoding="utf-8").read())
eq_paras = [p for p in doc.paragraphs if p._element.find(f".//{M}oMathPara") is not None]
assert len(eq_paras) == len(tags), (len(eq_paras), len(tags))
for p, tag in zip(eq_paras, tags):
    mpara = p._element.find(f".//{M}oMathPara")
    omath = mpara.find(f"{M}oMath")
    parent = mpara.getparent(); idx = list(parent).index(mpara); parent.remove(mpara)
    r1 = OxmlElement("w:r"); r1.append(OxmlElement("w:tab"))
    r2 = OxmlElement("w:r"); r2.append(OxmlElement("w:tab"))
    t = OxmlElement("w:t"); t.text = f"({tag})"; r2.append(t)
    for i, node in enumerate([r1, omath, r2]):
        parent.insert(idx + i, node)
    pf = p.paragraph_format
    pf.tab_stops.add_tab_stop(int(COL_W / 2), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(COL_W, WD_TAB_ALIGNMENT.RIGHT)
    fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 3, 3, first=Inches(0))
    for r in p.runs:
        setfont(r, 10)

# ---------------------------------------------------------------- paragraph styling
markers = {}
after_heading = False
for p in doc.paragraphs:
    sty, txt = p.style.name, p.text.strip()
    if txt in ("@@SEC1END@@", "@@WIDESTART@@", "@@WIDEEND@@"):
        markers.setdefault(txt, []).append(p); continue
    if p in eq_paras:
        continue
    has_img = p._element.find(".//" + qn("w:drawing")) is not None
    runs = p.runs
    if sty == "IEEE Title":
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 0, 10, first=Inches(0))
        for r in runs: setfont(r, 24)
    elif sty == "IEEE Authors":
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 0, 14, first=Inches(0))
        for r in runs: setfont(r, 11)
    elif sty == "IEEE Abstract":
        fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 6, first=Inches(0.125))
        for r in runs: setfont(r, 9, bold=True)
    elif sty == "IEEE Heading 1":
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 8, 4, first=Inches(0), keep=True)
        for r in runs:
            setfont(r, 10, bold=False, italic=False, smallcaps=True)
        after_heading = True; continue
    elif sty == "IEEE Heading 2":
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 6, 2, first=Inches(0), keep=True)
        for r in runs: setfont(r, 10, bold=False, italic=True)
        after_heading = True; continue
    elif has_img:
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 4, 2, first=Inches(0), keep=True)
    elif sty == "IEEE Figure Caption":
        fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 8, first=Inches(0))
        for r in runs: setfont(r, 8)
    elif sty == "IEEE Table Number":
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 8, 0, first=Inches(0), keep=True)
        for r in runs: setfont(r, 8)
    elif sty == "IEEE Table Title":
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 0, 3, first=Inches(0), keep=True)
        for r in runs: setfont(r, 8, smallcaps=True)
    elif sty == "IEEE Algorithm Title":
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 8, 2, first=Inches(0), keep=True)
        for r in runs: setfont(r, 9)
    elif sty == "IEEE Algorithm Line":
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 0, 0, first=Inches(-0.2), left=Inches(0.2), keep=True)
        for r in runs: setfont(r, 8.5)
    elif sty == "IEEE List":
        fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 1, 1, first=Inches(-0.2), left=Inches(0.2))
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.2))
        for r in runs: setfont(r, 10)
    elif sty == "IEEE Reference":
        fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 2, first=Inches(-0.3), left=Inches(0.3))
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
        for r in runs: setfont(r, 8)
    else:  # body text and list items
        is_list = p._element.find(".//" + qn("w:numPr")) is not None
        fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 0, first=Inches(0) if is_list else Inches(0.125))
        for r in runs: setfont(r, 10)
    after_heading = False

# algorithm box: rules above the title, below the title and below the last line
def border(p, edges):
    pPr = p._p.get_or_add_pPr()
    old = pPr.find(qn("w:pBdr"))
    if old is not None:
        pPr.remove(old)
    b = OxmlElement("w:pBdr")
    for e in ("top", "bottom"):
        if e in edges:
            el = OxmlElement(f"w:{e}"); el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "6")
            el.set(qn("w:space"), "1"); el.set(qn("w:color"), "000000"); b.append(el)
    # schema: pBdr after keepNext/keepLines/pageBreakBefore/framePr/widowControl/numPr/suppressLineNumbers
    anchor = None
    for tag in ("w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap", "w:overflowPunct",
                "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi", "w:adjustRightInd", "w:snapToGrid",
                "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc"):
        anchor = pPr.find(qn(tag))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addprevious(b)
    else:
        pPr.append(b)


alg = [p for p in doc.paragraphs if p.style.name in ("IEEE Algorithm Title", "IEEE Algorithm Line")]
if alg:
    border(alg[0], ("top", "bottom")); border(alg[-1], ("bottom",))
    alg[-1].paragraph_format.keep_with_next = False
    alg[-1].paragraph_format.space_after = Pt(8)

# ---------------------------------------------------------------- tables: three-line, 7.5 pt
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
    for tag in ("w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription"):
        nxt = tblPr.find(qn(tag))
        if nxt is not None:
            break
    nxt.addprevious(b) if nxt is not None else tblPr.append(b)


# relative column widths of Tables I-VI (column count is checked; otherwise equal widths)
REL = [[1.05, 0.62, 0.28, 0.58, 0.62, 0.72, 0.55],
       [0.40, 0.60],
       [0.47, 0.53],
       [1.15, 0.48, 0.48, 0.48, 0.5, 0.48, 0.5, 0.55],
       [0.8, 1.25, 0.42, 0.45, 0.45, 0.45],
       [1.25, 0.5, 0.5, 0.42, 0.55, 0.55]]
for ti, t in enumerate(doc.tables):
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_borders(t, {"top": 8, "bottom": 8})
    tblPr = t._tbl.tblPr
    # fixed layout at column width; column widths proportional to content length
    total = int(COL_W.inches * 1440 if hasattr(COL_W, "inches") else COL_W / 635) - 40
    ncol = len(t.columns)
    rel = REL[ti] if ti < len(REL) and len(REL[ti]) == ncol else [1] * ncol
    widths = [int(total * r / sum(rel)) for r in rel]
    w = tblPr.find(qn("w:tblW"))
    if w is None:
        w = OxmlElement("w:tblW"); tblPr.append(w)
    w.set(qn("w:type"), "dxa"); w.set(qn("w:w"), str(total))
    lay = tblPr.find(qn("w:tblLayout"))
    if lay is None:
        lay = OxmlElement("w:tblLayout")
        nxt = None
        for tag in ("w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription"):
            nxt = tblPr.find(qn(tag))
            if nxt is not None:
                break
        nxt.addprevious(lay) if nxt is not None else tblPr.append(lay)
    lay.set(qn("w:type"), "fixed")
    grid = t._tbl.find(qn("w:tblGrid"))
    for gc, wd in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(wd))
    for row in t.rows:
        for j, c in enumerate(row.cells):
            tcPr = c._tc.get_or_add_tcPr(); tcW = tcPr.find(qn("w:tcW"))
            if tcW is None:
                tcW = OxmlElement("w:tcW"); tcPr.insert(0, tcW)
            tcW.set(qn("w:type"), "dxa"); tcW.set(qn("w:w"), str(widths[j]))
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), "45"); e.set(qn("w:type"), "dxa"); mar.append(e)
    old = tblPr.find(qn("w:tblCellMar"))
    if old is not None:
        tblPr.remove(old)
    look = tblPr.find(qn("w:tblLook"))
    (look.addprevious(mar) if look is not None else tblPr.append(mar))
    for i, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr(); trPr.append(OxmlElement("w:cantSplit"))
        for j, cell in enumerate(row.cells):
            if i == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                tb = OxmlElement("w:tcBorders"); bt = OxmlElement("w:bottom")
                bt.set(qn("w:val"), "single"); bt.set(qn("w:sz"), "4"); bt.set(qn("w:color"), "000000"); tb.append(bt)
                anchor = None
                for tag in ("w:vMerge", "w:gridSpan", "w:tcW"):
                    anchor = tcPr.find(qn(tag))
                    if anchor is not None:
                        break
                anchor.addnext(tb) if anchor is not None else tcPr.insert(0, tb)
            for p in cell.paragraphs:
                fmt(p, WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER, 0, 0, first=Inches(0),
                    keep=i < len(t.rows) - 1)
                for r in p.runs:
                    setfont(r, 7.5, bold=(i == 0) or None)

# ---------------------------------------------------------------- footer page number
fp = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run()
for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    if kind:
        fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); run._r.append(fc)
    else:
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; run._r.append(it)
setfont(run, 8)

# ---------------------------------------------------------------- sections (1 col title, 2 col body, 1 col wide figures)
body_sect = doc.element.body.find(qn("w:sectPr"))


def make_sect(cols, first=False):
    s = copy.deepcopy(body_sect)
    for e in s.findall(qn("w:type")) + s.findall(qn("w:cols")):
        s.remove(e)
    if not first:
        ty = OxmlElement("w:type"); ty.set(qn("w:val"), "continuous")
        s.find(qn("w:pgSz")).addprevious(ty)
    c = OxmlElement("w:cols"); c.set(qn("w:num"), str(cols)); c.set(qn("w:space"), str(int(GAP.inches * 1440)))
    anchor = s.find(qn("w:docGrid"))
    anchor.addprevious(c) if anchor is not None else s.append(c)
    return s


def end_section_at(marker, cols, first=False):
    """Make the marker paragraph the last paragraph of a section with the given column count."""
    for r in list(marker.runs):
        r._r.getparent().remove(r._r)
    pPr = marker._p.get_or_add_pPr()
    pf = marker.paragraph_format
    pf.space_before = pf.space_after = Pt(0); pf.line_spacing = Pt(1)
    rpr = OxmlElement("w:rPr"); sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "2"); rpr.append(sz)
    pPr.append(rpr)
    pPr.append(make_sect(cols, first))


end_section_at(markers["@@SEC1END@@"][0], 1, first=True)
for p in markers.get("@@WIDESTART@@", []):
    end_section_at(p, 2)
for p in markers.get("@@WIDEEND@@", []):
    end_section_at(p, 1)
# final section: two columns, continuous
for e in body_sect.findall(qn("w:type")) + body_sect.findall(qn("w:cols")):
    body_sect.remove(e)
ty = OxmlElement("w:type"); ty.set(qn("w:val"), "continuous"); body_sect.find(qn("w:pgSz")).addprevious(ty)
c = OxmlElement("w:cols"); c.set(qn("w:num"), "2"); c.set(qn("w:space"), str(int(GAP.inches * 1440)))
dg = body_sect.find(qn("w:docGrid")); dg.addprevious(c) if dg is not None else body_sect.append(c)

# footnote: IEEE first-page affiliation note is unmarked
fn_part = None
for rel in doc.part.rels.values():
    if rel.reltype.endswith("/footnotes"):
        fn_part = rel.target_part
doc.core_properties.title = "Wave Aware Risk Controlled Link Selection for Hybrid LoRa and LEO Satellite Ocean Monitoring IoT"
doc.core_properties.author = "G. Arul Dalton; S. V. J. Mani"
doc.core_properties.keywords = "Maritime IoT; LoRa; LR-FHSS; direct-to-satellite IoT; conformal risk control"
doc.save(dst)
os.remove(raw)


# ---------------------------------------------------------------- XML clean-up (Word-strict ordering) + footnote style
def fix(name, x):
    if name == "word/document.xml":
        x = re.sub(r"<m:mcPr>(<m:mcJc [^>]*/>)(<m:count [^>]*/>)</m:mcPr>", r"<m:mcPr>\2\1</m:mcPr>", x)
        x = re.sub(r"<m:rPr>(<m:sty [^>]*/>)(<m:scr [^>]*/>)</m:rPr>", r"<m:rPr>\2\1</m:rPr>", x)
        x = re.sub(r"<w:pPr>(<w:numPr>.*?</w:numPr>)(<w:pStyle [^>]*/>)", r"<w:pPr>\2\1", x)
        x = re.sub(r"<m:nor/><m:sty [^>]*/>", "<m:nor/>", x)
        x = re.sub(r"(<m:begChr [^>]*/>)(<m:endChr [^>]*/>)(<m:sepChr [^>]*/>)", r"\1\3\2", x)
        # hide the footnote reference mark on the author line (IEEE: unmarked affiliation footnote)
        x = re.sub(r'(<w:r><w:rPr><w:rStyle w:val="FootnoteReference"\s*/>)', r'\1<w:vanish/>', x, count=1)
    if name == "word/footnotes.xml":
        x = re.sub(r'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"\s*/></w:rPr><w:footnoteRef\s*/></w:r>', "", x)
    if name == "word/numbering.xml":
        x = re.sub(r'(<w:nsid w:val=")([0-9A-Fa-f]{1,7})(")', lambda m: m.group(1) + m.group(2).rjust(8, "0") + m.group(3), x)
    if name == "word/styles.xml":
        x = re.sub(r"(<w:color [^>]*/>)(?:&gt;|>)\s*", r"\1", x)
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
        root = etree.fromstring(x.encode("utf-8")); kids = list(root)
        key = lambda e: ORDER.index(etree.QName(e).localname) if etree.QName(e).localname in ORDER else len(ORDER)
        for k in kids: root.remove(k)
        for k in sorted(kids, key=key): root.append(k)
        x = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True).decode("utf-8")
    return x


tmp = dst + ".tmp"
with zipfile.ZipFile(dst) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        if item.filename.endswith("/"):
            continue
        data = zin.read(item.filename)
        if item.filename in ("word/document.xml", "word/numbering.xml", "word/settings.xml", "word/styles.xml",
                             "word/footnotes.xml"):
            data = fix(item.filename, data.decode("utf-8")).encode("utf-8")
        zout.writestr(item, data)
shutil.move(tmp, dst)
print("saved", dst, "| equations:", len(tags), "| tables:", len(doc.tables), "| sections:",
      1 + len(markers.get("@@WIDESTART@@", [])) + len(markers.get("@@WIDEEND@@", [])) + 1)
