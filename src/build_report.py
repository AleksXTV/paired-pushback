"""Build report/report.pdf from report/report.md using the official Apart template (report/template.docx).

    python src/build_report.py            # needs python-docx and LibreOffice (soffice) for the PDF step

Markdown subset: front matter (title/author/affiliation), "# " -> Heading 2, "### " -> Heading 3, paragraphs,
"- " bullets, "1. " numbered items, **bold**, *italic*, `code`, pipe tables, ![caption](path) figures,
[PILOT_TABLE] / [RESULTS_TABLE] placeholders filled from report/results*.md.
"""
from __future__ import annotations

import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import docx
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "report"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def inline(par, text: str) -> None:
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1]); r.font.name = "Courier New"; r.font.size = Pt(9)
        elif tok.startswith("*") and len(tok) > 2:
            par.add_run(tok[1:-1]).italic = True
        else:
            par.add_run(tok)


def table_rows(lines: list[str]) -> list[list[str]]:
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    return [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]


def load_table(path: Path) -> list[str]:
    if not path.exists():
        return ["(table not generated yet: run src/analyze.py)"]
    return [l for l in path.read_text().splitlines() if l.startswith("|")]


def main() -> None:
    md = (REP / "report.md").read_text()
    fm, body = {}, md
    m = re.match(r"---\n(.*?)\n---\n", md, re.S)
    if m:
        for l in m.group(1).splitlines():
            k, _, v = l.partition(":"); fm[k.strip()] = v.strip().strip('"')
        body = md[m.end():]
    abstract = ""
    am = re.search(r"^# Abstract\n(.*?)(?=^# )", body, re.S | re.M)
    if am:
        abstract = " ".join(am.group(1).split()); body = body[: am.start()] + body[am.end():]
    body = body.replace("[PILOT_TABLE]", "\n".join(load_table(REP / "results_pilot.md")))
    body = body.replace("[RESULTS_TABLE]", "\n".join(load_table(REP / "results.md")))
    body = body.replace("[RESULTS_TABLE_MAIN]", "\n".join(load_table(REP / "results_main.md")))
    body = body.replace("[ROBUSTNESS_TABLE]", "\n".join(load_table(REP / "robustness.md")))

    d = docx.Document(str(REP / "template.docx"))
    # Title block
    t0 = d.tables[0]
    for p in t0.rows[0].cells[0].paragraphs:
        if p.text.strip() == "PROJECT TITLE":
            for r in p.runs[1:]:
                r.text = ""
            p.runs[0].text = fm.get("title", "Untitled")
    # Author block: replace the 6-slot author grid with one centred paragraph
    for tc in list(t0._tbl.iter(W + "tc")):
        txt = "".join(x.text or "" for x in tc.iter(W + "t"))
        if "Author name 1" in txt and "Author name 2" not in txt:  # the innermost author slot
            grid = tc.getparent().getparent()  # tc -> tr -> tbl
            if grid.tag == W + "tbl":
                holder = d.add_paragraph()
                holder.alignment = 1
                r = holder.add_run(fm.get("author", "")); r.bold = True
                holder.add_run("\n" + fm.get("affiliation", ""))
                grid.addprevious(holder._p)
                grid.getparent().remove(grid)
                break
    abs_cell = t0.rows[1].cells[0].tables[1].rows[0].cells[0]
    for p in abs_cell.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    p = abs_cell.paragraphs[0]
    for r in p.runs:
        r.text = ""
    inline(p, abstract)
    for r in p.runs:
        r.italic = False
    # Drop guidance box and everything after the title block
    bodyel = d.element.body
    started = False
    for ch in list(bodyel.iterchildren()):
        if ch is t0._tbl:
            started = True; continue
        if started and ch.tag != W + "sectPr":
            bodyel.remove(ch)

    lines = body.splitlines()
    i = 0
    fig_n = 0
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            i += 1; continue
        if l.startswith("# "):
            d.add_paragraph(l[2:].strip(), style="Heading 2")
        elif l.startswith("### "):
            d.add_paragraph(l[4:].strip(), style="Heading 3")
        elif l.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            rows = table_rows(block)
            tb = d.add_table(rows=len(rows), cols=len(rows[0]))
            tb.style = d.styles["Table Grid"] if "Table Grid" in [s.name for s in d.styles] else tb.style
            for ri, row in enumerate(rows):
                for ci, val in enumerate(row[: len(rows[0])]):
                    cell = tb.cell(ri, ci); cell.text = ""
                    inline(cell.paragraphs[0], val)
                    for r in cell.paragraphs[0].runs:
                        r.font.size = Pt(6.5 if len(rows[0]) > 8 else 8); r.bold = r.bold or ri == 0
            continue
        elif l.startswith("!["):
            mm = re.match(r"!\[(.*?)\]\((.*?)\)", l)
            path = (REP / mm.group(2)) if mm else None
            if path and path.exists():
                wm = re.search(r"\|([\d.]+)$", mm.group(1))  # optional width in inches: ![caption|4.2](path)
                d.add_picture(str(path), width=Inches(float(wm.group(1)) if wm else 6.2))
                d.paragraphs[-1].alignment = 1
            else:
                d.add_paragraph(f"[figure missing: {mm.group(2) if mm else l}]")
        elif re.match(r"^(- |\d+\. )", l):
            p = d.add_paragraph()
            bullet = "• " if l.startswith("- ") else re.match(r"^\d+\. ", l).group(0)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.first_line_indent = Inches(-0.18)
            inline(p, bullet + re.sub(r"^(- |\d+\. )", "", l))
        else:
            para = [l]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|- |\d+\. |\||!\[)", lines[i + 1]):
                i += 1; para.append(lines[i])
            pp = d.add_paragraph(); pp.paragraph_format.space_after = Pt(6)
            inline(pp, " ".join(x.strip() for x in para))
        i += 1
    out = REP / "report.docx"
    d.save(str(out))
    r = subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(REP), str(out)],
                       capture_output=True, text=True, timeout=180)
    print(r.stdout.strip() or r.stderr.strip())
    pdf = REP / "report.pdf"
    if not pdf.exists():
        sys.exit("PDF conversion failed")
    print("wrote", pdf)


if __name__ == "__main__":
    main()
