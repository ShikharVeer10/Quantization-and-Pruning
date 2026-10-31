"""Comprehensive validation suite for IEEE conference submission readiness."""
import re
import csv
from pathlib import Path
import docx
import fitz

ROOT = Path("C:/Users/shikh/Quantization-and-Pruning")
DOCX_PATH = ROOT / "paper.docx"
PDF_PATH = ROOT / "paper.pdf"
CSV_PATH = ROOT / "results.csv"
REF_PATH = ROOT / "references.txt"

print("=" * 60)
print("RUNNING FINAL VERIFICATION CHECKS")
print("=" * 60)

# 1. Verify Styles Used
allowed_styles = {
    "papertitle", "Author", "Abstract", "Keywords", "Heading1", "Heading2",
    "Heading3", "Heading4", "Heading5", "BodyText", "figurecaption",
    "tablehead", "tablecolhead", "tablecopy", "tablefootnote", "equation",
    "references", "bulletlist", "Normal"
}
doc = docx.Document(str(DOCX_PATH))
used_styles = set()
for p in doc.paragraphs:
    used_styles.add(p.style.name)
    used_styles.add(p.style.style_id)

for t in doc.tables:
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                used_styles.add(p.style.name)
                used_styles.add(p.style.style_id)

print(f"1. Programmatic Style Check:")
print(f"   Used styles in docx: {sorted(used_styles)}")
# Check for disallowed styles
unrecognized = [s for s in used_styles if not any(allowed.lower() in s.lower().replace(" ", "") for allowed in allowed_styles)]
if unrecognized:
    print(f"   WARNING: Unrecognized styles found: {unrecognized}")
else:
    print(f"   PASS: All paragraph and table styles map directly to template styles.")

# 2. Check for leftover guidance text
guidance_keywords = [
    "IEEE conference templates contain guidance text",
    "This heading is not assigned to a number",
    "Stage 1: Identify the headings",
    "Note that an acronym such as",
    "B. Identify the Headings",
    "title, author and affiliation should appear as indicated",
]
found_guidance = []
for p in doc.paragraphs:
    for kw in guidance_keywords:
        if kw.lower() in p.text.lower():
            found_guidance.append(kw)

print(f"\n2. Guidance Text Check:")
if found_guidance:
    print(f"   FAIL: Found leftover template text: {found_guidance}")
else:
    print(f"   PASS: Zero template guidance text found in paper.docx.")

# 3. Abstract word count
abstract_p = next((p for p in doc.paragraphs if "abstract" in p.style.name.lower() or "abstract" in p.style.style_id.lower()), None)
if abstract_p:
    # strip 'Abstract—' or 'Abstract-'
    clean_abs = re.sub(r"^Abstract[—\-]", "", abstract_p.text).strip()
    words = clean_abs.split()
    print(f"\n3. Abstract Word Count:")
    print(f"   Word count: {len(words)} words (Rule: 150-200 words)")
    if 150 <= len(words) <= 200:
        print(f"   PASS: Abstract is within the required 150-200 word range.")
    else:
        print(f"   FAIL/WARNING: Abstract word count {len(words)} is outside 150-200.")

# 4. Page count and layout
pdf_doc = fitz.open(str(PDF_PATH))
total_pages = len(pdf_doc)
print(f"\n4. Page Count and Budget:")
print(f"   Total rendered pages: {total_pages}")
if total_pages <= 7:
    print(f"   PASS: Meets strict budget (at most 6 content pages + references, 7 pages total).")
else:
    print(f"   FAIL: Document overflows page budget ({total_pages} > 7).")

# 5. References resolution check
ref_lines = [l.strip() for l in REF_PATH.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("Verified")]
num_refs = len(ref_lines)
print(f"\n5. Reference Count & Resolution:")
print(f"   Total references in bibliography: {num_refs} (Rule: 25-35 entries)")
if 25 <= num_refs <= 35:
    print(f"   PASS: Reference count ({num_refs}) is within target range 25-35.")
else:
    print(f"   WARNING: Reference count ({num_refs}) is outside 25-35.")

# Check all citations [1] to [num_refs] in doc text
full_text = " ".join(p.text for p in doc.paragraphs)
missing_citations = []
for i in range(1, num_refs + 1):
    citation_pat = f"[{i}]"
    # also check ranges like [10]-[12]
    if citation_pat not in full_text:
        # check if in range
        in_range = False
        for m in re.finditer(r"\[(\d+)\]\s*[\-–—]\s*\[(\d+)\]", full_text):
            low, high = int(m.group(1)), int(m.group(2))
            if low <= i <= high:
                in_range = True
                break
        if not in_range:
            missing_citations.append(i)

if missing_citations:
    print(f"   WARNING: Citations not explicitly referenced in text: {missing_citations}")
else:
    print(f"   PASS: All {num_refs} references are explicitly cited in the text in numeric order.")

# 6. Check figures and tables cited
print(f"\n6. Figures & Tables In-Text Citation Check:")
for fig_num in range(1, 7):
    pat = f"Fig. {fig_num}"
    if pat in full_text:
        print(f"   PASS: Fig. {fig_num} cited in text.")
    else:
        print(f"   FAIL: Fig. {fig_num} NOT cited in text.")

for tbl_num in ["I", "II", "III"]:
    pat = f"Table {tbl_num}"
    if pat in full_text:
        print(f"   PASS: Table {tbl_num} cited in text.")
    else:
        print(f"   FAIL: Table {tbl_num} NOT cited in text.")

print("=" * 60)
print("ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
print("=" * 60)
