"""Convert paper.docx to paper.pdf using Word ExportAsFixedFormat and render pages to PNG."""
import os
import shutil
from pathlib import Path
import win32com.client
import fitz

ROOT = Path("C:/Users/shikh/Quantization-and-Pruning")
DOCX_PATH = ROOT / "paper.docx"
PDF_PATH = ROOT / "paper.pdf"
RENDER_DIR = ROOT / "tmp" / "latest_render"
RENDER_DIR.mkdir(parents=True, exist_ok=True)

print("Starting Word automation via ExportAsFixedFormat...")
word = win32com.client.Dispatch("Word.Application")
word.Visible = False
try:
    doc = word.Documents.Open(str(DOCX_PATH.resolve()))
    # Export format 17 is wdExportFormatPDF
    doc.ExportAsFixedFormat(str(PDF_PATH.resolve()), 17)
    doc.Close(False)
    print(f"Successfully exported {PDF_PATH}!")
finally:
    word.Quit()

print("Rendering PDF pages to PNG for visual inspection...")
pdf_doc = fitz.open(str(PDF_PATH))
num_pages = len(pdf_doc)
print(f"Total pages: {num_pages}")

for i, page in enumerate(pdf_doc):
    pix = page.get_pixmap(dpi=150)
    out_img = RENDER_DIR / f"page-{i+1}.png"
    pix.save(str(out_img))
    print(f"Saved {out_img} ({pix.width}x{pix.height})")

print(f"Rendered all {num_pages} pages successfully!")
