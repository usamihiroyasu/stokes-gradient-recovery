"""Build the main paper, supplement, and a combined PDF with resolved links."""
from pathlib import Path
import logging
import re
import shutil
import subprocess

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, NameObject, TextStringObject

logging.getLogger("pypdf").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Physical Gradient Recovery and Stopping Guarantees for a Folded Stokes Parametrization"


def main():
    if not (ROOT / "latex/ieeeaccess.cls").is_file():
        raise SystemExit("Run scripts/prepare_template.py first")
    build = ROOT / "build"
    paper = ROOT / "paper"
    build.mkdir(exist_ok=True)
    paper.mkdir(exist_ok=True)
    writer = PdfWriter()
    expected_links = []
    source_pages = []
    source_uris = []
    offset = 0
    for stem, label in [("main", "Main paper"), ("supplement", "Supplementary material")]:
        for _ in range(3):
            completed = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 f"-output-directory={build}", f"{stem}.tex"],
                cwd=ROOT / "latex", capture_output=True, text=True,
            )
            (build / f"{stem}.stdout").write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise SystemExit(f"TeX failed; see build/{stem}.stdout")
        log = (build / f"{stem}.log").read_text()
        if re.search(r"undefined|Rerun to get cross-references right|LaTeX Error", log):
            raise SystemExit(f"Unresolved TeX references; see build/{stem}.log")
        shutil.copyfile(build / f"{stem}.pdf", paper / f"{stem}.pdf")
        reader = PdfReader(paper / f"{stem}.pdf")
        for page_index, page in enumerate(reader.pages):
            source_pages.append((page.get_contents().get_data(), page.extract_text()))
            uris = []
            for annotation_index, ref in enumerate(page.get("/Annots", [])):
                annotation = ref.get_object()
                action = annotation.get("/A")
                action = action.get_object() if action else None
                if action and action.get("/S") == "/URI":
                    uris.append(str(action["/URI"]))
                if action and action.get("/S") == "/GoTo":
                    container, key = action, "/D"
                elif "/Dest" in annotation:
                    container, key = annotation, "/Dest"
                else:
                    continue
                dest = container[key]
                if isinstance(dest, TextStringObject):
                    named = reader.named_destinations[str(dest)]
                    target = reader.get_destination_page_number(named)
                    container[NameObject(key)] = named.dest_array
                elif isinstance(dest, ArrayObject):
                    target = reader.get_page_number(dest[0].get_object())
                else:
                    raise ValueError("Unsupported PDF destination")
                expected_links.append((offset + page_index, annotation_index, key, offset + target))
            source_uris.append(uris)
        writer.append(reader, outline_item=label, import_outline=False)
        offset += len(reader.pages)
    writer.add_metadata({"/Title": TITLE + " — Main and Supplement", "/Author": "Hiroyasu Usami, Ayato Tsuboi, Keisuke Hara"})
    combined = paper / "complete.pdf"
    with combined.open("wb") as handle:
        writer.write(handle)
    reader = PdfReader(combined)
    assert len(reader.pages) == offset
    for page_index, page in enumerate(reader.pages):
        assert (page.get_contents().get_data(), page.extract_text()) == source_pages[page_index]
        uris = []
        for ref in page.get("/Annots", []):
            action = ref.get_object().get("/A")
            action = action.get_object() if action else None
            if action and action.get("/S") == "/URI":
                uris.append(str(action["/URI"]))
        assert uris == source_uris[page_index]
    for page_index, annotation_index, key, expected in expected_links:
        annotation = reader.pages[page_index]["/Annots"][annotation_index].get_object()
        container = annotation["/A"].get_object() if key == "/D" else annotation
        assert reader.get_page_number(container[key][0].get_object()) == expected
    print(f"Built {offset} pages in paper/; checked {len(expected_links)} internal links")


if __name__ == "__main__":
    main()
