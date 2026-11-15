"""
tools/package_zenodo_release.py

Packages the complete Lipi dataset for Zenodo deposition:
  - data/canonical/ (16,350 images)
  - data/pairs.json
  - data/charset.csv
  - data/dataset_summary.json
  - data/splits/*.json
  - DATASET_CARD.md
  - LICENSE

Outputs:
  dist/lipi_dataset_v1.0.zip
  dist/.zenodo.json
"""

import json
import os
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIST = REPO / "dist"
DATA = REPO / "data"

ZENODO_METADATA = {
    "title": "Lipi: Handwritten Glyphs of Seven Historical Scripts of Nepal",
    "upload_type": "dataset",
    "description": (
        "<p><strong>Lipi</strong> is a standardized corpus of 16,350 handwritten glyph images across 327 character classes "
        "representing seven historical writing systems of Nepal spanning over two millennia of orthographic evolution: "
        "Ashokan Brahmi, Licchavi, Bhujimol, Nandinagari, Maithili/Tirhuta, Tibetan Uchen, and Kirat Sirijunga.</p>"
        "<p>The glyphs were handwritten on standardized grid sheets following reference charts provided by the "
        "Department of Archaeology (DoA), Government of Nepal. Each class holds 25 collected glyphs paired with 25 "
        "synthetically augmented copies, complete with verified pair mappings, benchmark splits, and a datasheet.</p>"
        "<p>Code repository: <a href=\"https://github.com/ashimst/lipi\">https://github.com/ashimst/lipi</a></p>"
    ),
    "creators": [
        {
            "name": "Shrestha, Ashim",
            "affiliation": "Department of Artificial Intelligence, School of Engineering, Kathmandu University, Nepal"
        },
        {
            "name": "Bajracharya, Sujal",
            "affiliation": "Department of Artificial Intelligence, School of Engineering, Kathmandu University, Nepal"
        },
        {
            "name": "Tuladhar, Yajjyu",
            "affiliation": "Department of Artificial Intelligence, School of Engineering, Kathmandu University, Nepal"
        },
        {
            "name": "Kunwar, Kushal",
            "affiliation": "Department of Artificial Intelligence, School of Engineering, Kathmandu University, Nepal"
        },
        {
            "name": "Regmi, Sunil",
            "affiliation": "Department of Artificial Intelligence, School of Engineering, Kathmandu University, Nepal"
        },
        {
            "name": "Bal, Bal Krishna",
            "affiliation": "Department of Artificial Intelligence, School of Engineering, Kathmandu University, Nepal"
        }
    ],
    "access_right": "open",
    "license": "cc-by-nc-4.0",
    "keywords": [
        "Historical scripts of Nepal",
        "Few-shot learning",
        "Indic epigraphy",
        "Paleography",
        "Handwritten character recognition",
        "Brahmi",
        "Licchavi",
        "Bhujimol",
        "Nandinagari",
        "Maithili",
        "Tibetan Uchen",
        "Kirat Sirijunga",
        "Document Analysis",
        "Cultural Heritage Preservation"
    ],
    "related_identifiers": [
        {
            "identifier": "https://github.com/ashimst/lipi",
            "relation": "isSupplementTo",
            "scheme": "url"
        }
    ]
}

LICENSE_TEXT = """Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- NonCommercial — You may not use the material for commercial purposes.

Full text of the license: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""

def main():
    DIST.mkdir(exist_ok=True)
    
    # 1. Write .zenodo.json in REPO root and dist/
    zenodo_json_path = REPO / ".zenodo.json"
    with open(zenodo_json_path, "w", encoding="utf-8") as f:
        json.dump(ZENODO_METADATA, f, indent=2)
    with open(DIST / ".zenodo.json", "w", encoding="utf-8") as f:
        json.dump(ZENODO_METADATA, f, indent=2)
    print(f"[✓] Wrote {zenodo_json_path}")

    # 2. Write LICENSE file if not present
    lic_path = REPO / "LICENSE"
    if not lic_path.exists():
        with open(lic_path, "w", encoding="utf-8") as f:
            f.write(LICENSE_TEXT)
        print(f"[✓] Wrote {lic_path}")

    # 3. Create lipi_dataset_v1.0.zip
    target_zip = DIST / "lipi_dataset_v1.0.zip"
    print(f"Creating Zenodo archive: {target_zip} ...")
    
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Canonical images
        canonical_dir = DATA / "canonical"
        if not canonical_dir.exists():
            raise SystemExit("Error: data/canonical does not exist!")
            
        print("  Packing data/canonical/ (16,350 images)...")
        for root, dirs, files in os.walk(canonical_dir):
            for file in files:
                p = Path(root) / file
                arcname = p.relative_to(DATA)
                zf.write(p, arcname)
                
        # Metadata files
        for meta in ["pairs.json", "charset.csv", "dataset_summary.json"]:
            p = DATA / meta
            if p.exists():
                print(f"  Packing data/{meta}...")
                zf.write(p, p.relative_to(DATA))
                
        # Splits
        splits_dir = DATA / "splits"
        if splits_dir.exists():
            print("  Packing data/splits/...")
            for sfile in splits_dir.glob("*.json"):
                zf.write(sfile, sfile.relative_to(DATA))
                
        # Documentation & License
        if (REPO / "DATASET_CARD.md").exists():
            zf.write(REPO / "DATASET_CARD.md", "DATASET_CARD.md")
        zf.write(lic_path, "LICENSE")

    size_mb = target_zip.stat().st_size / (1024 * 1024)
    print(f"\n[✓] Successfully created {target_zip} ({size_mb:.1f} MB)")
    print(f"[✓] Zenodo preparation complete. Ready to upload!")

if __name__ == "__main__":
    main()
