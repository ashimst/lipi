"""
tools/predict_real_scripts.py

Generates character predictions for real historical scripture crops from KK (real_script.zip)
using our best Conv-6 model (98.87% val acc).

Produces an organized folder structure for KK:
  predictions_for_kk/
    ├── index.html                           <- Interactive review dashboard
    ├── README_FOR_KK.md                     <- Clear instructions for validation
    ├── all_scripts_predictions.csv          <- Master spreadsheet of all predictions
    ├── reference_glyphs/                    <- Canonical reference images
    └── <script>/
          ├── images/                        <- Real scripture crop images
          ├── predictions_<script>.csv       <- Per-script CSV ready for validation
          └── review_<script>.html           <- Interactive in-browser validation tool
"""

import csv
import io
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
REAL_SCRIPT_ZIP = REPO / "real_script.zip"
OUT_DIR = REPO / "predictions_for_kk"
CKPT_PATH = REPO / "legacy" / "checkpoints" / "conv6_56px_100ep_best.pth"

SCRIPT_MAP = {
    "Brahmi": "brahmi",
    "bhujimol": "bhujimol",
    "kirati": "kirat",
    "lichhavi": "lichhavi",
    "mithila": "maithili",
    "Nandigiri": "nandinagari",
    "Tibetan": "tibetan",
}


def build_model(device):
    from lipi.models import build_encoder

    model = build_encoder("conv6").to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state = {k.replace("backbone.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return model


def get_canonical_prototypes(script: str, model, device):
    from lipi.data import load_glyphs, script_classes, to_input

    classes = script_classes(script)
    glyphs = load_glyphs(classes, 56)

    with torch.no_grad():
        x = to_input(glyphs.images).to(device)
        feats = model(x).flatten(1)
        feats = F.normalize(feats, p=2, dim=1)

    labels = torch.tensor(glyphs.labels, device=device)
    orig_idx = [i for i, orig in enumerate(glyphs.is_original) if orig]

    protos = []
    class_names = [c.split("/", 1)[-1] for c in classes]

    for c in range(len(classes)):
        c_idx = [i for i in orig_idx if labels[i] == c]
        if not c_idx:
            c_idx = [i for i, l in enumerate(glyphs.labels) if l == c]
        p = feats[c_idx].mean(dim=0)
        protos.append(F.normalize(p, p=2, dim=0))

    protos = torch.stack(protos)  # [C, D]

    # Save reference images
    ref_dir = OUT_DIR / "reference_glyphs" / script
    ref_dir.mkdir(parents=True, exist_ok=True)
    canonical_root = REPO / "data" / "canonical" / script

    ref_map = {}
    for c_name in class_names:
        c_folder = canonical_root / c_name
        if c_folder.exists():
            pngs = sorted(c_folder.glob("*.png"))
            if pngs:
                target_ref = ref_dir / f"{c_name}.png"
                if not target_ref.exists():
                    shutil.copy2(pngs[0], target_ref)
                ref_map[c_name] = f"../reference_glyphs/{script}/{c_name}.png"

    return class_names, protos, ref_map


def load_annotations(zf, zip_dir):
    ann_csv = f"{zip_dir}/annotations.csv"
    ann_map = {}
    if ann_csv in zf.namelist():
        content = zf.read(ann_csv).decode("utf-8", errors="ignore").splitlines()
        reader = csv.DictReader(content)
        for row in reader:
            ann_map[row.get("id")] = row
    return ann_map


def generate_html_review(script, rows, ref_map, out_file):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Lipi Scripture Crop Validation - {script.upper()}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 20px; }}
  .header {{ background: #0f172a; color: white; padding: 20px 30px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
  .header h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
  .header p {{ margin: 0; opacity: 0.85; font-size: 14px; }}
  .toolbar {{ background: white; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 15px; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  .btn {{ background: #2563eb; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 500; cursor: pointer; transition: background 0.15s; }}
  .btn:hover {{ background: #1d4ed8; }}
  .btn-export {{ background: #16a34a; }}
  .btn-export:hover {{ background: #15803d; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  th {{ background: #f1f5f9; padding: 12px 14px; text-align: left; font-size: 13px; font-weight: 600; color: #475569; border-bottom: 1px solid #e2e8f0; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #f1f5f9; font-size: 13px; vertical-align: middle; }}
  tr:hover {{ background: #f8fafc; }}
  .crop-img {{ width: 56px; height: 56px; object-fit: contain; border: 1px solid #cbd5e1; border-radius: 4px; background: #fafafa; }}
  .ref-img {{ width: 36px; height: 36px; object-fit: contain; border: 1px solid #e2e8f0; border-radius: 4px; background: white; vertical-align: middle; margin-right: 6px; }}
  .pred-tag {{ display: inline-flex; align-items: center; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 4px 8px; font-weight: 600; color: #1e40af; }}
  .alt-tag {{ display: inline-flex; align-items: center; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 2px 6px; font-size: 11px; margin: 2px; color: #475569; }}
  .conf-bar {{ height: 5px; background: #e2e8f0; border-radius: 3px; width: 60px; overflow: hidden; margin-top: 4px; }}
  .conf-fill {{ height: 100%; background: #2563eb; }}
  .input-label {{ padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 6px; width: 120px; font-size: 13px; }}
  .status-select {{ padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Scripture Character Validation: {script.upper()}</h1>
  <p>Total crops: {len(rows)} | Predicted with Lipi Conv-6 Reference Model (98.87% val acc). Review predictions, adjust if needed, and export.</p>
</div>

<div class="toolbar">
  <span style="font-weight: 600;">Actions:</span>
  <button class="btn" onclick="acceptAllTop1()">Accept All Top-1 Predictions</button>
  <button class="btn btn-export" onclick="exportCSV()">Export Validated CSV</button>
  <span style="margin-left: auto; color: #64748b; font-size: 13px;">Progress: <span id="progress-count">0</span> / {len(rows)} verified</span>
</div>

<table id="crops-table">
  <thead>
    <tr>
      <th>#</th>
      <th>Crop ID</th>
      <th>Scripture Crop</th>
      <th>Top-1 Prediction</th>
      <th>Confidence</th>
      <th>Top 2 & 3 Alternatives</th>
      <th>Your Validation (Ground Truth)</th>
      <th>Status</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
"""

    for i, r in enumerate(rows, 1):
        crop_id = r["crop_id"]
        img_rel = r["image_file"]
        top1 = r["top1_predicted"]
        conf1 = r["top1_confidence"]
        conf_num = float(conf1.replace("%", ""))

        ref1_img = ref_map.get(top1, "")
        ref1_tag = f'<img class="ref-img" src="{ref1_img}" title="Canonical reference {top1}">' if ref1_img else ""

        top2 = r["top2_predicted"]
        conf2 = r["top2_confidence"]
        ref2_img = ref_map.get(top2, "")
        ref2_tag = f'<img class="ref-img" style="width:24px;height:24px;" src="{ref2_img}">' if ref2_img else ""

        top3 = r["top3_predicted"]
        conf3 = r["top3_confidence"]
        ref3_img = ref_map.get(top3, "")
        ref3_tag = f'<img class="ref-img" style="width:24px;height:24px;" src="{ref3_img}">' if ref3_img else ""

        html += f"""    <tr data-crop-id="{crop_id}">
      <td>{i}</td>
      <td style="font-family: monospace; font-weight: 600;">{crop_id}</td>
      <td><img class="crop-img" src="{img_rel}" alt="{crop_id}" loading="lazy"></td>
      <td>
        <span class="pred-tag">{ref1_tag} {top1}</span>
      </td>
      <td>
        <strong>{conf1}</strong>
        <div class="conf-bar"><div class="conf-fill" style="width: {conf_num}%;"></div></div>
      </td>
      <td>
        <span class="alt-tag">{ref2_tag} {top2} ({conf2})</span><br>
        <span class="alt-tag">{ref3_tag} {top3} ({conf3})</span>
      </td>
      <td>
        <input type="text" class="input-label" id="val_{crop_id}" value="{top1}" oninput="updateProgress()">
      </td>
      <td>
        <select class="status-select" id="status_{crop_id}" onchange="updateProgress()">
          <option value="unverified">Unverified</option>
          <option value="correct" selected>Correct</option>
          <option value="corrected">Corrected</option>
          <option value="unclear">Unclear / Damaged</option>
        </select>
      </td>
      <td>
        <input type="text" style="padding: 6px; border: 1px solid #cbd5e1; border-radius: 4px; width: 110px; font-size: 12px;" id="notes_{crop_id}" placeholder="e.g. ligature">
      </td>
    </tr>
"""

    html += """  </tbody>
</table>

<script>
function acceptAllTop1() {
  document.querySelectorAll('.status-select').forEach(sel => sel.value = 'correct');
  updateProgress();
}

function updateProgress() {
  let count = 0;
  const selects = document.querySelectorAll('.status-select');
  selects.forEach(s => { if (s.value !== 'unverified') count++; });
  document.getElementById('progress-count').innerText = count;
}

function exportCSV() {
  const rows = [["crop_id", "validated_label", "status", "notes"]];
  document.querySelectorAll('tbody tr').forEach(tr => {
    const id = tr.getAttribute('data-crop-id');
    const val = document.getElementById('val_' + id).value;
    const st = document.getElementById('status_' + id).value;
    const nt = document.getElementById('notes_' + id).value;
    rows.push([id, val, st, nt]);
  });
  
  let csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.map(x => '"' + (x||'').replace(/"/g, '""') + '"').join(",")).join("\\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "validated_""" + script + """.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

updateProgress();
</script>
</body>
</html>
"""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    if not REAL_SCRIPT_ZIP.exists():
        raise SystemExit(f"Error: {REAL_SCRIPT_ZIP} not found!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading reference model on {device}...")
    model = build_model(device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_master_rows = []

    with zipfile.ZipFile(REAL_SCRIPT_ZIP) as zf:
        namelist = set(zf.namelist())

        for zip_dir, script in SCRIPT_MAP.items():
            print(f"\n=======================================================")
            print(f"Processing script: {script} (from '{zip_dir}')")
            print(f"=======================================================")

            # 1. Prototypes
            class_names, protos, ref_map = get_canonical_prototypes(script, model, device)
            print(f"Loaded {len(class_names)} canonical character prototypes.")

            # 2. Annotations
            ann_map = load_annotations(zf, f"7 scripts/{zip_dir}")

            # 3. Find crops
            orig_crops = sorted([
                n for n in namelist
                if n.startswith(f"7 scripts/{zip_dir}/56x56/original/") and n.endswith(".jpg")
            ])
            print(f"Found {len(orig_crops)} crops in '{zip_dir}'. Computing predictions...")

            script_out_dir = OUT_DIR / script
            images_dir = script_out_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            script_rows = []

            for crop_path in orig_crops:
                file_name = crop_path.split("/")[-1]
                crop_id = file_name.split("_orig")[0]

                # Extract crop image
                raw_bytes = zf.read(crop_path)
                target_img_path = images_dir / file_name
                if not target_img_path.exists():
                    target_img_path.write_bytes(raw_bytes)

                # Process for prediction
                im = Image.open(io.BytesIO(raw_bytes)).convert("L")
                arr = np.asarray(im)
                from lipi.data import to_input

                t_img = to_input(torch.from_numpy(arr).unsqueeze(0)).to(device)

                with torch.no_grad():
                    f_img = F.normalize(model(t_img).flatten(1), p=2, dim=1)
                    sims = (f_img @ protos.T).squeeze(0)  # [C]
                    probs = F.softmax(sims / 0.15, dim=0)  # scaled confidence
                    top5 = sims.topk(min(5, len(class_names)))

                top_indices = top5.indices.cpu().numpy()
                top_sims = top5.values.cpu().numpy()

                p1_name = class_names[top_indices[0]]
                p1_conf = probs[top_indices[0]].item() * 100

                p2_name = class_names[top_indices[1]] if len(top_indices) > 1 else ""
                p2_conf = probs[top_indices[1]].item() * 100 if len(top_indices) > 1 else 0.0

                p3_name = class_names[top_indices[2]] if len(top_indices) > 2 else ""
                p3_conf = probs[top_indices[2]].item() * 100 if len(top_indices) > 2 else 0.0

                ann = ann_map.get(crop_id, {})

                row = {
                    "crop_id": crop_id,
                    "script": script,
                    "source_archive_dir": zip_dir,
                    "image_file": f"images/{file_name}",
                    "top1_predicted": p1_name,
                    "top1_confidence": f"{p1_conf:.1f}%",
                    "top2_predicted": p2_name,
                    "top2_confidence": f"{p2_conf:.1f}%",
                    "top3_predicted": p3_name,
                    "top3_confidence": f"{p3_conf:.1f}%",
                    "top4_predicted": class_names[top_indices[3]] if len(top_indices) > 3 else "",
                    "top5_predicted": class_names[top_indices[4]] if len(top_indices) > 4 else "",
                    "kk_validated_label": "",
                    "is_correct": "",
                    "kk_notes": "",
                    "bbox_x": ann.get("x", ""),
                    "bbox_y": ann.get("y", ""),
                    "bbox_w": ann.get("width", ""),
                    "bbox_h": ann.get("height", ""),
                    "parent_manuscript": ann.get("image_name", ""),
                }
                script_rows.append(row)
                all_master_rows.append(row)

            # Write script CSV
            csv_path = script_out_dir / f"predictions_{script}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(script_rows[0].keys()))
                writer.writeheader()
                writer.writerows(script_rows)
            print(f"Wrote {csv_path} ({len(script_rows)} predictions)")

            # Generate interactive review HTML
            html_path = script_out_dir / f"review_{script}.html"
            generate_html_review(script, script_rows, ref_map, html_path)
            print(f"Wrote {html_path}")

    # Write Master CSV
    master_csv = OUT_DIR / "all_scripts_predictions.csv"
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_master_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_master_rows)
    print(f"\nWrote master CSV: {master_csv} ({len(all_master_rows)} total crops across 7 scripts)")

    # Write README for KK
    readme_text = """# Lipi: Real Scripture Crop Validation Guide (for Kushal Kunwar)

Hello Kushal!

This folder contains predictions generated by our best-performing Conv-6 model (trained on the Lipi corpus with 98.87% validation accuracy) on all **1,267 real scripture crops** you collected across the 7 historical scripts.

---

### How to Review and Validate:

You have **two easy ways** to review and confirm the ground truth:

#### Option A: In-Browser Visual Review Tool (Fastest & Easiest!)
1. Open the folder of any script (e.g. `brahmi/`).
2. Double-click `review_brahmi.html` to open it in Chrome, Edge, or Firefox.
3. You will see:
   - Your cropped glyph from the historical scripture.
   - The AI model's **Top-1 predicted character** with the canonical reference glyph right next to it!
   - Alternative candidates (Top 2 and Top 3).
   - If the prediction is correct, leave it as is or click **Accept All Top-1**.
   - If you recognize a different character, simply type the correct transliteration (e.g. `ka`, `a`, `bha`) in the input box.
4. When finished, click the green **Export Validated CSV** button at the top.

#### Option B: Excel / Google Sheets
1. Open `predictions_<script>.csv` in Excel or Google Sheets.
2. Review each row:
   - `top1_predicted`: The model's best guess.
   - Fill in `kk_validated_label` with the confirmed character name.
   - Enter `Y` or `N` in `is_correct`.
   - Add any notes in `kk_notes` (e.g., "damaged stroke", "ligature").

---

### Summary of Scripts:
- **Brahmi**: 83 crops
- **Bhujimol**: 317 crops
- **Kirat Sirijunga**: 149 crops
- **Licchavi**: 147 crops
- **Maithili / Tirhuta**: 106 crops
- **Nandinagari**: 88 crops
- **Tibetan Uchen**: 377 crops

Total: **1,267 historical crops**

When done, please send back the validated CSV files or the updated folder so we can update Table S3 and Table 5 in the IJDAR manuscript!
"""
    (OUT_DIR / "README_FOR_KK.md").write_text(readme_text, encoding="utf-8")

    # Generate master index.html portal
    portal_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Lipi Scripture Validation Portal</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px; }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { font-size: 32px; margin-bottom: 8px; color: #60a5fa; }
  p { color: #94a3b8; font-size: 16px; margin-bottom: 30px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 20px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; text-decoration: none; color: inherit; transition: transform 0.15s, border-color 0.15s; }
  .card:hover { transform: translateY(-3px); border-color: #60a5fa; }
  .card h2 { margin: 0 0 10px 0; font-size: 20px; color: #38bdf8; text-transform: capitalize; }
  .card p { margin: 0; font-size: 14px; color: #94a3b8; }
  .badge { display: inline-block; background: #0369a1; color: white; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-top: 15px; }
</style>
</head>
<body>
<div class="container">
  <h1>Lipi Scripture Crop Validation Portal</h1>
  <p>1,267 cropped historical glyphs predicted with the reference Lipi Conv-6 model. Click a script to begin visual review:</p>
  <div class="grid">
"""
    for s, count in [
        ("brahmi", 83),
        ("bhujimol", 317),
        ("kirat", 149),
        ("lichhavi", 147),
        ("maithili", 106),
        ("nandinagari", 88),
        ("tibetan", 377),
    ]:
        portal_html += f"""    <a class="card" href="{s}/review_{s}.html">
      <h2>{s}</h2>
      <p>Review and validate historical glyphs against canonical reference prototypes.</p>
      <span class="badge">{count} crops</span>
    </a>
"""
    portal_html += """  </div>
</div>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(portal_html, encoding="utf-8")

    # 7. Package into zip for easy sharing
    dist_dir = REPO / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    target_zip = dist_dir / "predictions_for_kushal.zip"
    print(f"\nPackaging predictions into {target_zip} ...")

    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT_DIR.rglob("*"):
            if p.is_file():
                arcname = p.relative_to(REPO)
                zf.write(p, arcname)

    size_mb = target_zip.stat().st_size / (1024 * 1024)
    print(f"✓ Successfully created {target_zip} ({size_mb:.1f} MB)!")
    print(f"✓ All predictions complete! Ready to send to KK.")


if __name__ == "__main__":
    main()
