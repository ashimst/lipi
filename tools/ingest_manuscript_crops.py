"""
tools/ingest_manuscript_crops.py

Turns VGG Image Annotator (VIA 2.x) CSV exports of real inscription / manuscript
photographs into a standardized crop set matching data/canonical.

Inputs
  --sources  CSV with one row per photographed page:
             filename,source_id,script,institution,shelfmark,folio,url,licence,accessed,dpi
  --via      one or more VIA CSV exports (Annotation > Export Annotations (as csv)).
             Each box needs region attribute "class" (corpus class name, e.g. "ka")
             and optionally "set" ("bare" or "extended", default "bare"),
             "annotator" and "verified_by".
  --images   folder holding the photographed pages

Output (default data/manuscripts/)
  <script>/<class>/<crop_id>.png   128x128 grayscale, square-padded, 10% margin
  raw/<crop_id>.png                unresized crop for inspection
  manifest.csv                     one row per crop with full provenance
  report.txt                       counts per script / class and validation problems

Example
  python tools/ingest_manuscript_crops.py --sources pilot/sources.csv --via pilot/tibetan_via.csv pilot/bhujimol_via.csv --images pilot/pages
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

SUMMARY = json.load(open('data/dataset_summary.json'))
CORPUS_CLASSES = {s: set(v['characters']) for s, v in SUMMARY['scripts'].items()}
SIZE = 128
MARGIN = 0.10


def square_pad(img, fill):
    w, h = img.size
    side = max(w, h)
    canvas = Image.new('L', (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', required=True)
    ap.add_argument('--via', nargs='+', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', default='data/manuscripts')
    args = ap.parse_args()

    out = Path(args.out)
    (out / 'raw').mkdir(parents=True, exist_ok=True)
    sources = {r['filename']: r for r in csv.DictReader(open(args.sources, encoding='utf-8'))}
    problems, rows = [], []
    counts = defaultdict(Counter)
    n = 0

    for via_file in args.via:
        for r in csv.DictReader(open(via_file, encoding='utf-8')):
            if not r.get('region_shape_attributes') or r['region_shape_attributes'] == '{}':
                continue
            fname = r['filename']
            src = sources.get(fname)
            if src is None:
                problems.append(f'{fname}: not listed in sources.csv')
                continue
            shape = json.loads(r['region_shape_attributes'])
            attrs = json.loads(r['region_attributes'] or '{}')
            script, cls = src['script'].strip().lower(), (attrs.get('class') or '').strip()
            if shape.get('name') != 'rect':
                problems.append(f'{fname} region {r["region_id"]}: shape {shape.get("name")} is not a rectangle')
                continue
            if script not in CORPUS_CLASSES:
                problems.append(f'{fname}: unknown script "{script}"')
                continue
            kind = (attrs.get('set') or 'bare').strip().lower()
            if kind == 'bare' and cls not in CORPUS_CLASSES[script]:
                problems.append(f'{fname} region {r["region_id"]}: class "{cls}" is not a {script} corpus class')
                continue

            page = Image.open(Path(args.images) / fname).convert('L')
            x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
            mx, my = int(w * MARGIN), int(h * MARGIN)
            box = (max(0, x - mx), max(0, y - my), min(page.width, x + w + mx), min(page.height, y + h + my))
            crop = page.crop(box)
            n += 1
            crop_id = f'{script}_{n:05d}'
            crop.save(out / 'raw' / f'{crop_id}.png')
            # pad with the crop's own border colour so the background stays continuous
            border = ImageOps.expand(crop, 1).getextrema()[1]
            std = square_pad(crop, fill=border).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            dest = out / kind / script / (cls or 'unlabelled')
            dest.mkdir(parents=True, exist_ok=True)
            std.save(dest / f'{crop_id}.png')
            counts[(script, kind)][cls] += 1
            rows.append({'crop_id': crop_id, 'source_id': src['source_id'], 'script': script, 'class': cls,
                         'set': kind, 'x': x, 'y': y, 'w': w, 'h': h, 'filename': fname,
                         'institution': src.get('institution', ''), 'shelfmark': src.get('shelfmark', ''),
                         'folio': src.get('folio', ''), 'url': src.get('url', ''), 'licence': src.get('licence', ''),
                         'annotator': attrs.get('annotator', ''), 'verified_by': attrs.get('verified_by', '')})

    with open(out / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['crop_id'])
        wr.writeheader()
        wr.writerows(rows)

    lines = [f'{n} crops written to {out}']
    for (script, kind), c in sorted(counts.items()):
        lines.append(f'{script} [{kind}]: {sum(c.values())} crops, {len(c)} classes, '
                     f'{sum(v >= 5 for v in c.values())} classes with >= 5 crops')
    unverified = sum(1 for r in rows if not r['verified_by'])
    lines.append(f'unverified crops: {unverified}')
    lines += ['', f'{len(problems)} problems:'] + problems
    (out / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[: 12 + len(counts)]))


if __name__ == '__main__':
    main()
