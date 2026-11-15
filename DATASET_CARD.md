# Lipi: Handwritten Glyphs of Seven Historical Scripts of Nepal

Dataset card following *Datasheets for Datasets* (Gebru et al., 2021). Items marked **[TO FILL]** need information only the authors have.

## Summary

| | |
|---|---|
| Images | 16,350 grayscale PNG, 128 × 128 px |
| Distinct handwritten glyphs | 8,175 (each has one augmented copy) |
| Classes | 327 (one class = one character of one script) |
| Images per class | 50 (25 glyphs + 25 augmented copies) |
| Scripts | Ashokan Brahmi, Licchavi, Bhujimol, Nandinagari, Maithili/Tirhuta, Tibetan Uchen, Kirat Sirijunga |
| Licence | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| Citation | Shrestha et al. (2026), Zenodo: [10.5281/zenodo.24240673](https://doi.org/10.5281/zenodo.24240673) |

| Script | Classes | Images |
|---|---:|---:|
| Brahmi | 43 | 2,150 |
| Licchavi | 38 | 1,900 |
| Bhujimol | 52 | 2,600 |
| Nandinagari | 51 | 2,550 |
| Maithili | 52 | 2,600 |
| Tibetan | 52 | 2,600 |
| Kirat | 39 | 1,950 |

## What the data is, and what it is not

The images are **modern handwriting** of historical glyph forms. Writers copied each character from reference charts supplied by the Department of Archaeology, Government of Nepal. The images are not crops of original inscriptions or manuscripts. Results on this dataset measure recognition of the scripts' letter shapes in clean handwriting on paper, not recognition of original epigraphic material.

## Motivation

Most historical scripts of Nepal have no labelled character data, which rules out supervised recognition. The dataset supports few-shot recognition research, in particular transfer from some scripts to scripts that have no labelled data beyond a reference chart.

## Collection

1. **Reference charts.** Character charts for each script were provided by the Department of Archaeology (DoA), Government of Nepal.
2. **Writing.** The glyphs were written by the authors, by members of the Monipa Pucha calligraphy group (who wrote part of the character set), and by student volunteers from the Department of Artificial Intelligence, Kathmandu University. Writers filled A4 sheets with a 5 × 6 grid. Row 1 shows the reference character, row 2 a reduced-opacity tracing guide, and rows 3–6 hold unguided handwriting.
3. **Scanning and segmentation.** Sheets were scanned in colour at 300–600 DPI. Cells were located by grid alignment and cropped automatically; malformed or clipped glyphs were removed by hand.
4. **Preprocessing.** Each colour crop was resized to 128 × 128 and converted to grayscale. No binarization, centring or other normalization was applied.

- Writer identity was not recorded, so the data cannot be split by writer.
- Collection period: **[TO FILL]**
- Contributors took part voluntarily. No personal information about writers is included.

## Augmented copies

Each class contains 25 collected glyphs, and each glyph is followed in sorted file order by **one augmented copy**. The copy applies one transform chosen at random from seven: Gaussian blur (radius 1.0–2.5), Gaussian noise (σ = 30), contrast (×0.4–1.6), brightness (×0.4–1.5), sharpness (×0.2–3.0), elastic deformation (α = 20, σ = 5), or erosion/dilation (2 × 2 kernel). The script is `tools/augment_dataset.py`. No random seed was fixed, so individual copies cannot be regenerated exactly.

In every pair, **the first file is the original** and the second is its copy. This follows from the augmentation script and is confirmed by the data: the noise transform's signature appears in 15.4% of second files and 1 of 8,175 first files (`tools/verify_pair_order.py`).

`data/pairs.json` lists the 25 file pairs of every class. The pairing was verified by image similarity: for 96.7% of the 16,350 images, the most similar image in the class is its file-order partner.

**Use the pairs when you evaluate.** A random few-shot episode can put a glyph in the support set and its copy in the query set, which inflates accuracy. Sample support and query from different pairs (the `disjoint` mode of `EpisodeSampler` in `lipi/episodes.py`). In our experiments the inflation from ignoring the pairs was at most 0.85 accuracy points. For the strictest test, sample episodes from the original files only (the first file of each pair; the `originals` sampling mode in `lipi/episodes.py`); in our experiments this changed accuracies by −0.45 to +1.04 points relative to source-disjoint sampling.

## Structure

```
data/canonical/<script>/<character>/<NNNN>.png   images; files 2k and 2k+1 (sorted) form a pair
data/pairs.json                                  pair map
data/dataset_summary.json                        class lists and counts
data/charset.csv                                 character inventory
data/splits/ancestry.json                        train: Brahmi + Licchavi (81 classes)
                                                 val: Nandinagari (51), test: Bhujimol, Maithili, Tibetan, Kirat (195)
data/splits/loso_<script>.json                   leave-one-script-out splits
```

Character folder names are romanized transliterations (e.g. `ka`, `kha`, `aa`). The scripts do not have identical inventories; 26 characters occur in all seven.

## Recommended evaluation

- Few-shot protocol: N-way K-shot episodes from the test classes, Q = 10 queries per class, 1,000 episodes, source-disjoint sampling, 95% confidence intervals across episodes.
- Report the sampler and seeds used, so others can regenerate identical episodes. Episode seeds are derived from readable tags in `lipi/jobs.py`; training seeds are listed in `lipi/experiments.py`.
- Supervised use: split each class by pair (for example pairs 0–19 for training, 20–24 for testing), never by image.

## Known limitations

- Contemporary handwriting only; no original epigraphic or manuscript images.
- Half of the images are augmented copies of the other half.
- Splits are disjoint in classes and scripts, but not in writers.
- Isolated characters only: no conjuncts with vowel signs, words or lines.
- The script attributions and dates in the summary file follow the reference charts and standard paleographic surveys, not a new paleographic study.

## Maintenance

Maintainer and contact: **[TO FILL]**. Report errors in labels or pairs through the repository's issue tracker.
