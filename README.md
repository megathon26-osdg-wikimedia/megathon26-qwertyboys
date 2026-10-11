# Commons Category Suggester

**Team QWERTYBOYS** | Hackathon project: *Automatic Category Suggestion for Wikimedia Commons*

Upload an image (and optionally a title) and get a ranked list of Wikimedia Commons categories to accept or
reject. Nothing is edited automatically: a human approves every suggestion.

The system is retrieval-first. It does not train one classifier over millions of categories. It embeds the image
with a frozen CLIP encoder and treats categories as things to look up, so new images and new categories can be
added **without retraining**. It runs on a laptop CPU (and uses a GPU automatically if PyTorch can see one).

## How it works

| Component | What it does |
|---|---|
| CLIP image encoder (ViT-B/32) | Turns every image into a 512-number embedding |
| kNN retrieval (FAISS) | The nearest categorised images vote for their categories; the neighbours are shown as the explanation |
| Text prototypes | Compares the image with the CLIP text embedding of each category name; covers categories with few examples |
| Supervised head | Linear multi-label head initialised from the text prototypes, trained with asymmetric loss |
| Title scorer | Compares the file title with category names; helps for places and events |
| Fusion | Logistic regression over the four scores, fitted on a validation split |
| Hierarchy step | Drops a parent category when a more specific child is also suggested |
| Feedback loop | Accepted images join the index immediately; rejected suggestions are logged as hard negatives |

Data is split **by upload time** (never randomly), the category vocabulary is counted on training images only, and
near-duplicates of training images are removed from validation and test.

## Results (pilot dataset: 35 categories, 361 test images)

| Method | P@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| Text prototype (zero-shot) | 0.812 | 0.928 | 0.957 |
| kNN retrieval | 0.607 | 0.812 | 0.870 |
| Title scorer | 0.704 | 0.798 | 0.823 |
| Fusion (fixed weights, no head) | 0.745 | 0.909 | 0.938 |

- On a laptop CPU: about 92 ms to encode one image, under 1 ms to rank, about 21.6 images/s batched.
- Scores are strict: stored Commons labels are incomplete, so correct but untagged suggestions count as wrong.
- With 361 test images, differences of a few points are within sampling noise.
- These numbers come from the 35-category pilot. The final dataset has more categories (see `final_report.pdf`
  for the full discussion and limitations). To reproduce evaluation on your own data, see "Evaluate" below.

## Setup (Linux, tested on Fedora)

```bash
git clone <this repository> && cd commons-suggester      # or unzip the project and cd into it
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build of PyTorch
pip install -r requirements.txt
```

Then open `config.py` and set `CONTACT_EMAIL` to your own address. Wikimedia requires a descriptive User-Agent
with contact details for API requests.

Check the installation (prints the device, speed and a few sanity checks):

```bash
python check_env.py
python tests/test_logic.py        # offline logic tests, no internet needed
```

Each time you open a new terminal, run `cd <project folder>` and `source .venv/bin/activate` first.

## Usage

### 1. Get data (choose one or both)

```bash
python 01_collect.py              # auto-collect from Commons using SEEDS / PER_CAT in config.py
python 01b_import_local.py --folders my_dataset                          # folder per category
python 01b_import_local.py --csv labels.csv --images my_images/          # CSV, multi-label, real dates
```

### 2. Build the model

```bash
python 02_prepare.py                              # clean labels, temporal split, vocabulary
python 03_embed.py                                # CLIP embeddings (cached; only new images are embedded)
python run_pipeline.py --from 4 --tag v5_full     # train head, fit fusion, evaluate on the test split
```

Read the output of `02_prepare.py`: it prints the categories kept and the categories removed by the filter.
If real categories are removed, or junk survives, edit `MAINT_PATTERNS` in `config.py` and re-run.

### 3. Evaluate

```bash
python 06_evaluate.py --tag v5_degraded --degrade   # same test set, blurred / downscaled / JPEG-compressed
python 07_benchmark.py                              # latency, throughput, index size
python compare_runs.py                              # table comparing all saved runs
```

Results are saved to `data/results_<tag>.json`. `data/review_sample.csv` holds a random sample of suggestions
for hand-checking a human-reviewed score.

### 4. Run the demo

```bash
python app.py                     # open http://127.0.0.1:7860
```

Upload an image, optionally add a title, and press **Suggest categories**. Tick the correct suggestions and
press **Submit feedback**: accepted images are added to the index at once, and rejected suggestions are logged
as hard negatives in `data/feedback.jsonl`. If a category is missing, type its name in the add-category box: it is
usable immediately through its text prototype, and example images are fetched from Commons in the background.

The app checks at startup that `meta.json`, `emb.npz`, `head.npz` and `fusion.joblib` come from the same run.
If it reports they are out of sync, run the command it prints.

## Project layout

```
config.py            all tunable settings (model, seeds, filters, kNN, head, evaluation)
common.py            CLIP loading (GPU if available, else CPU) and embedding helpers
suggester.py         scorers, fusion, hierarchy step, index updates
category_fetch.py    looks up a category on Commons and fetches example images (add-category feature)
01_collect.py        collect images and categories from Commons via the MediaWiki API
01b_import_local.py  import your own dataset
02_prepare.py        label cleaning, temporal split, vocabulary, category parents
03_embed.py          embeddings, cache, near-duplicate removal
04_train_head.py     supervised head with asymmetric loss
05_fit_fusion.py     logistic-regression fusion on the validation split
06_evaluate.py       metrics, head/torso/tail recall, ablations, degraded-image test
07_benchmark.py      latency, throughput, memory
run_pipeline.py      runs stages 2 to 6 in order
compare_runs.py      compares saved evaluation runs
app.py               demo application
tests/               offline logic tests
data/                generated data (images, embeddings, results); not part of the source code
```

## Configuration

Everything tunable is in `config.py`. After changing a setting, re-run from the stage that depends on it:

| Change | Re-run from |
|---|---|
| Encoder (`MODEL_NAME`), text prompt | `03_embed.py` |
| Categories, filters, split, dedup threshold | `01_collect.py` or `02_prepare.py` |
| kNN settings (`KNN_K`, `KNN_TAU`), head settings | `run_pipeline.py --from 4` |

## Limitations

- The label set is a tiny fraction of Commons' millions of categories and comes from seed topics we chose, so
  scores do not forecast performance on the full site.
- Strict scoring undercounts correct suggestions because volunteer labels are incomplete.
- File titles often name the subject; a real upload may have a poor title or none.
- Events and abstract concepts rely on titles and text prototypes and were not evaluated separately.
- Geographic and cultural coverage was not audited.
- The system never identifies individuals, and suggestions are limited to content categories.

## Data and licences

The code is released under the MIT licence (see `LICENSE`). The images are not part of this repository. They are
fetched from Wikimedia Commons, where each file keeps its own free licence and attribution requirements. The CLIP
model weights are downloaded from Hugging Face on first use and are covered by their own licence.

## Acknowledgements

Wikimedia Commons and the MediaWiki API for the data, OpenAI CLIP, Hugging Face Transformers, FAISS,
scikit-learn, PyTorch and Gradio.
