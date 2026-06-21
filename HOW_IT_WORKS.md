# How This System Works — Full Technical Walkthrough

**Project:** Early-Stage Diabetic Retinopathy (DR) Detection
**Biothon 2026 · Problem P2**

This document explains, from scratch, how every part of this project works —
the Colab training notebook and the Streamlit/FastAPI inference app — cell by
cell and function by function. For each piece you'll find **what the code
does**, **why we chose to do it that way**, and **what actually happens when
it runs**. If you've never trained a deep learning model before, read this
top to bottom; it assumes no prior ML background beyond basic Python.

---

## Table of Contents

1. [The big picture](#1-the-big-picture)
2. [Why two separate environments (Colab + laptop)?](#2-why-two-separate-environments-colab--laptop)
3. [Part A — The training notebook, section by section](#3-part-a--the-training-notebook-section-by-section)
4. [Part B — The inference app, file by file](#4-part-b--the-inference-app-file-by-file)
5. [End-to-end data flow (one image's journey)](#5-end-to-end-data-flow-one-images-journey)
6. [Glossary of terms used throughout](#6-glossary-of-terms-used-throughout)

---

## 1. The big picture

**The problem we're solving:** given a photo of the back of someone's eye
(a "retinal fundus image"), automatically determine:
1. Does this person have diabetic retinopathy (DR)?
2. If yes, how severe is it, on a 0–4 scale?

The 5 grades, in order of severity:

| Grade | Name | Meaning |
|---|---|---|
| 0 | No DR | Healthy retina |
| 1 | Mild NPDR | Earliest visible changes |
| 2 | Moderate NPDR | More widespread changes |
| 3 | Severe NPDR | Substantial damage, high risk of progressing |
| 4 | Proliferative DR | Most advanced, vision-threatening |

**The approach, in one sentence:** we take a pretrained image-recognition
neural network (originally trained to recognize everyday objects), and
*fine-tune* it on thousands of labeled retinal images so it learns to
recognize the visual patterns of DR severity instead.

This is called **transfer learning** — instead of teaching a neural network
to understand "edges, textures, shapes" from zero (which needs millions of
images and huge compute), we start from a network that already understands
those general visual building blocks, and just teach it the new task on top.

---

## 2. Why two separate environments (Colab + laptop)?

This is a deliberate architectural decision, not an accident, so it's worth
explaining up front.

**Training a neural network** is computationally expensive — it involves
millions of arithmetic operations per image, repeated over thousands of
images, repeated again for many passes ("epochs") over the dataset. This is
**much** faster on a GPU (a chip designed for doing many calculations in
parallel) than a CPU. Google Colab gives free temporary access to a GPU
(a T4), which is why training happens there.

**Running the trained model on a single new image** (inference) is cheap by
comparison — one image, one forward pass through the network, done in under
a second even on a normal laptop CPU. There's no need for a GPU at demo time.

So the workflow is:
```
Colab (has a free GPU, but disconnects/resets) 
    → train the model once
    → save the trained weights to Google Drive (persists forever)
        → download that one file to your laptop
            → laptop runs the lightweight inference app (Streamlit/FastAPI), no GPU needed
```

This means your hackathon demo doesn't depend on conference Wi-Fi reaching
Google's servers, doesn't depend on Colab still being connected, and doesn't
need anyone in the room to have a gaming laptop.

---

## 3. Part A — The training notebook, section by section

File: `notebook/train_dr_model.ipynb`

### Section 1 — Setup & Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
DRIVE_PROJECT_DIR = '/content/drive/MyDrive/biothon2026-dr-detection'
os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
os.makedirs(f'{DRIVE_PROJECT_DIR}/models', exist_ok=True)
os.makedirs(f'{DRIVE_PROJECT_DIR}/logs', exist_ok=True)
```

**What it does:** connects this Colab notebook to your personal Google
Drive, and creates a folder structure to save outputs into.

**Why:** Colab's own disk (`/content/...`) is **wiped every time the runtime
disconnects** (after ~90 minutes idle, or when you close the tab). Google
Drive is permanent storage. By saving the trained model to Drive instead of
just `/content`, the model survives even if Colab kicks you out mid-training
— and you can download it to your laptop afterward.

**What happens when it runs:** a pop-up asks you to authorize Colab to
access your Google Drive. Once approved, your Drive appears as a regular
folder at `/content/drive/MyDrive/...` inside the notebook, and any file
written there is the literal same file you'd see in drive.google.com.

```python
!pip install -q timm opencv-python-headless albumentations scikit-learn tqdm --upgrade
```

**What it does:** installs Python packages not pre-installed in Colab.

**Why each one:**
- `opencv-python-headless` — image reading/resizing/blurring (used heavily in Ben Graham preprocessing, Section 3)
- `scikit-learn` — train/validation splitting and evaluation metrics (QWK, classification report)
- `tqdm` — progress bars so long-running loops (like resizing 3,662 images) show visible progress instead of looking frozen
- `timm` / `albumentations` — installed for flexibility (alternate backbones / augmentation pipelines) even though the current notebook's core path uses `torchvision` models directly

```python
import torch, torch.nn as nn, ...
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**What it does:** imports the deep learning library (PyTorch) and checks
whether a GPU (`cuda`) is available.

**Why:** every later cell that does math on images/tensors needs to know
whether to run on GPU (`cuda`) or CPU. If no GPU is detected, the notebook
prints a warning telling you to enable one in Colab's settings — training on
CPU here would take many hours instead of under an hour.

---

### Section 2 — Get the APTOS 2019 dataset

This is the most "fought with" part of the project (see the troubleshooting
history), so it's worth explaining thoroughly.

**The dataset:** [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection)
is a public Kaggle competition dataset of 3,662 labeled retinal fundus
images, each tagged with a grade 0–4 by medical professionals. It's the
standard dataset used for this exact task.

**Why we don't just `pip install` the dataset:** Kaggle requires
authentication (you must have an account and accept the competition's terms)
before downloading competition data — this is Kaggle's policy, not something
we can bypass, and it's there because some competitions have data usage
restrictions.

```python
NEW_KAGGLE_TOKEN = ''
KAGGLE_USERNAME = ''
...
if NEW_KAGGLE_TOKEN and KAGGLE_USERNAME:
    with open('/root/.kaggle/kaggle.json', 'w') as f:
        json.dump({'username': KAGGLE_USERNAME, 'key': NEW_KAGGLE_TOKEN}, f)
    os.chmod('/root/.kaggle/kaggle.json', 0o600)
elif not os.path.exists('/root/.kaggle/kaggle.json'):
    uploaded = files.upload()
    ...
```

**What it does:** sets up your Kaggle credentials one of two ways —
either you paste your API token directly as a string (newer Kaggle token
format), or you upload a `kaggle.json` credentials file (classic format).
Either path ends with the same result: a `kaggle.json` file sitting at
`/root/.kaggle/kaggle.json`, which is the exact location the `kaggle`
command-line tool looks for credentials.

**Why two paths:** Kaggle changed their token UI partway through this
project — some accounts only get the new `KGAT_...`-style token with no
direct file download, others can still generate the classic `kaggle.json`
under "Legacy API Credentials." Supporting both means the notebook works
regardless of which one your account shows you.

**`os.chmod(..., 0o600)`** — this sets the file's permissions so only your
user account can read it (not world-readable). This isn't optional ceremony;
it's a real security practice for credential files, and the `kaggle` tool
will actually warn you if the file is too permissive.

```python
!kaggle competitions download -c aptos2019-blindness-detection -p {RAW_DIR}
```

**What it does:** downloads the *entire* competition bundle (a single zip
containing train images, train labels, test images, and test labels) as one
file, into `/content/data_raw/`.

**Why the whole bundle, not just the training images:** an earlier version
of this notebook tried to download `train.csv` and `train_images.zip`
individually using Kaggle's "download one specific file" API endpoint. That
approach **intermittently 404'd** on the large `train_images.zip` file
specifically, even with valid credentials — `train.csv` (small) downloaded
fine, but the larger file did not. Downloading the whole bundle at once uses
a different, more reliable API code path on Kaggle's side. The tradeoff is a
temporarily larger download (~9-10GB instead of a smaller selective one),
but since we delete everything we don't need immediately after extracting
what we want (next cell), it doesn't cost us long-term disk space.

```python
print('If you see a 401/403 error above: ...')
```

**What it does:** prints plain-English explanations for the two most common
failure modes, *inline*, right where the error would appear.

**Why:** a `401` (Unauthorized) means your credentials are wrong/expired. A
`403` (Forbidden) almost always means you haven't clicked "I Understand and
Accept" on the competition's rules page yet — Kaggle blocks downloads until
you do, separately from whether your token is valid. Both errors look
similar to a beginner but have completely different fixes, so spelling out
which is which saves a debugging round-trip.

---

### The extraction cell (right after download)

```python
outer_zip_candidates = glob.glob(f'{RAW_DIR}/*.zip')
outer_zip_candidates = [p for p in outer_zip_candidates if 'train_images' not in p and 'test_images' not in p]
...
with zipfile.ZipFile(outer_zip_path, 'r') as outer_zip:
    names = outer_zip.namelist()
    print(f'Bundle contains {len(names)} entries. First 30 shown:')
```

**What it does:** opens the big downloaded zip *without fully extracting it
to disk yet*, and lists what's inside it.

**Why open it "without extracting" first:** Python's `zipfile` module can
read a zip's internal file listing and pull out individual files from it
without unpacking the *entire* archive to disk first. Since the bundle
contains test images we never want (≈half the bundle's size), we read the
listing, find just the files we need, and extract only those — never
touching the rest.

```python
train_images_zip_name = next((n for n in names if n.endswith('train_images.zip')), None)
loose_train_pngs = [n for n in names if 'train_images' in n.lower() and n.lower().endswith('.png')]
```

**What it does:** checks for two different possible internal layouts —
either the bundle contains a *nested* zip file (`train_images.zip` inside
the outer zip) or it contains the images as loose `.png` files directly.

**Why both:** Kaggle doesn't guarantee a single consistent internal
structure for every competition bundle, and we hit exactly this ambiguity
during development — the first version of this code only checked for the
nested-zip case and failed silently when the real bundle used loose files
instead. Checking both (plus a final "just grab any `.png` not in
test_images" fallback) makes the code resilient to either layout.

```python
extraction_succeeded = False
...
src_files = list(Path(extract_tmp).rglob('*.png'))
if not src_files:
    raise FileNotFoundError(...)
extraction_succeeded = True
if extraction_succeeded:
    os.remove(outer_zip_path)
```

**What it does:** only deletes the big downloaded bundle *after* confirming
that `.png` files actually landed on disk from it.

**Why this matters a lot:** an earlier version of this cell deleted the
~9.6GB bundle unconditionally, right after attempting extraction — even if
extraction had silently matched zero files. That meant a failed extraction
would not only fail, but also destroy the 9.6GB download you'd just have to
redo from scratch. This verify-then-delete ordering means a failure now
costs you nothing — the bundle stays put and you can just fix the matching
logic and re-run.

```python
for src_path in tqdm(src_files):
    img = cv2.imread(str(src_path))
    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
```

**What it does:** loads each extracted image, and if its longest side is
bigger than `MAX_DIM` (512 pixels), shrinks it down (preserving aspect
ratio) before saving.

**Why:** original APTOS images come straight from fundus cameras and can be
several megapixels each — far more resolution than the model actually needs
(it only looks at 224×224 pixels at training time, see Section 4). Without
this resize step, the dataset would occupy far more disk space than
necessary for no accuracy benefit. Resizing once, up front, to 512px keeps
total dataset size in the **2–4GB range** while still leaving headroom above
the 224px the model actually consumes (so later cropping/augmentation still
has detail to work with).

`cv2.INTER_AREA` is specifically the interpolation algorithm OpenCV
recommends for *shrinking* images (as opposed to `INTER_LINEAR` or
`INTER_CUBIC`, which are better suited to enlarging) — it gives cleaner
results with less aliasing when reducing resolution.

`IMWRITE_PNG_COMPRESSION, 6` — a middle-ground PNG compression level
(0=none, 9=max). Higher compression saves more disk space but takes longer
to write; 6 is a reasonable default balance.

---

### The size-check cell

```python
def get_dir_size_gb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            total += os.path.getsize(os.path.join(dirpath, fn))
    return total / (1024 ** 3)

dataset_size_gb = get_dir_size_gb(DATA_DIR)
print(f'Final dataset size: {dataset_size_gb:.2f} GB  (target: 2-4 GB)')
```

**What it does:** walks every file under the dataset folder, sums up their
byte sizes, and converts to gigabytes.

**Why:** this is a direct confirmation step — rather than just *assuming*
the resize step worked, we measure the actual result and print it, with an
explicit comparison against the 2–4GB target. If it's over, the printed
message tells you exactly which variable to change (`MAX_DIM`) to fix it.

---

### Section 3 — Ben Graham preprocessing

```python
def crop_image_from_gray(img, tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img > tol
    ...
    return np.stack([img1, img2, img3], axis=-1)
```

**What it does:** removes the black border/background around the actual
circular eye image. Fundus cameras produce a photo where the eye occupies a
circle in the middle of an otherwise black rectangular frame; this function
finds the bounding box of "non-black" pixels and crops to just that region.

**Why:** without cropping, the model would waste a meaningful fraction of
every image on solid black pixels that carry zero diagnostic information,
and different photos have different amounts of black border — that
inconsistency adds noise the model has to learn to ignore instead of
focusing capacity on the actual retina.

```python
def ben_graham_preprocess(img, sigma_x=10, target_size=224):
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (target_size, target_size))

    blurred = cv2.GaussianBlur(img, (0, 0), sigma_x)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)

    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, (1, 1, 1), -1)
    img = img * mask + 128 * (1 - mask)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

This is the single most important preprocessing step for this specific
dataset, named after Ben Graham, who published this technique for the
original Kaggle diabetic retinopathy competition. Three operations, explained
individually:

**1. `cv2.GaussianBlur` + `cv2.addWeighted(img, 4, blurred, -4, 128)`** —
this is "subtract the local average color." `GaussianBlur` produces a
heavily blurred version of the image (essentially: "what's the average color
in this neighborhood of pixels?"). `addWeighted` then computes
`4*original - 4*blurred + 128`, which amplifies the *difference* between
each pixel and its local neighborhood average, recentered around mid-gray
(128).

**Why this matters:** APTOS images come from many different fundus cameras
across different clinics, with wildly inconsistent lighting, white balance,
and exposure. Two images of *equally healthy* retinas can look very
different in raw color just because of which camera took them. Subtracting
the local average color removes this per-image lighting bias and leaves
behind what actually varies *within* the image — texture, vessel patterns,
lesions — which is the diagnostically relevant signal. Without this step,
the model risks partly learning to recognize "which camera/clinic this photo
came from" instead of "what does this retina show," which would hurt
real-world generalization.

**2. The circular mask** — after the contrast operation above, the four
corners of the square image (outside the circular eye region) can pick up
visual artifacts from the blur operation near the crop boundary. The mask
zeroes out (resets to mid-gray) everything outside a circle sized to match
the eye, so those corner artifacts don't influence the model.

**What happens when you run the visualization cell after this:** you'll see
a 2-row, 5-column grid — top row is the raw image for each grade (0-4),
bottom row is the same image after Ben Graham processing. The processed
versions should look flatter/more uniform in lighting but with vessel
patterns and any lesions noticeably more visible — that's the preprocessing
doing its job. This is a sanity check, not just decoration: if the processed
images look broken (e.g., mostly gray with no visible retina), something's
wrong before you spend an hour training on bad data.

---

### Section 4 — Dataset & DataLoader

```python
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize(imagenet_mean, imagenet_std),
])
```

**What it does:** defines a chain of random transformations applied to
*training* images only (not validation images — see `val_transform`, which
skips the random parts).

**Why each one:**
- `RandomHorizontalFlip` / `RandomVerticalFlip` — a retina photographed
  flipped is still medically identical; this teaches the model that
  orientation doesn't matter, effectively multiplying the variety of
  training examples for free.
- `RandomRotation(20)` — fundus images can be captured at slightly different
  angles in practice; small rotations make the model robust to that.
- `ColorJitter` — small random brightness/contrast tweaks simulate
  additional camera variation beyond what Ben Graham preprocessing already
  normalized, adding a bit more robustness.
- `Normalize(imagenet_mean, imagenet_std)` — rescales pixel values using the
  same mean/standard deviation statistics the original ImageNet-pretrained
  network expects. This is **required**, not optional — the pretrained
  EfficientNet-B0 weights were learned assuming inputs are normalized this
  exact way; skipping it would feed the network data in a totally different
  numerical range than it was trained to expect, badly hurting accuracy.

This combination of small random transformations is called **data
augmentation** — it's a way to artificially expand a limited dataset (3,662
images is small by deep learning standards) by showing the model many
slightly-varied versions of the same images across different training
passes, which reduces overfitting.

```python
class APTOSDataset(Dataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['img_path'])
        img = ben_graham_preprocess(img, target_size=IMG_SIZE)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row['diagnosis'], dtype=torch.float32)
        return img, label
```

**What it does:** defines how to load one training example — read the image
file, apply Ben Graham preprocessing, apply the augmentation transforms, and
package it with its label.

**Why `dtype=torch.float32` for the label (not an integer/long type):**
this is a deliberate and important choice tied to the ordinal regression
design (explained fully in Section 5 below) — the label needs to be a
floating-point number because we're training the model to predict a
*continuous* value, not to pick from 5 discrete categories.

```python
class_counts = train_df['diagnosis'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
sample_weights = train_df['diagnosis'].map(lambda c: class_weights[c]).values
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, ...)
```

**What it does:** this is the **class imbalance fix**. APTOS has far more
Grade 0 (healthy) images than Grade 3-4 (severe) images — left alone, a
model trained on this would become biased toward predicting "healthy" since
that's the statistically safe bet most of the time.

**How it works mechanically:** `class_weights = 1.0 / class_counts` gives
rare classes (small `class_counts`) a *large* weight, and common classes a
*small* weight. `WeightedRandomSampler` then uses these weights as
probabilities when randomly picking which training example to include in
each batch — so even though there are far fewer Grade 4 images in the
dataset, they get picked roughly as often as Grade 0 images during training,
because each Grade 4 image individually has a much higher chance of being
selected per draw.

**Why this is the *only* imbalance-correction mechanism (and not also
class-weighted loss):** a separate, equally valid technique for imbalance is
weighting the *loss function* itself, so mistakes on rare classes count for
more. Using **both** the sampler and a weighted loss at the same time
applies the correction twice — the sampler already rebalances what the
model sees per batch; multiplying that rebalanced batch's loss by class
weights again over-corrects, which in practice can hurt the majority
class's accuracy more than necessary without actually improving minority
class performance further. We deliberately use exactly one mechanism
(the sampler) and leave the loss function plain.

---

### Section 5 — Model: EfficientNet-B0 + ordinal regression head

```python
base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = base.classifier[1].in_features
base.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, 1)
)
```

**What it does:** loads EfficientNet-B0 with its original ImageNet-trained
weights (so it already "knows" general visual features), then replaces its
final classification layer.

**The original EfficientNet-B0** ends in a layer that outputs 1000 numbers
(one score per ImageNet category, like "cat," "stop sign," etc.). We don't
want any of those 1000 categories — we strip that final layer off
(`base.classifier`) and replace it with our own much smaller layer.

**Why `nn.Linear(in_features, 1)` — outputting just ONE number, not 5:**
this is the most important design decision in the whole model, so it's
worth explaining carefully.

The "obvious" approach to a 5-grade classification problem is to output 5
numbers (one confidence score per grade, picking whichever is highest — this
is what `nn.Linear(in_features, 5)` plus a softmax would do). **We
deliberately did not do this.** Here's why:

DR grades are **ordinal** — they have a meaningful order (0 < 1 < 2 < 3 <
4), and the "distance" between grades matters clinically. Mistaking a
healthy eye (Grade 0) for proliferative DR (Grade 4) is a far more serious
error than mistaking Grade 0 for Grade 1. But standard 5-way classification
treats every wrong answer as equally wrong — there's no built-in notion that
predicting "3" when the truth is "4" is a *better* mistake than predicting
"0" when the truth is "4." The model gets the exact same penalty either way.

By outputting a **single continuous number** instead, and training the model
to make that number as close as possible to the true grade (treating "2.0"
as the target for a Grade 2 image), the model is mathematically pushed to
produce predictions that are *close* to the right grade even when not
exact — predicting 3.6 for a true Grade 4 is treated as a much smaller error
than predicting 0.2 for a true Grade 4. This naturally respects the ordinal
structure of the problem in a way plain classification doesn't.

`nn.Dropout(0.3)` — during training, this randomly "turns off" 30% of the
neurons feeding into the final layer on each pass. This is a standard
regularization technique that prevents the model from over-relying on any
single specific neuron/feature, which reduces overfitting — especially
important here since our dataset (3,662 images) is small relative to the
millions of images deep networks are often trained on.

```python
def forward(self, x):
    return self.base(x).squeeze(1)
```

**What it does:** `.squeeze(1)` removes a redundant dimension from the
output. The network technically outputs a tensor of shape `(batch_size, 1)`
(one number per image, wrapped in an extra dimension); squeezing turns it
into shape `(batch_size,)` — a flat list of single numbers, which is what
the loss function and metrics expect.

---

### Section 6 — Training loop

```python
criterion = nn.SmoothL1Loss()
```

**What it does:** defines the loss function — the mathematical measure of
"how wrong" a prediction is, which the model adjusts itself to minimize.

**Why `SmoothL1Loss` (Huber loss) specifically, not plain MSE:** MSE (Mean
Squared Error) squares the error, which means large errors get punished
*disproportionately* harder than small ones (an error of 2 contributes 4x
more loss than an error of 1, not 2x). This can make training unstable if a
few mislabeled or unusual images produce huge errors early in training. Huber
loss behaves like MSE for small errors (still encourages precision) but like
a gentler linear penalty for large errors (doesn't let a few outliers
dominate the training signal). This is a standard, more robust choice for
regression with real-world noisy labels.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
```

**What it does:** AdamW is the algorithm that actually updates the model's
internal numbers (weights) after each batch, based on the loss. `lr` (learning
rate) controls how big each update step is. `weight_decay` is another
regularization technique — it gently pulls weights toward zero over time,
discouraging the model from relying too heavily on any single weight growing
very large, which again helps prevent overfitting on a relatively small
dataset.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
```

**What it does:** automatically shrinks the learning rate (multiplying it by
0.5) if validation QWK (see below) hasn't improved for 2 consecutive epochs.

**Why:** early in training, large update steps help the model learn quickly.
Later, when it's close to a good solution, large steps can cause it to
overshoot and bounce around instead of settling. Shrinking the learning rate
once progress stalls lets the model fine-tune more precisely in its later
epochs.

```python
def grades_from_regression(raw_preds):
    return np.clip(np.round(raw_preds), 0, 4).astype(int)
```

**What it does:** converts the model's raw continuous output (e.g., `2.73`)
back into a usable integer grade, by rounding to the nearest whole number
and clamping the result to stay within the valid 0–4 range (in case the
model outputs something like `-0.3` or `4.6`, which can happen since nothing
mathematically forces a regression output to stay in range).

**Why this is needed:** this is the necessary complement to the ordinal
regression design — the model's *raw* output is a free-floating real number,
but for reporting purposes ("this is Grade 2") we need a concrete category,
so this function bridges that gap at evaluation/inference time.

```python
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    ...
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

**What it does:** this is the actual core training mechanics, run once per
epoch (one full pass through the dataset), either in training mode (updates
weights) or evaluation mode (just measures performance, no updates).

**Why `model.train()` vs `model.eval()` matters:** some layers (like
Dropout, used in our model's final layer) behave differently during training
vs. evaluation — Dropout randomly disables neurons during training but
should NOT do so during evaluation (you want the model's full, deterministic
prediction when actually measuring how good it is). Calling the right mode
ensures this switch happens correctly.

**The training step, line by line:**
- `optimizer.zero_grad()` — clears old gradient information from the
  previous batch (PyTorch accumulates gradients by default, so this reset is
  required each time)
- `loss.backward()` — computes how much each of the model's millions of
  internal weights contributed to the current error (this is
  "backpropagation")
- `optimizer.step()` — actually nudges every weight slightly in the
  direction that would have reduced the error, scaled by the learning rate

`torch.set_grad_enabled(train)` — gradient computation (needed for training)
is computationally expensive and unnecessary during evaluation. Wrapping
evaluation in `grad_enabled(False)` (which happens automatically when
`train=False`) saves memory and runs faster.

```python
qwk = cohen_kappa_score(true_grades, pred_grades, weights='quadratic')
```

**What it does:** computes **Quadratic Weighted Kappa (QWK)** — the metric
we actually use to decide if the model is improving, rather than plain
accuracy.

**Why QWK instead of accuracy:** this is the evaluation-side counterpart of
the ordinal regression design choice. Accuracy treats every wrong prediction
as equally bad. QWK specifically penalizes predictions in proportion to how
far off they are — a Grade 0 image predicted as Grade 1 barely dents the
QWK score, but predicted as Grade 4 hurts it substantially. QWK is also the
**exact metric the original APTOS/DR Kaggle competitions were scored on**,
so it's the standard, comparable way to report performance on this task.

```python
for epoch in range(1, EPOCHS + 1):
    train_loss, train_qwk, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_qwk, val_acc = run_epoch(val_loader, train=False)
    scheduler.step(val_qwk)
    ...
    if val_qwk > best_qwk:
        best_qwk = val_qwk
        patience_counter = 0
        torch.save({...}, MODEL_SAVE_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping...')
            break
```

**What it does:** the actual training loop — for up to 20 epochs, train on
the training set, then evaluate on the held-out validation set, and save the
model **only when validation QWK improves**.

**Why save only on improvement (not every epoch):** this implements
**early stopping** combined with keeping the best checkpoint. A model can
start *overfitting* the longer it trains — memorizing quirks of the training
images rather than learning generalizable patterns — at which point training
loss keeps dropping but validation performance gets worse. By only keeping
the checkpoint from the best validation epoch, and stopping entirely if 5
epochs pass with no improvement (`PATIENCE = 5`), we avoid shipping an
overfit model and avoid wasting Colab GPU time training past the point of
diminishing returns.

**Why validate on a separate held-out set at all:** `train_df`/`val_df` were
split earlier (85%/15%, Section 4) precisely so we have data the model
*never sees during training* to honestly measure how well it generalizes to
new images — testing on training data would give a falsely optimistic
picture, since the model could simply be memorizing those exact images.

---

### Section 7 — Final evaluation

```python
cm = confusion_matrix(true_grades, pred_grades)
```

**What it does:** builds a 5×5 grid showing, for every true grade, how the
predictions were distributed across all 5 predicted grades.

**Why this matters more than a single accuracy number:** a confusion matrix
reveals *where* the model struggles — e.g., it might tell you the model is
excellent at separating "DR present vs not" (Grade 0 vs 1-4) but
occasionally confuses adjacent severity grades like 2 and 3, which is a much
more useful diagnostic picture than one overall accuracy percentage.

---

### Section 8 — Export the model

```python
example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save(traced_path)
```

**What it does:** converts the trained model into **TorchScript** format —
a serialized, self-contained version of the model that doesn't require the
original Python class definition (`DRModel`) to be present to load it later.

**Why this matters for the app:** the inference app (`app/inference.py`)
loads this file with `torch.jit.load(...)` and never needs to import or
redefine the `DRModel` class — the traced file contains everything needed to
run the network's computations on its own. This decouples the training code
from the inference code cleanly; `app.py` doesn't need to know anything
about how the model was built, only how to call it.

`torch.jit.trace` works by literally running one example input through the
model and recording every operation that happens — which is why an
`example_input` of the right shape (`1, 3, IMG_SIZE, IMG_SIZE` — one image,
3 color channels, height, width) is needed to "trace" the path.

---

## 4. Part B — The inference app, file by file

### `app/preprocessing.py`

This file contains a **byte-for-byte identical copy** of `crop_image_from_gray`
and `ben_graham_preprocess` from the notebook's Section 3.

**Why duplicated instead of imported from the notebook:** the notebook only
exists as a `.ipynb` file meant to run in Colab — it's not something the
local Streamlit app can import as a Python module. Since the model was
*trained* on Ben-Graham-processed images, every image it's asked to predict
on **must** go through the exact same preprocessing, or its predictions will
be meaningless (the model has learned to recognize patterns in
already-processed images, not raw photos). This file is the single
source of truth for that preprocessing on the inference side — its
docstring explicitly warns future editors that it must be kept in sync with
the notebook.

### `app/inference.py`

```python
nearest_int = round(raw_score)
distance = abs(raw_score - nearest_int)
confidence = 1.0 - (distance / BOUNDARY_ZONE) * 0.65
```

**What it does:** this is the **confidence score** calculation, and it's
specifically designed around the ordinal regression model — it does NOT use
a generic softmax probability, because our model doesn't produce one (it
only outputs a single number).

**The logic:** if the model's raw output is very close to a whole number
(e.g., `2.02`), that's confident — the model is firmly landing on Grade 2
with little ambiguity, so `distance` is near 0 and confidence is near 1.0.
If the raw output sits near the midpoint between two grades (e.g., `2.49`,
almost exactly between Grade 2 and Grade 3), that's genuinely ambiguous —
`distance` is near 0.5 (the maximum possible) and confidence drops to its
floor (~0.35, not all the way to 0, because being "on the fence" between two
*adjacent* clinically-similar grades isn't meaningless information, just
uncertain).

**Why this distinction matters clinically:** without this, a raw score of
`2.02` and a raw score of `2.49` would both just get reported as "Grade 2" —
hiding the fact that one of those is a confident call and the other is a
coin flip. Collapsing that distinction would mean the app gives equally
confident-sounding output for genuinely different situations, which is
exactly the failure mode this design avoids.

```python
def _soft_probability_bars(raw_score, spread=0.6):
    grades = np.arange(5)
    weights = np.exp(-0.5 * ((grades - raw_score) / spread) ** 2)
    weights = weights / weights.sum()
    return {int(g): float(w) for g, w in zip(grades, weights)}
```

**What it does:** builds the 5 probability bars you see in the UI, using a
**Gaussian (bell curve) shape centered on the raw regression output**.

**Why a Gaussian, and why this is explicitly *not* a real probability
distribution:** since the model has no softmax layer, there's no
mathematically "correct" probability distribution to report — it only
produces one number. This function manufactures a visually intuitive
distribution for the UI: grades close to the raw score get high bars, grades
far away get low bars, and they sum to 100%. This is a deliberate
visualization aid, not a claim about calibrated uncertainty — the code
comments are explicit about this so nobody mistakes it for something it
isn't.

```python
class DRPredictor:
    def __init__(self, model_path, img_size=224, device='cpu'):
        ...
        self.model = torch.jit.load(str(path), map_location=self.device)
        self.model.eval()
```

**What it does:** loads the TorchScript model file exported from the
notebook. `map_location=self.device` ensures it loads correctly even though
it was trained on a GPU (Colab) and is now running on a CPU (your laptop) —
without this, loading a GPU-trained model on a CPU-only machine can throw an
error.

`self.model.eval()` — same reasoning as in the training loop: ensures
Dropout and similar layers behave in their deterministic (non-random) mode
during actual predictions.

```python
def predict(self, bgr_img):
    processed = ben_graham_preprocess(bgr_img, target_size=self.img_size)
    tensor = self._to_tensor(processed)
    with torch.no_grad():
        raw_output = self.model(tensor)
```

**What it does:** the actual end-to-end prediction pipeline for one image —
preprocess it exactly like training data, convert it to the tensor format
PyTorch expects, and run it through the model.

`torch.no_grad()` — tells PyTorch not to bother tracking gradient
information, since we're not training here, only predicting. This makes
inference faster and uses less memory.

### `app/app.py` (Streamlit UI)

```python
@st.cache_resource(show_spinner=False)
def load_predictor():
    return DRPredictor(model_path=str(MODEL_PATH), img_size=224, device="cpu")
```

**What it does:** loads the model once and caches it in memory.

**Why `@st.cache_resource`:** Streamlit re-runs your *entire script* from
top to bottom every time the user interacts with anything (clicking a
button, uploading a file). Without caching, the multi-hundred-megabyte model
would be reloaded from disk on every single click, making the app
unbearably slow. This decorator tells Streamlit "run this function once,
then reuse its result across all future re-runs" — so the model loads
exactly once per app session.

```python
if not MODEL_PATH.exists():
    st.error(f"**Model file not found** at `{MODEL_PATH}`.\n\n1. Run the notebook...")
    st.stop()
```

**What it does:** checks the model file exists *before* trying to load it,
and gives a specific, actionable error message (not a generic crash) if it's
missing.

**Why:** this is the most likely first-run failure for anyone setting up the
app — forgetting Part A (training) or forgetting to copy the downloaded file
into the right folder. A clear, specific message here saves significant
confusion compared to a raw Python stack trace.

```python
if result.needs_manual_review:
    st.warning(f"⚠️ Confidence ({result.confidence*100:.1f}%) is below the {UNCERTAINTY_THRESHOLD*100:.0f}% threshold...")
```

**What it does:** explicitly surfaces the uncertainty flag from
`inference.py` as a visible warning banner in the UI whenever confidence
drops below 60%.

**Why this is a deliberate UX choice, not just a debug message:** in a
clinical-adjacent context, silently reporting "Grade 2" with no indication
of confidence is misleading regardless of whether the underlying number is
51% or 94% — both look identically authoritative to someone reading the
screen. Surfacing the uncertainty explicitly, with a visually distinct
warning, makes the tool honest about when its own prediction shouldn't be
trusted at face value, which is the responsible way to present a borderline
ML output in a healthcare-adjacent demo.

### `app/api.py` (FastAPI, optional)

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    file_array = np.frombuffer(contents, dtype=np.uint8)
    bgr_img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    predictor = get_predictor()
    result = predictor.predict(bgr_img)
    return PredictionResponse(...)
```

**What it does:** exposes the exact same `DRPredictor` used by the Streamlit
app as a REST API endpoint instead — accepts an uploaded image file, runs
the same prediction pipeline, returns the result as JSON.

**Why this exists alongside Streamlit, not instead of it:** the Streamlit
app is the primary, self-contained demo deliverable (no separate
frontend/backend to keep in sync). The FastAPI layer is there in case the
team wants to show a more traditional API-style integration point — e.g.,
plugging a future custom frontend into this same model — without retraining
or duplicating the prediction logic. Both files import the same
`inference.py`, so there's exactly one implementation of "how prediction
works," just exposed two different ways.

---

## 5. End-to-end data flow (one image's journey)

To tie it all together, here's literally everything that happens to a
single image, from upload to displayed result:

1. **You upload a `.png`/`.jpg`** in the Streamlit file uploader.
2. **`app.py`** reads the raw bytes and decodes them into a BGR pixel array
   using OpenCV (`cv2.imdecode`).
3. **`app.py`** calls `predictor.predict(bgr_img)`.
4. **`inference.py`** calls `ben_graham_preprocess()` — crops the black
   border, subtracts local average color, applies the circular mask. Output:
   a clean 224×224 RGB image.
5. **`inference.py`** converts that image into a normalized PyTorch tensor
   (`_to_tensor`) — scales pixel values to 0-1, subtracts ImageNet mean,
   divides by ImageNet standard deviation, rearranges dimensions to what
   PyTorch expects.
6. **The TorchScript model** runs a forward pass — the tensor flows through
   EfficientNet-B0's convolutional layers (learned during training to
   recognize visual patterns), ending in the single linear output layer,
   producing one raw floating-point number (e.g., `2.73`).
7. **`inference.py`** rounds/clips that number to a grade (`3`), computes a
   confidence score based on how close `2.73` is to `3` versus the boundary
   with `2`, and builds the Gaussian-shaped probability bars for display.
8. **`app.py`** displays the grade, confidence, DR-present yes/no, the
   clinical recommendation text (a static lookup keyed by grade), the
   probability bars, and — if confidence is below 60% — the manual-review
   warning banner.

Every step above mirrors exactly what happened to every training image
during Section 3-4 of the notebook (same preprocessing function, same
normalization constants, same image size) — which is the whole point: the
model only behaves correctly on inputs prepared the same way it learned from.

---

## 6. Glossary of terms used throughout

| Term | Meaning |
|---|---|
| **Transfer learning** | Starting from a model already trained on a different (usually larger) dataset/task, and fine-tuning it on your specific task instead of training from random initialization |
| **Fine-tuning** | Continuing to train a pretrained model's weights on new data, so it adapts to the new task |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | A small group of images processed together in one training step (here, 32 at a time) |
| **Loss function** | A mathematical measure of how wrong the model's predictions are; training tries to minimize this |
| **Gradient / backpropagation** | The process of calculating how much each internal weight contributed to the error, so it can be adjusted |
| **Overfitting** | When a model memorizes quirks of the training data instead of learning generalizable patterns, performing well on training data but poorly on new data |
| **Regularization** | Techniques (like Dropout, weight decay) that discourage overfitting |
| **Ordinal regression** | Predicting a single continuous number for an inherently ordered category set, instead of treating categories as unordered classes |
| **QWK (Quadratic Weighted Kappa)** | An evaluation metric that penalizes predictions proportionally to how far off they are from the true ordinal value |
| **TorchScript** | A serialized, portable format for PyTorch models that doesn't require the original model-defining code to load and run |
| **Inference** | Using an already-trained model to make a prediction on new data (as opposed to training) |
| **Tensor** | The fundamental data structure deep learning frameworks use — essentially a multi-dimensional array of numbers |

---

## Where to go next

- To actually run this: see the main [`README.md`](./README.md) for the
  step-by-step setup and run procedure.
- To modify the model architecture or training behavior: everything lives in
  `notebook/train_dr_model.ipynb`, Sections 5-6.
- To modify the app's behavior or UI: `app/app.py` (Streamlit) or
  `app/api.py` (FastAPI) — both depend on `app/inference.py` and
  `app/preprocessing.py`, which should stay in sync with the notebook's
  preprocessing if you ever change it there.
