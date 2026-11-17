# MindTrace NG: algorithm, flow and build plan

## 1. What the write-up requires

| Write-up section | Requirement | Where it is implemented |
| --- | --- | --- |
| 1.3.2, 3.12.3 | Random Forest, binary AD / No AD | `train_model.py`, `mindtrace/ml/classifier.py` |
| 2.2.2, 3.12.2 | Validate, grayscale, resize, normalise | `mindtrace/ml/preprocess.py` |
| 2.2.3, 2.7 | GLCM texture + intensity statistics | `mindtrace/ml/features.py` |
| 2.2.5, 2.10 | Model-level feature importance (not a heatmap) | report page, performance page |
| 3.12.4 | Predicted class + class probability | report page |
| 3.12.6 | MySQL: users, prediction records, date and time | `mindtrace/models.py`, `database/schema.sql` |
| 3.12.15 | Login, upload, validation, history, errors, logout | `mindtrace/routes.py` |
| 3.17 | Accuracy, precision, recall, F1, confusion matrix, ROC-AUC | `models/metrics.json`, `/performance` page |
| 3.18 | Unit, integration, system, performance tests, TC01 to TC12 | `tests/test_app.py` |

## 2. Algorithm

### 2.1 Validation
1. Extension is png, jpg, jpeg or bmp; size at most 8 MB.
2. OpenCV can decode the bytes.
3. Shorter side is at least 64 px.
4. Colour channels are nearly equal (grayscale).
5. Standard deviation of gray values is at least 8 (not blank).
6. Border is dark and the centre is brighter (a head on a black background).

### 2.2 Preprocessing
```
gray  = BGR -> grayscale
gray  = resize(gray, 128 x 128, area interpolation)
gray  = round((gray - min) / (max - min) * 255)        # min-max normalisation
q     = floor(gray * 32 / 256)                         # 32 gray levels, for GLCM only
```

### 2.3 Feature extraction (69 values)
**GLCM texture, 42 values.** For each distance d in {1, 2, 4} and each direction
(0, 45, 90, 135 degrees):
```
P[i][j] = number of pixel pairs (i, j) separated by the offset
P       = (P + P^T) / sum(P + P^T)                     # symmetric, normalised
contrast      = sum P[i][j] * (i - j)^2
dissimilarity = sum P[i][j] * |i - j|
homogeneity   = sum P[i][j] / (1 + (i - j)^2)
ASM           = sum P[i][j]^2          energy = sqrt(ASM)
correlation   = sum P[i][j] * (i - mu_i)(j - mu_j) / (sigma_i * sigma_j)
entropy       = -sum P[i][j] * log2 P[i][j]
```
Each descriptor is stored as its mean over the four directions (21 values) and its
spread, max minus min, over the four directions (21 values).

**Intensity statistics, 27 values**, on tissue pixels (gray above 20): mean, standard
deviation, skewness, kurtosis, median, 10th and 90th percentiles, interquartile range,
histogram entropy, tissue-area fraction, bright-pixel fraction, and a 16-bin histogram.

### 2.4 Training
```
load 6,400 slices; drop duplicates and unreadable files
label: No AD = Non-demented;  AD = Very mild + Mild + Moderate
X = features(preprocess(image)) for every image
split 70 / 15 / 15, stratified by disease stage, seed 42
for each of 24 parameter sets (trees, depth, leaf size, features per split):
    fit on train, score on validation
keep the best; 5-fold cross-validate it on the training set
evaluate once on the test set -> accuracy, precision, recall, F1, specificity, ROC-AUC, confusion matrix
save rf_model.joblib + metrics.json
```
Random Forest (Breiman 2001): each tree is grown on a bootstrap sample of the training
rows and considers a random subset of features at every split; the forest's class
probability is the average of the trees' votes; class = AD if P(AD) >= 0.5.

### 2.5 Prediction (web request)
```
bytes -> validate -> preprocess -> 69 features -> forest.predict_proba
      -> class, probability, per-stage timings
      -> save MRIImage + Prediction rows, save the 128 x 128 image
      -> result page: class, probability, top-8 feature importance
```

## 3. System flow (Fig. 3.2)

Login -> MRI upload -> image validation -> preprocessing -> GLCM + intensity features
-> Random Forest -> AD / No AD -> class probability -> feature importance
-> results display -> prediction storage -> history -> logout

Invalid login returns to the login form with an error. An invalid image returns to the
upload form with the reason. Nothing is stored for a rejected image.

## 4. Database (Fig. 3.8)

`users` 1 to many `mri_images` 1 to 1 `predictions`. Passwords are stored as salted
hashes. Deleting an image deletes its prediction and file. Every query for a result,
image or history row is filtered by the signed-in user's id.

## 5. Pages

| Page | Purpose |
| --- | --- |
| `/` | Landing: live sample viewer, six-step method, sample report, evidence, zone browser, questions |
| `/signup`, `/signin` | Account with role, facility and state |
| `/overview` | Totals, latest records, class mix, scans by geopolitical zone, model summary |
| `/scan` | Three-step upload with drag and drop, preview and client-side checks |
| `/report/<id>` | Outcome banner with probability meter; tabs for summary, feature importance and all 69 values; print view |
| `/records` | Record cards with search and class filter, delete, CSV export (`/records.csv`) |
| `/performance` | Metrics board, confusion matrix, ROC curve, importance chart, per-stage accuracy, limitations |

## 6. Front-end design

- **Rules taken from the ui-ux-pro-max skill data** (healthcare and medical rows):
  accessible, minimal, trust-first; body text 16 px or larger; 4.5:1 contrast; visible
  focus; class never shown by colour alone (icon + text on every tag); reduced-motion
  support; no purple and pink "AI" gradients.
- **Identity:** adire indigo `#27348B` and night navy on ivory, with saffron `#F5A524`
  as the single signal colour; rose for AD, teal for No AD; a small green-white-green
  flag chip marks the Nigerian focus.
- **Type:** Bricolage Grotesque for headings, DM Sans for text, IBM Plex Mono for
  measurements and labels. All bundled locally.
- **Layout:** public pages use a dark top bar; the workspace uses a fixed sidebar on
  desktop and a bottom tab bar on phones. Hairline borders and 6 to 14 px corners.
- **Effects (written from scratch in CSS and vanilla JS, after the style of popular
  21st.dev components):** falling light beams on a dot grid, tilting viewer window,
  marquee ticker, drawn underline on the headline, marching-ants drop zone, count-up
  numbers, tabbed panels, animated ROC trace and bar growth.
- **Framer / Awwwards patterns:** editorial numbered list for the method, oversized
  metric grid, product-window hero showing real pipeline output, accordion questions.

## 7. Work plan (Agile iterations, 3.13)

| Iteration | Output | Status |
| --- | --- | --- |
| 1 | Requirements from Chapters 1 to 3, dataset check | done |
| 2 | Preprocessing + feature extractor, verified against scikit-image | done |
| 3 | Training, tuning, evaluation, saved model and metrics | done |
| 4 | Flask app, database, authentication, upload and result flow | done |
| 5 | Front-end, responsive at 390 px and 1440 px | done |
| 6 | 22 automated tests, browser walkthrough of every page | done |
| 7 | Deployment files, README, write-up corrections | done |
