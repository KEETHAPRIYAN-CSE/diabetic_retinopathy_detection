# Model weights go here

This folder is intentionally empty in the repo. After training in
`notebook/train_dr_model.ipynb`, the notebook saves two files to your
Google Drive under `biothon2026-dr-detection/models/`:

- `dr_model_best.pt` — full PyTorch checkpoint (state_dict + metadata)
- `dr_model_traced.pt` — TorchScript-traced version (**this is the one the app uses**)

Download `dr_model_traced.pt` from Google Drive and place it here as:

```
app/model/dr_model_traced.pt
```

The app (`app.py` and `api.py`) will refuse to start cleanly without this file
and will tell you exactly this path if it's missing.
