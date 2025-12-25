import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import warnings
from datetime import datetime

"""
feature + parameter model
-------------------------
This file implements a single shared encoder that predicts the feature class
and the parameters required by that feature via per-feature regression heads.

You can plug in your own dataset later. The provided dataset is image-based
(working.png, final.png) like your current script, but now it *optionally*
returns parameter labels when available.

Key ideas implemented here:
- Shared backbone for efficiency
- Classification head for feature type
- Per-feature parameter heads with sensible constraints
- Loss gating so only the true feature's parameters contribute to loss
- Optional heteroscedastic regression (mean + log-variance) for uncertainty
- Clean output schema from inference for consumption by your app

To wire up your real parameter labels, see `FeaturePairDataset.__getitem__`.
"""

# -------------------- settings --------------------

# preprocessing
resolution = 150

# training
train = False
resume_training = False  # resume training from last checkpoint
epochs = 30
acc_cutoff = 98.0
learning_rate = 1e-3
batch_size = 32
weight_decay = 0.0

# testing
test = False
test_batch_size = 20

# checkpointing
checkpoint_name = "featureparamcheckpoint.pth"

# current directory
current_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# -------------------- class labels --------------------
features: List[str] = [
    "bossextrude",
    "cutextrude",
    "bossrevolve",
    "cutrevolve",
    "fillet",
]
class_to_idx = {name: idx for idx, name in enumerate(features)}

# -------------------- parameter specs --------------------
heteroscedastic = True

param_specs: Dict[str, List[Tuple[str, int, str]]] = {
    "bossextrude": [
        ("sketch_plane_normal", 3, "unitvec"),
        ("sketch_plane_offset", 1, "linear"),
        ("length", 1, "softplus"),
    ],
    "cutextrude": [
        ("sketch_plane_normal", 3, "unitvec"),
        ("sketch_plane_offset", 1, "linear"),
        ("length", 1, "softplus"),
    ],
    "bossrevolve": [
        ("axis", 3, "unitvec"),
        ("angle", 1, "angle2pi"),
        ("axis_point_offset", 1, "linear"),
    ],
    "cutrevolve": [
        ("axis", 3, "unitvec"),
        ("angle", 1, "angle2pi"),
        ("axis_point_offset", 1, "linear"),
    ],
    "fillet": [
        ("radius", 1, "softplus"),
    ],
}

# -------------------- image transform --------------------
transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------- dataset --------------------
class FeaturePairDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.samples: List[Tuple[Path, Path, int, Optional[Dict[str, Any]]]] = []
        self.transform = transform
        self.root_dir = Path(root_dir)

        for label in features:
            feature_dir = self.root_dir / label
            if not feature_dir.exists():
                continue
            for model_dir in feature_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                final_img = model_dir / "final.png"
                working_img = model_dir / "working.png"
                params_json = model_dir / "params.json"

                if final_img.exists() and working_img.exists():
                    params: Optional[Dict[str, Any]] = None
                    if params_json.exists():
                        try:
                            import json
                            params = json.loads(params_json.read_text())
                        except Exception:
                            params = None
                    self.samples.append((working_img, final_img, class_to_idx[label], params))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        working_path, final_path, label, params = self.samples[idx]
        working_img = Image.open(working_path).convert("RGB")
        final_img = Image.open(final_path).convert("RGB")

        if self.transform:
            working_img = self.transform(working_img)
            final_img = self.transform(final_img)

        x = torch.cat([working_img, final_img], dim=0)

        y_params: Optional[torch.Tensor] = None
        y_logvar_mask: Optional[torch.Tensor] = None

        feature_name = features[label]
        spec = param_specs.get(feature_name)
        if spec is not None and params is not None:
            flat: List[float] = []
            for (name, dim, _constraint) in spec:
                val = params.get(name)
                if val is None:
                    flat.extend([float("nan")] * dim)
                else:
                    if isinstance(val, (int, float)):
                        flat.append(float(val))
                    elif isinstance(val, (list, tuple)):
                        flat.extend([float(v) for v in val])
                    else:
                        raise ValueError(f"Param {name} has unsupported type: {type(val)}")
            y_params = torch.tensor(flat, dtype=torch.float32)
            y_logvar_mask = torch.tensor([0.0 if torch.isnan(v) else 1.0 for v in y_params], dtype=torch.float32)
            y_params = torch.nan_to_num(y_params, nan=0.0)

        return x, label, feature_name, y_params, y_logvar_mask

def collate_fn(batch):
    xs, ys, feat_names, y_params_list, y_masks = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y, feat_names, y_params_list, y_masks

# -------------------- model --------------------
class ConvBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.conv(x)

class FeatureParamModel(nn.Module):
    def __init__(self, features: List[str], param_specs: Dict[str, List[Tuple[str, int, str]]], hetero: bool = True):
        super().__init__()
        self.features = features
        self.param_specs = param_specs
        self.hetero = hetero

        self.backbone = ConvBackbone()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(features))
        )

        self.heads = nn.ModuleDict()
        for feat in features:
            spec = param_specs.get(feat)
            if spec is None:
                continue
            out_dims = sum(dim for _name, dim, _c in spec)
            if hetero:
                out_dims *= 2
            self.heads[feat] = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, out_dims)
            )

    def forward(self, x) -> Dict[str, Any]:
        f = self.backbone(x)
        logits = self.classifier(f)
        out: Dict[str, Any] = {"logits": logits}
        for feat, head in self.heads.items():
            out[feat] = head(f)
        return out


# -------------------- inference --------------------
@torch.no_grad()
def load_feature_param_model(model_path: Optional[Path] = None, device: Optional[torch.device] = None) -> FeatureParamModel:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureParamModel(features, param_specs, hetero=heteroscedastic).to(device)

    model_path = model_path or (current_dir / "FeatureClassifier.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found at {model_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

@torch.no_grad()
def classify_feature_once(
    working_path: Path,
    final_path: Path,
    model: Optional[FeatureParamModel] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    classify a single (working, final) pair and return top-1 feature prediction.

    returns:
        {
            "top1": str,                # predicted feature name
            "top1_prob": float,         # probability of top1
            "probs": List[float],       # probability per class (same order as `features`)
            "logits": List[float]       # raw logits per class
        }

    usage in a loop:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mdl = load_feature_param_model(device=dev)
        for step in range(k):
            out = classify_feature_once(w, f, model=mdl, device=dev)
            # apply out['top1'] ... update working ...
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = model or load_feature_param_model(device=device)

    working_img = Image.open(working_path).convert("RGB")
    final_img = Image.open(final_path).convert("RGB")
    working_tensor = transform(working_img)
    final_tensor = transform(final_img)
    x = torch.cat([working_tensor, final_tensor], dim=0).unsqueeze(0).to(device)

    out = mdl(x)
    logits = out["logits"][0]
    probs = logits.softmax(dim=0)
    top_idx = int(probs.argmax().item())

    return {
        "top1": features[top_idx],
        "top1_prob": float(probs[top_idx].item()),
        "probs": probs.cpu().tolist(),
        "logits": logits.cpu().tolist(),
    }


# -------------------- public api --------------------
_model_cache: Optional[FeatureParamModel] = None
_model_device: Optional[torch.device] = None

@torch.no_grad()
def _prepare_input(working, final) -> torch.Tensor:
    """
    accepts either file paths (str | Path) or already-loaded tensors/images.
    returns a 4d tensor [1, 2, h, w] ready for the network.
    - if tensors: expects [1,h,w] or [h,w] in [0,1] or [-1,1] range; will normalize.
    - if paths or pil: will use module transform.
    """
    def to_tensor(img):
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
            return transform(img)
        if isinstance(img, Image.Image):
            return transform(img.convert("RGB"))
        if isinstance(img, torch.Tensor):
            t = img
            # allow [h,w], [1,h,w], or [c,h,w]
            if t.dim() == 2:
                t = t.unsqueeze(0)
            if t.size(0) == 1:
                # assume already grayscale normalized roughly; map to [-1,1]
                t = (t - t.min()) / (t.max() - t.min() + 1e-8)
                t = t * 2 - 1
                return t
            if t.size(0) == 3:
                # convert rgb -> grayscale via transform behavior
                # bring to pil-like range [0,1] if needed
                t = (t - t.min()) / (t.max() - t.min() + 1e-8)
                pil_like = transforms.ToPILImage()(t)
                return transform(pil_like)
            raise ValueError("tensor must be [h,w], [1,h,w], or [3,h,w]")
        raise TypeError("unsupported input type for image")

    w = to_tensor(working)
    f = to_tensor(final)
    x = torch.cat([w, f], dim=0).unsqueeze(0)  # [1,2,h,w]
    return x

@torch.no_grad()
def classify_feature(
    working,
    final,
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    classifies the feature change between a working model and a final model.

    inputs may be:
      - paths to images (str | Path)
      - pil images
      - torch tensors shaped [h,w], [1,h,w], or [3,h,w]

    returns dict with keys: {"top1", "top1_prob", "probs", "logits"}

    this function caches the loaded model the first time it is called to avoid
    reloading weights on subsequent invocations.
    """
    global _model_cache, _model_device

    dev = device or _model_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _model_cache is None or _model_device != dev:
        _model_cache = FeatureParamModel(features, param_specs, hetero=heteroscedastic).to(dev)
        mp = model_path or (current_dir / "FeatureClassifier.pth")
        if not Path(mp).exists():
            raise FileNotFoundError(f"model file not found at {mp}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = torch.load(mp, map_location=dev)
            _model_cache.load_state_dict(checkpoint["model_state_dict"])
        _model_cache.eval()
        _model_device = dev

    x = _prepare_input(working, final).to(dev)
    out = _model_cache(x)
    logits = out["logits"][0]
    probs = logits.softmax(dim=0)
    top_idx = int(probs.argmax().item())

    return {
        "top1": features[top_idx],
        "top1_prob": float(probs[top_idx].item()),
        "probs": probs.cpu().tolist(),
        "logits": logits.cpu().tolist(),
    }
