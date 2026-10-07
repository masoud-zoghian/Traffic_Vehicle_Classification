# Libraries Used

This document lists all libraries and modules used in this project, separated into third-party packages (must be installed via `pip`) and Python standard library modules (included with Python, no installation needed).

## Third-Party Libraries

These must be installed (see `requirements.txt`):

| Package | Used For |
|---|---|
| `torch` | Core deep learning framework (`torch.nn`, `torch.nn.functional`, `torch.utils.data`) |
| `torchvision` | Pretrained models and image transforms (`torchvision.transforms`, `torchvision.models`) |
| `numpy` | Numerical operations |
| `pandas` | Data handling and tabular data |
| `matplotlib` | Plotting (`matplotlib.pyplot`) |
| `scikit-learn` | Metrics and model selection (`sklearn.metrics`, `sklearn.model_selection`) |
| `Pillow` | Image loading and processing (`PIL.Image`) |

## Python Standard Library

These come bundled with Python and require no installation:

| Module | Used For |
|---|---|
| `os` | File system operations |
| `collections` | `defaultdict`, `Counter` |
| `copy` | Object copying |
| `time` | Timing operations |
| `random` | Random number generation |
| `json` | Reading/writing JSON files |
| `pathlib` | `Path` — file path handling |
| `hashlib` | Hashing (e.g. for file/data integrity checks) |
| `argparse` | Command-line argument parsing |
| `datetime` | `datetime`, `timezone` — date/time handling |

## Installation

To install all third-party libraries at once:

```bash
pip install -r requirements.txt
```
