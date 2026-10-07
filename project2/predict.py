"""
predict.py

Standalone prediction script for the traffic vehicle classification project.
Runs the two-stage cascade pipeline (coarse ResNet18 + cascade ResNet18 for
kamyun/kamyunet) on a single image and writes a JSON result.

Usage (command line):
    python predict.py --image path/to/image.jpg

Usage (as an importable function):
    from predict import predict_single_image
    result = predict_single_image("path/to/image.jpg")
"""

# cd "C:\Users\Lenovo\Desktop\project 2\project2"
# python predict.py --image "14_resNet/newData/dataTest/test/vanet/File_Name.jpg"

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# paths 
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR / "14_resNet" 
MODELS_DIR = BASE_DIR / "saved_models"

COARSE_MODEL_PATH = MODELS_DIR / "coarse_resnet18_fine_tuning_weighted.pt"
CASCADE_MODEL_PATH = MODELS_DIR / "cascade_resnet18_kamyun_kamyunet.pt"
THRESHOLD_PATH = MODELS_DIR / "confidence_threshold.json"

OUTPUT_JSON_DIR = SCRIPT_DIR 

# fixed configuration (must match training exactly)
coarse_class_to_idx = {
    "ambulance": 0, "autobus": 1, "minibus": 2, "savari": 3,
    "taxi": 4, "truck_merged": 5, "vanet": 6,
}
coarse_idx_to_class = {idx: name for name, idx in coarse_class_to_idx.items()}

cascade_class_to_idx = {"kamyun": 0, "kamyunet": 1}
cascade_idx_to_class = {idx: name for name, idx in cascade_class_to_idx.items()}

inference_transform  = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

# model architecture (must match training exactly)
def build_resnet18(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_models(device):
    coarse_model = build_resnet18(len(coarse_class_to_idx))
    coarse_model.load_state_dict(torch.load(COARSE_MODEL_PATH, map_location=device))
    coarse_model = coarse_model.to(device)
    coarse_model.eval()

    cascade_model = build_resnet18(len(cascade_class_to_idx))
    cascade_model.load_state_dict(torch.load(CASCADE_MODEL_PATH, map_location=device))
    cascade_model = cascade_model.to(device)
    cascade_model.eval()

    return coarse_model, cascade_model

def load_confidence_threshold():
    with open(THRESHOLD_PATH, "r") as f:
        record = json.load(f)
    return record["confidence_threshold"]

# inference
def predict_vehicle(image, coarse_model, cascade_model, device, confidence_threshold):
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    # first model
    with torch.no_grad():
        coarse_logits = coarse_model(input_tensor)
        coarse_probs = F.softmax(coarse_logits, dim=1).squeeze(0).cpu().numpy()

    coarse_pred_idx = int(coarse_probs.argmax())
    coarse_pred_class = coarse_idx_to_class[coarse_pred_idx]
    coarse_confidence = float(coarse_probs[coarse_pred_idx])

    result = {
        "coarse_prediction": coarse_pred_class,
        "coarse_probs": {coarse_idx_to_class[i]: float(p) for i, p in enumerate(coarse_probs)},
        "cascade_probs": None,
    }

    # if it is not a truck, we are done
    if coarse_pred_class != "truck_merged":
        result["predicted_class"] = coarse_pred_class
        result["confidence"] = coarse_confidence
        result["needs_review"] = coarse_confidence < confidence_threshold
        return result

    # second model  
    with torch.no_grad():
        cascade_logits = cascade_model(input_tensor)
        cascade_probs = F.softmax(cascade_logits, dim=1).squeeze(0).cpu().numpy()

    cascade_pred_idx = int(cascade_probs.argmax())
    cascade_pred_class = cascade_idx_to_class[cascade_pred_idx]
    cascade_confidence = float(cascade_probs[cascade_pred_idx])

    final_confidence = coarse_confidence * cascade_confidence

    result["predicted_class"] = cascade_pred_class
    result["confidence"] = final_confidence
    result["cascade_probs"] = {cascade_idx_to_class[i]: float(p) for i, p in enumerate(cascade_probs)}
    result["needs_review"] = final_confidence < confidence_threshold
    return result

# ---------------------------------------------------------
def predict_single_image(image_path, save_json=True):
    """
    کل پایپ لاین را روی یک تصویر اجرا می‌کند و نتیجه را برمی‌گرداند.
    If save_json=True, also writes a JSON file to project2/ named after
    the input image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coarse_model, cascade_model = load_models(device)
    confidence_threshold = load_confidence_threshold()

    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")

    prediction = predict_vehicle(image, coarse_model, cascade_model, device, confidence_threshold)

    output = {
        "filepath": str(image_path),
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "needs_review": prediction["needs_review"],
        "coarse_prediction": prediction["coarse_prediction"],
        "coarse_probs": prediction["coarse_probs"],
        "cascade_probs": prediction["cascade_probs"],
        "confidence_threshold_used": confidence_threshold,
        "model_paths": {
            "coarse_model": str(COARSE_MODEL_PATH),
            "cascade_model": str(CASCADE_MODEL_PATH),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_used": str(device),
    }

    if save_json:
        json_filename = f"prediction_{image_path.stem}.json"
        json_path = OUTPUT_JSON_DIR / json_filename
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        output["_saved_json_path"] = str(json_path)

    return output

# command-line interface
def main():
    parser = argparse.ArgumentParser(description="Predict vehicle class for a single image.")
    parser.add_argument("--image", type=str, required=False, default=None, help="Path to the input image.")
    parser.add_argument("--no-save", action="store_true", help="Do not write a JSON output file.")
    args = parser.parse_args()

    if args.image is None:
        print("No --image argument given. Example usage:")
        print('  python predict.py --image "path/to/image.jpg"')
        return
    
    result = predict_single_image(args.image, save_json=not args.no_save)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
 