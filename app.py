import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES_DEFAULT = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MODEL_DIR = Path("models")
CHECKPOINT_CNN = MODEL_DIR / "cnn_scene.pth"
CHECKPOINT_RESNET = MODEL_DIR / "resnet18_scene.pth"


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 18 * 18, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@st.cache_resource
def load_model(model_type: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = CLASS_NAMES_DEFAULT
    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        class_names = checkpoint["class_names"]

    num_classes = len(class_names)

    if model_type == "CNN":
        model = SimpleCNN(num_classes=num_classes)
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_names, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image).unsqueeze(0)


def predict(model: nn.Module, image_tensor: torch.Tensor, device: torch.device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def main() -> None:
    st.set_page_config(page_title="Natural Scene Classifier", page_icon="🌿", layout="centered")

    st.title("Natural Scene Image Classifier")
    st.write("Upload an image and classify it as one of: buildings, forest, glacier, mountain, sea, or street.")

    st.sidebar.header("Model")
    model_type = st.sidebar.selectbox("Choose model", ["ResNet18", "CNN"])

    default_path = CHECKPOINT_RESNET if model_type == "ResNet18" else CHECKPOINT_CNN
    checkpoint_input = st.sidebar.text_input("Checkpoint path", value=str(default_path))

    checkpoint_exists = os.path.exists(checkpoint_input)

    if not checkpoint_exists:
        st.warning(
            "Checkpoint not found. Save your trained model to the path above, then refresh. "
            "Example files: models/resnet18_scene.pth or models/cnn_scene.pth"
        )
        st.code(
            "\n".join(
                [
                    "# Example save from notebook",
                    "import os, torch",
                    "os.makedirs('models', exist_ok=True)",
                    "torch.save({",
                    "    'model_state_dict': resnet_model.state_dict(),",
                    "    'class_names': class_names",
                    "}, 'models/resnet18_scene.pth')",
                ]
            ),
            language="python",
        )
        st.stop()

    try:
        model_key = "ResNet18" if model_type == "ResNet18" else "CNN"
        model, class_names, device = load_model(model_key, checkpoint_input)
        st.success(f"Loaded {model_type} checkpoint on {device}.")
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.stop()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict", type="primary"):
            image_tensor = preprocess_image(image)
            probs = predict(model, image_tensor, device)

            top_idx = int(probs.argmax())
            top_class = class_names[top_idx]
            top_conf = float(probs[top_idx])

            st.subheader("Prediction")
            st.write(f"Class: {top_class}")
            st.write(f"Confidence: {top_conf:.2%}")

            prob_table = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            st.subheader("Class probabilities")
            st.bar_chart(prob_table)


if __name__ == "__main__":
    main()
