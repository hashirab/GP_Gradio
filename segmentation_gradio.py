import torch
import cv2
import os
import tempfile
import numpy as np
from PIL import Image
from torchvision import transforms as T
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_w, resize_h = 640, 384
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std)
])
color_map = np.random.RandomState(42).randint(0, 255, size=(23, 3), dtype=np.uint8)

def apply_mask(image, mask):
    mask_color = color_map[mask]
    return cv2.addWeighted(image, 0.6, mask_color, 0.4, 0)

def load_segmentation_model(name):
    model = smp.Unet(
        encoder_name="mobilenet_v2" if name == "MobileNet" else "resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=23
    )
    path = "unet_mobilenet_final_50.pt" if name == "MobileNet" else "unet_resnet34_final_50.pt"
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def segment_video(video_file, model_name):
    model = load_segmentation_model(model_name)
    input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path.write(video_file.read())
    input_path.close()

    cap = cv2.VideoCapture(input_path.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(tempfile.gettempdir(), "segmentation_output.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (resize_w, resize_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            mask = torch.argmax(model(tensor), dim=1).squeeze().cpu().numpy()
        resized_mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        overlay = apply_mask(frame, resized_mask)
        out.write(overlay)

    cap.release()
    out.release()
    return out_path
