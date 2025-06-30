import torch
import cv2
import os
import tempfile
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from torchvision.transforms import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 14

def load_anomaly_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load("Anomaly/resnet50_final.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model

model = load_anomaly_model()

def draw_boxes(frame, outputs, threshold=0.5):
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label.item()}:{score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def detect_anomalies(video_file):
    input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path.write(video_file.read())
    input_path.close()

    cap = cv2.VideoCapture(input_path.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(tempfile.gettempdir(), "anomaly_output.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(img).to(device)
        with torch.no_grad():
            pred = model([tensor])[0]
        result = draw_boxes(frame, pred)
        out.write(result)

    cap.release()
    out.release()
    return out_path
