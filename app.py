# app.py
import gradio as gr
from anomaly_gradio import detect_anomalies
from segmentation_gradio import segment_video

anomaly_app = gr.Interface(
    fn=detect_anomalies,
    inputs=gr.Video(label="Upload video for anomaly detection"),
    outputs=gr.Video(label="Processed anomaly video"),
    title="Anomaly Detection with Faster R-CNN"
)

segmentation_app = gr.Interface(
    fn=segment_video,
    inputs=[
        gr.Video(label="Upload video for segmentation"),
        gr.Radio(["MobileNet", "ResNet34"], label="Choose segmentation model")
    ],
    outputs=gr.Video(label="Segmented output video"),
    title="Semantic Segmentation with UNet"
)

demo = gr.TabbedInterface(
    [anomaly_app, segmentation_app],
    ["Anomaly Detection", "Segmentation"]
)

demo.launch()
