import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="螺栓缺陷AI检测", layout="centered")
st.title("🔩 螺栓缺陷AI视觉检测系统")
st.markdown("上传图片或使用摄像头实时检测螺栓缺陷（正常、锈蚀、划痕、变形、螺纹损坏）")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.conf_threshold = st.sidebar.slider("置信度阈值", 0.1, 0.9, 0.25, 0.05)
        self.resize_factor = st.sidebar.slider("图像缩放比例 (提高速度)", 0.2, 1.0, 0.5, 0.05)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        new_w = int(w * self.resize_factor)
        new_h = int(h * self.resize_factor)
        img_small = cv2.resize(img, (new_w, new_h))
        results = self.model(img_small, conf=self.conf_threshold)
        annotated_small = results[0].plot()
        annotated = cv2.resize(annotated_small, (w, h))
        return annotated

st.sidebar.header("设置")
confidence = st.sidebar.slider("置信度阈值", 0.1, 0.9, 0.25, 0.05)
mode = st.sidebar.radio("检测模式", ["图片上传", "实时摄像头"])

if mode == "图片上传":
    uploaded_file = st.file_uploader("上传一张螺栓图片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = model(img, conf=confidence)
        annotated = results[0].plot()
        st.image(annotated, channels="BGR", caption="检测结果")
        if len(results[0].boxes) > 0:
            st.success(f"检测到 {len(results[0].boxes)} 个目标")
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf)
                class_name = model.names[cls]
                st.write(f"- {class_name}: {conf:.2f}")
        else:
            st.warning("未检测到螺栓")
else:
    st.markdown("### 实时摄像头检测")
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=lambda: VideoTransformer(),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if webrtc_ctx.state.playing:
        st.info("摄像头已开启，正在实时检测...")
    else:
        st.warning("请点击“开始”按钮并允许摄像头权限。")