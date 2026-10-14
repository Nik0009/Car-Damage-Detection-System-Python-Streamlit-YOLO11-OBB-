
# from pathlib import Path
# import io

# import cv2
# import numpy as np
# import pandas as pd
# import streamlit as st
# from PIL import Image
# from ultralytics import YOLO


# # ============================================================
# # CONFIGURATION
# # ============================================================

# ROOT = Path(__file__).resolve().parent
# MODEL_PATH = ROOT / "models" / "best.pt"

# st.set_page_config(
#     page_title="Car Damage Detection System",
#     page_icon="🚗",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# CLASS_NAMES = {
#     0: "crack",
#     1: "dent",
#     2: "glass shatter",
#     3: "lamp broken",
#     4: "scratch",
#     5: "tire flat",
# }


# # ============================================================
# # LOAD MODEL
# # ============================================================

# @st.cache_resource
# def load_model(model_path, modified_time):
#     return YOLO(model_path)


# # ============================================================
# # HEADER
# # ============================================================

# st.title("🚗 Car Damage Detection System")

# st.write(
#     "Upload a car image to identify damage using a trained "
#     "YOLO11 Oriented Bounding Box (OBB) model."
# )

# st.caption(
#     "Detectable classes: crack, dent, glass shatter, "
#     "lamp broken, scratch, and tire flat."
# )

# st.divider()


# # ============================================================
# # CHECK MODEL
# # ============================================================

# if not MODEL_PATH.is_file():
#     st.error(
#         "Trained model not found!\n\n"
#         f"Expected location: {MODEL_PATH}\n\n"
#         "Run train.py first to create models/best.pt."
#     )
#     st.stop()

# try:
#     model = load_model(
#         str(MODEL_PATH),
#         MODEL_PATH.stat().st_mtime,
#     )
# except Exception as exc:
#     st.error(f"Failed to load the trained model: {exc}")
#     st.stop()


# # ============================================================
# # SIDEBAR SETTINGS
# # ============================================================

# st.sidebar.title("⚙️ Detection Settings")

# confidence = st.sidebar.slider(
#     "Confidence threshold",
#     min_value=0.05,
#     max_value=0.95,
#     value=0.25,
#     step=0.05,
#     help="Lower the threshold to show more possible detections.",
# )

# image_size = st.sidebar.select_slider(
#     "Inference image size",
#     options=[256, 320, 416, 512, 640],
#     value=256,
# )

# st.sidebar.info(
#     "The inference image size can differ from the training "
#     "size. Larger sizes may take longer on a CPU."
# )


# # ============================================================
# # IMAGE UPLOAD
# # ============================================================

# uploaded_file = st.file_uploader(
#     "📤 Upload a car image",
#     type=["jpg", "jpeg", "png", "webp", "bmp"],
# )

# if uploaded_file is None:
#     st.info("Upload a car image to start damage detection.")
#     st.stop()


# # ============================================================
# # DISPLAY ORIGINAL IMAGE
# # ============================================================

# try:
#     image = Image.open(uploaded_file).convert("RGB")
#     image_array = np.array(image)

# except Exception as exc:
#     st.error(f"Could not read the uploaded image: {exc}")
#     st.stop()


# left, right = st.columns(2)

# with left:
#     st.subheader("Original Image")
#     st.image(image, use_container_width=True)

# with right:
#     st.subheader("Image Information")
#     st.write(f"**Filename:** {uploaded_file.name}")
#     st.write(f"**Width:** {image.width} pixels")
#     st.write(f"**Height:** {image.height} pixels")
#     st.write(f"**Color mode:** RGB")


# # ============================================================
# # RUN DETECTION
# # ============================================================

# if st.button("🔍 Detect Car Damage", type="primary"):

#     try:
#         with st.spinner("Analyzing image and drawing bounding boxes..."):

#             result = model.predict(
#                 source=image_array,
#                 conf=confidence,
#                 imgsz=image_size,
#                 device="cpu",
#                 verbose=False,
#             )[0]

#         st.divider()
#         st.header("🎯 Bounding Box Detection Result")

#         # Ultralytics draws oriented boxes, labels and confidence.
#         annotated_bgr = result.plot(
#     boxes=True,
#     labels=True,
#     conf=True,
# )

#         annotated_rgb = cv2.cvtColor(
#             annotated_bgr,
#             cv2.COLOR_BGR2RGB,
#         )

#         annotated_image = Image.fromarray(annotated_rgb)

#         st.image(
#             annotated_image,
#             caption="Detected damage regions with bounding boxes",
#             use_container_width=True,
#         )

#         # Prepare download buffer.
#         image_buffer = io.BytesIO()
#         annotated_image.save(image_buffer, format="JPEG")

#         st.download_button(
#             label="⬇️ Download Annotated Image",
#             data=image_buffer.getvalue(),
#             file_name="car_damage_bounding_boxes.jpg",
#             mime="image/jpeg",
#         )

#         # ====================================================
#         # EXTRACT OBB DETECTIONS
#         # ====================================================

#         detections = []

#         if result.obb is not None and len(result.obb) > 0:

#             corners = result.obb.xyxyxyxy.cpu().numpy()
#             class_ids = result.obb.cls.cpu().numpy().astype(int)
#             scores = result.obb.conf.cpu().numpy()

#             for index, class_id in enumerate(class_ids):

#                 class_name = CLASS_NAMES.get(
#                     int(class_id),
#                     result.names.get(int(class_id), str(class_id)),
#                 )

#                 points = corners[index].reshape(4, 2)

#                 row = {
#                     "Detection ID": index + 1,
#                     "Damage Type": class_name,
#                     "Confidence (%)": round(
#                         float(scores[index]) * 100, 2
#                     ),
#                 }

#                 # Coordinates of the four oriented-box corners.
#                 for point_index, (x, y) in enumerate(points, 1):
#                     row[f"Point {point_index} X"] = round(
#                         float(x), 2
#                     )
#                     row[f"Point {point_index} Y"] = round(
#                         float(y), 2
#                     )

#                 detections.append(row)

#         # ====================================================
#         # SUMMARY
#         # ====================================================

#         st.divider()
#         st.header("📊 Detection Summary")

#         if detections:

#             df = pd.DataFrame(detections)

#             metric1, metric2, metric3 = st.columns(3)

#             with metric1:
#                 st.metric(
#                     "Total Damage Regions",
#                     len(detections),
#                 )

#             with metric2:
#                 st.metric(
#                     "Damage Types",
#                     df["Damage Type"].nunique(),
#                 )

#             with metric3:
#                 st.metric(
#                     "Average Confidence",
#                     f'{df["Confidence (%)"].mean():.2f}%',
#                 )

#             st.subheader("Damage Counts")

#             summary = (
#                 df.groupby("Damage Type")
#                 .size()
#                 .reset_index(name="Count")
#                 .sort_values("Count", ascending=False)
#             )

#             st.dataframe(
#                 summary,
#                 use_container_width=True,
#                 hide_index=True,
#             )

#             st.subheader("Individual Bounding Boxes")

#             st.dataframe(
#                 df,
#                 use_container_width=True,
#                 hide_index=True,
#             )

#             # CSV report download.
#             csv_data = df.to_csv(index=False).encode("utf-8")

#             st.download_button(
#                 label="⬇️ Download Detection Report (CSV)",
#                 data=csv_data,
#                 file_name="car_damage_detection_report.csv",
#                 mime="text/csv",
#             )

#         else:
#             st.warning(
#                 "No damage was detected above the selected "
#                 "confidence threshold. Try a different image or "
#                 "lower the confidence threshold."
#             )

#         st.divider()

#         st.caption(
#             "This result is generated by a machine-learning model. "
#             "It may miss damage or produce false detections. "
#             "Confirm important damage assessments manually."
#         )

#     except Exception as exc:
#         st.error(f"Detection failed: {exc}")


# # ============================================================
# # FOOTER
# # ============================================================

# st.divider()
# st.caption(
#     "Car Damage Detection | Python · Streamlit · YOLO11 OBB"
# )




from pathlib import Path
import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"

DAMAGE_CLASSES = {
    0: "crack",
    1: "dent",
    2: "glass shatter",
    3: "lamp broken",
    4: "scratch",
    5: "tire flat",
}

DAMAGE_ICONS = {
    0: "💥",
    1: "🚗",
    2: "🪟",
    3: "💡",
    4: "🩹",
    5: "🛞",
}

st.set_page_config(
    page_title="Car Damage Detection System",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        color: #777777;
        margin-bottom: 1.5rem;
    }

    div.stButton > button {
        min-height: 60px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        border: 1px solid #d1d5db;
    }

    div.stButton > button:hover {
        border-color: #2563eb;
        color: #2563eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚘 Car Damage Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered vehicle damage detection using YOLO11 OBB'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model(model_path, modified_time):
    return YOLO(model_path)


if not MODEL_PATH.is_file():
    st.error(
        f"Trained model not found:\n\n{MODEL_PATH}"
    )
    st.info(
        "Place your trained best.pt file inside the models folder. "
        "If training saved it elsewhere, copy it to models/best.pt."
    )
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not load YOLO model: {exc}")
    st.stop()


# ============================================================
# 5. SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Lower values may show more detections, including false positives.",
)

image_size = st.sidebar.select_slider(
    "Inference image size",
    options=[256, 320, 416, 512, 640],
    value=256,
)

st.sidebar.caption(
    "Larger image sizes can take longer on a CPU."
)


# ============================================================
# 6. UPLOAD IMAGE
# ============================================================

st.subheader("📤 Upload Car Image")

uploaded_file = st.file_uploader(
    "Choose an image of a damaged vehicle",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is None:
    st.info("Upload a car image to enable the detection buttons.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
except Exception as exc:
    st.error(f"Unable to open the uploaded image: {exc}")
    st.stop()


# ============================================================
# 7. DISPLAY ORIGINAL IMAGE
# ============================================================

original_col, details_col = st.columns([3, 1])

with original_col:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

with details_col:
    st.subheader("Image Details")
    st.write(f"**File:** {uploaded_file.name}")
    st.write(f"**Width:** {image.width} px")
    st.write(f"**Height:** {image.height} px")
    st.write(f"**Model:** YOLO11 OBB")


# ============================================================
# 8. DETECTION FUNCTION
# ============================================================

def detect_damage(target_class_id):
    damage_name = DAMAGE_CLASSES[target_class_id]
    damage_icon = DAMAGE_ICONS[target_class_id]

    try:
        with st.spinner(f"Detecting {damage_name}..."):
            result = model.predict(
                source=image_array,
                conf=confidence,
                imgsz=image_size,
                device="cpu",
                verbose=False,
            )[0]

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    detections = []

    # --------------------------------------------------------
    # Extract oriented bounding boxes (YOLO OBB model)
    # --------------------------------------------------------

    if result.obb is not None and len(result.obb) > 0:
        corners = result.obb.xyxyxyxy.cpu().numpy()
        class_ids = result.obb.cls.cpu().numpy().astype(int)
        scores = result.obb.conf.cpu().numpy()

        for i, class_id in enumerate(class_ids):
            if int(class_id) != target_class_id:
                continue

            detections.append({
                "class_id": int(class_id),
                "class_name": DAMAGE_CLASSES.get(
                    int(class_id),
                    str(class_id),
                ),
                "confidence": float(scores[i]),
                "points": corners[i].reshape(4, 2),
            })

    # --------------------------------------------------------
    # Support standard YOLO bounding boxes too
    # --------------------------------------------------------

    elif result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()

        for i, class_id in enumerate(class_ids):
            if int(class_id) != target_class_id:
                continue

            x1, y1, x2, y2 = boxes[i]

            points = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ])

            detections.append({
                "class_id": int(class_id),
                "class_name": DAMAGE_CLASSES.get(
                    int(class_id),
                    str(class_id),
                ),
                "confidence": float(scores[i]),
                "points": points,
            })

    # --------------------------------------------------------
    # Display selected damage result
    # --------------------------------------------------------

    st.divider()
    st.header(f"{damage_icon} {damage_name.title()} Detection")

    if not detections:
        st.warning(
            f"No {damage_name} detected at the current confidence "
            "threshold. Try lowering the threshold or using another image."
        )
        return

    # Draw only the selected class's bounding boxes.
    annotated = image_array.copy()

    for item in detections:
        points = np.round(item["points"]).astype(np.int32)

        cv2.polylines(
            annotated,
            [points.reshape((-1, 1, 2))],
            isClosed=True,
            color=(255, 0, 0),
            thickness=3,
        )

        label = (
            f"{damage_name.title()} "
            f"{item['confidence'] * 100:.1f}%"
        )

        x = max(0, int(points[:, 0].min()))
        y = max(22, int(points[:, 1].min()) - 8)

        cv2.putText(
            annotated,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    st.image(
        annotated,
        caption=f"{damage_name.title()} bounding boxes only",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    average_confidence = np.mean([
        item["confidence"] for item in detections
    ]) * 100

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric("Damage Regions", len(detections))

    with metric2:
        st.metric("Average Confidence", f"{average_confidence:.2f}%")

    # --------------------------------------------------------
    # Build detection table
    # --------------------------------------------------------

    rows = []

    for index, item in enumerate(detections, start=1):
        row = {
            "Detection ID": index,
            "Damage Type": item["class_name"].title(),
            "Confidence (%)": round(
                item["confidence"] * 100, 2
            ),
        }

        for point_index, (x, y) in enumerate(item["points"], start=1):
            row[f"Point {point_index} X"] = round(float(x), 2)
            row[f"Point {point_index} Y"] = round(float(y), 2)

        rows.append(row)

    df = pd.DataFrame(rows)

    st.subheader("📊 Detection Details")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Download annotated image
    # --------------------------------------------------------

    image_buffer = io.BytesIO()
    Image.fromarray(annotated).save(
        image_buffer,
        format="JPEG",
    )

    safe_name = damage_name.replace(" ", "_")

    st.download_button(
        label=f"⬇️ Download {damage_name.title()} Image",
        data=image_buffer.getvalue(),
        file_name=f"{safe_name}_detection.jpg",
        mime="image/jpeg",
        key=f"download_image_{target_class_id}",
    )

    # --------------------------------------------------------
    # Download CSV report
    # --------------------------------------------------------

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"⬇️ Download {damage_name.title()} Report",
        data=csv_data,
        file_name=f"{safe_name}_report.csv",
        mime="text/csv",
        key=f"download_csv_{target_class_id}",
    )


# ============================================================
# 9. SIX SEPARATE BUTTONS WITH DAMAGE ICONS
# ============================================================

st.divider()
st.header("🔎 Select Damage to Detect")

st.write(
    "Click one button to run detection for only that damage category."
)

button_rows = [
    [
        (0, "💥", "Crack"),
        (1, "🚗", "Dent"),
    ],
    [
        (2, "🪟", "Glass Shatter"),
        (3, "💡", "Lamp Broken"),
    ],
    [
        (4, "🩹", "Scratch"),
        (5, "🛞", "Tire Flat"),
    ],
]

for row_index, button_row in enumerate(button_rows):
    columns = st.columns(2)

    for column, (class_id, icon, name) in zip(columns, button_row):
        with column:
            if st.button(
                f"{icon} Detect {name}",
                key=f"detect_button_{class_id}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["selected_damage"] = class_id


# ============================================================
# 10. SHOW THE SELECTED RESULT
# ============================================================

selected_damage = st.session_state.get("selected_damage")

if selected_damage in DAMAGE_CLASSES:
    detect_damage(selected_damage)


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()
st.caption(
    "Car Damage Detection System | Python | Streamlit | YOLO11 OBB"
)
st.caption(
    "AI predictions may be inaccurate. Verify important damage "
    "assessments manually."
)
