# 🚘 Car Damage Detection System

An AI-powered web application that detects vehicle damage using **Python, Streamlit, OpenCV, and YOLO11 Oriented Bounding Boxes (OBB)**.

The application supports six damage categories:

- 💥 Crack
- 🚗 Dent
- 🪟 Glass Shatter
- 💡 Lamp Broken
- 🩹 Scratch
- 🛞 Tire Flat

## ✨ Features

- Upload vehicle images in JPG, JPEG, PNG, WEBP, or BMP format.
- Detect individual types of vehicle damage.
- Draw bounding boxes around detected damage.
- Display confidence scores and detection coordinates.
- Download annotated images.
- Export detection reports as CSV files.
- Adjust the confidence threshold and inference image size.
- Deploy the application using GitHub and Streamlit Community Cloud.

## 📁 Project Structure

```text
CarDamageDetection/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── best.pt
├── dataset/                  # Optional: training dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── outputs/                  # Optional: training results
```

**Important:** The trained model `models/best.pt` must exist before running the application. The dataset and training outputs are not required for normal inference.

## 🛠️ Requirements

- Python 3.10 or 3.11, compatible with the installed packages
- pip
- Git
- A trained YOLO11 OBB model
- Internet connection for installing packages

## 1. Clone the GitHub Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/car-damage-detection-system-python-streamlit-yolo11-obb-.git
```

Move into the project directory:

```bash
cd car-damage-detection-system-python-streamlit-yolo11-obb-
```

If you already have the project on your computer, open a terminal in that existing folder instead.

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

Create `requirements.txt` with the following contents:

```text
streamlit
ultralytics
opencv-python-headless
numpy
pandas
Pillow
```

Install the dependencies.

### Windows PowerShell

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are not activating the virtual environment, run:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Note:** Do not install both `opencv-python` and `opencv-python-headless` in the same environment. Both packages provide the `cv2` module.

## 4. Prepare the Trained Model

The application expects the model at:

```text
models/best.pt
```

### Windows PowerShell

Create the models folder if necessary:

```powershell
New-Item -ItemType Directory -Force -Path models
```

If your trained checkpoint is located in `outputs/quick_train/weights/best.pt`, copy it:

```powershell
Copy-Item ".\outputs\quick_train\weights\best.pt" ".\models\best.pt"
```

If the checkpoint is in another directory, replace the source path with the actual location.

Verify the model:

```powershell
Test-Path ".\models\best.pt"
```

Expected output:

```text
True
```

### Linux or macOS

```bash
mkdir -p models
cp outputs/quick_train/weights/best.pt models/best.pt
ls -lh models/best.pt
```

Adjust the source path if your training saved the model elsewhere.

## 5. Run the Application

Start Streamlit from the project root directory:

```bash
python -m streamlit run app.py
```

On Windows, if you want to use the virtual environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit normally opens the application in your browser at:

```text
http://localhost:8501
```

To stop the server, press `Ctrl+C` in the terminal.

## 6. Train the YOLO11 OBB Model (Optional)

Training is only necessary if you need to train a new model. You can skip this section if you already have `models/best.pt`.

### Dataset Format

The dataset should have separate image and label directories:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

For YOLO OBB datasets, each annotated object normally has a class ID followed by eight normalized coordinate values representing the four polygon corners.

Example:

```text
1 0.45 0.30 0.55 0.30 0.58 0.42 0.46 0.43
```

Use the dataset's actual annotation format and class IDs.

### Create `data.yaml`

Example configuration:

```yaml
path: dataset
train: train/images
val: valid/images
test: test/images

names:
  0: crack
  1: dent
  2: glass shatter
  3: lamp broken
  4: scratch
  5: tire flat
```

The paths must match the actual dataset location.

### Create `train.py`

```python
from ultralytics import YOLO

model = YOLO("yolo11n-obb.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    workers=0,
    device="cpu",
    project="outputs",
    name="car_damage",
)
```

Run training:

```bash
python train.py
```

The best checkpoint is typically saved to:

```text
outputs/car_damage/weights/best.pt
```

Copy it to the application model directory.

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path models
Copy-Item ".\outputs\car_damage\weights\best.pt" ".\models\best.pt"
```

**Important:** Training with a small number of epochs or a small dataset fraction is suitable for initial testing, not necessarily for reliable real-world damage detection. Evaluate the model on separate validation and test images before using its predictions for important decisions.

## 7. Test the Model

You can test the trained model directly with Python:

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source="test_car.jpg",
    conf=0.25,
    imgsz=256,
    device="cpu",
    save=True,
)

print("Prediction completed.")
```

Replace `test_car.jpg` with the path to your test image.

Run the script if you save it as `test_model.py`:

```bash
python test_model.py
```

Review the prediction output and verify that the model detects the correct damage categories.

## 8. Upload Changes to GitHub

Check repository status:

```bash
git status
```

Add the application files:

```bash
git add app.py requirements.txt README.md
git add models/best.pt
```

Commit the changes:

```bash
git commit -m "Update car damage detection application"
```

Push to GitHub:

```bash
git push origin main
```

If your repository uses a different branch, replace `main` with that branch's name.

### Important Git Notes

- Do not commit `.venv/`, Python cache files, or unnecessary training images.
- GitHub blocks ordinary Git files larger than 100 MB. Use Git LFS or suitable external model hosting for larger checkpoints.
- Do not upload passwords, API keys, or private credentials.
- Keep `models/best.pt` accessible to the deployed application.

## 9. Deploy on Streamlit Community Cloud

1. Push your project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using your GitHub account.
4. Select **Create app**.
5. Choose your repository and the `main` branch.
6. Set the main file path to `app.py`.
7. Select a compatible Python version in the advanced settings if required.
8. Click **Deploy**.

Your application will receive a URL similar to:

```text
https://your-app-name.streamlit.app
```

This is an example URL; Streamlit will provide the actual address after deployment.

### Streamlit Cloud Troubleshooting

**Error: `ModuleNotFoundError: No module named 'cv2'`**

Check that `requirements.txt` contains:

```text
opencv-python-headless
```

Confirm that the file is in the repository root, commit the changes, and allow Streamlit to reinstall dependencies.

**Error: Model file not found**

Confirm that the model exists at:

```text
models/best.pt
```

The path and filename are case-sensitive in the Linux deployment environment.

**Error: Application crashes or runs slowly**

- Check the full Streamlit deployment logs.
- Test the app with one image at a time.
- Reduce inference image size if needed.
- Check available memory and model size.

**Error: GitHub rejects the model file**

Check whether `best.pt` exceeds GitHub's 100 MB ordinary file limit. Use Git LFS or host the model separately and update the application to retrieve it securely.

## 🔬 Technologies Used

- Python
- Streamlit
- Ultralytics YOLO11 OBB
- OpenCV
- NumPy
- Pandas
- Pillow
- Git and GitHub
- Streamlit Community Cloud

## ⚠️ Disclaimer

This application provides AI-generated visual predictions and is intended for demonstration and research. It may miss damage or incorrectly classify objects. Its predictions should not replace a professional vehicle inspection or an insurance assessment.

## 👨‍💻 Project Summary

**Project:** Car Damage Detection System

**Platform:** Web application

**Language:** Python

**Framework:** Streamlit

**Model:** YOLO11 OBB

**Output:** Annotated images, damage classifications, confidence scores, and downloadable CSV reports
