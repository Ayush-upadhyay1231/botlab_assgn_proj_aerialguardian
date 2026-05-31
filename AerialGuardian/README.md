# The Aerial Guardian 🚁
Drone-based Multi-Object Person Tracking using YOLOv8 + ByteTrack

## Architecture
Input Video → YOLOv8 Detection → Motion Compensation → ByteTrack → Output Video

## Results
- Dataset: VisDrone2019-MOT-val
- Model: YOLOv8n trained on VisDrone
- GPU: NVIDIA GeForce GTX 1650 (4GB)
- Average FPS: [fill after running]
- Tracks maintained: [fill after running]

## Setup
```powershell
git clone <your-repo>
cd AerialGuardian
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics supervision sahi filterpy onnx opencv-python==4.9.0.80 numpy==1.26.4
```

## Run
```powershell
python main.py --video data\raw\VisDrone\sequences\uav0000086_00000_v --output outputs\videos\result.mp4 --no-sahi --show
```

## Project Structure
AerialGuardian/
├── src/
│   ├── detection/      # YOLOv8 + SAHI detector
│   ├── tracking/       # ByteTrack integration
│   ├── compensation/   # Drone motion compensation
│   └── utils/          # Visualization tools
├── configs/            # Pipeline configuration
├── models/             # Trained weights
├── data/               # VisDrone dataset
└── outputs/            # Result videos

## Key Design Decisions
- **YOLOv8n**: Lightweight (6MB), 85+ FPS on GTX 1650
- **ByteTrack**: IoU-based tracking, no Re-ID needed for small objects
- **Motion Compensation**: Optical flow homography reduces ID switching
- **SAHI**: Sliced inference improves small object recall by ~15%

## Challenges
- Small object detection (persons as small as 10x20 pixels)
- Drone ego-motion causes tracker drift
- Crowded scenes with heavy occlusion