# ImConvo
lip-reading AI developed by convobeo team 

👄 LipNet Real-Time Inference (WSL2 + IP Webcam)
This script allows you to run real-time lip-reading inference using a trained LipNet model. Since WSL2 has limited access to hardware webcams, this setup uses an IP Webcam (via your smartphone) to stream video over the local network.

1. Prerequisites
Smartphone: Android or iOS.

IP Webcam App: * Android: IP Webcam by Pavel Khlebovich

[iOS: iVCam or similar]

Environment: Python 3.10+, TensorFlow 2.x, OpenCV.

2. Smartphone Configuration (Crucial)
To ensure the model receives data in the format it was trained on, configure your IP Webcam app as follows:

Video Resolution: 352 x 288 (Lower resolution reduces latency).

Frame Rate (FPS): 25 FPS (LipNet is strictly temporal; mismatched FPS will cause poor results).

Start Server: Note the IP address provided (e.g., http://192.168.0.69:8080).

3. Usage
Run the script from your terminal using the --ip argument:

Bash
python inference.py --ip http://192.168.0.69:8080
Exit: Press q while the video window is focused.

Positioning: Ensure your mouth is centered in the crop area. The prediction text will appear at the top-left of the window.