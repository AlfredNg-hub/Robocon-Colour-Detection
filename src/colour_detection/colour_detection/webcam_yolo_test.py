import subprocess
import urllib.request
import cv2
from ultralytics import YOLO


def find_stream_url(port=8080):
    # In WSL2, Windows sits at the default gateway IP (or 127.0.0.1 in mirrored mode)
    candidates = ["127.0.0.1"]
    try:
        route = subprocess.check_output(
            ["ip", "route", "show", "default"]
        ).decode()
        candidates.insert(0, route.split()[2])
    except Exception:
        pass

    for ip in candidates:
        url = f"http://{ip}:{port}/video"
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return url
        except Exception:
            continue
    return None


def main():
    stream_url = find_stream_url(8080)
    if not stream_url:
        print(
            "Error: Could not reach the Windows camera stream on port 8080."
        )
        print(
            "Make sure win_cam_stream.py is running on Windows and allowed"
            " in Windows Firewall."
        )
        return

    print(f"Connected to Windows webcam stream: {stream_url}")

    # Load the lightweight YOLO11 nano model (auto-downloads ~6 MB on first run)
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: OpenCV could not open the video stream.")
        return

    print("YOLO active! Press 'q' in the video window to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Lost connection to video stream.")
            break

        # Run YOLO object detection on the frame
        results = model(frame, verbose=False)

        # Draw bounding boxes and labels onto the frame
        annotated_frame = results[0].plot()

        # Show the live output window
        cv2.imshow("WSL2 Webcam YOLO Test", annotated_frame)

        # Press 'q' to exit cleanly
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()