import subprocess
import cv2
import numpy as np

# Connect to Windows webcam stream (on Jetson Nano later, change to: cv2.VideoCapture(0))
win_ip = (
    subprocess.check_output(["ip", "route", "show", "default"])
    .decode()
    .split()[2]
)
cap = cv2.VideoCapture(f"http://{win_ip}:8080/video")

TEAM = "BLUE"  # Change to "RED" when playing as Red team

# Size thresholds (in pixels)
MIN_MAIN_AREA = 2500  # Minimum size to confirm a real block face
MIN_SLIVER_AREA = (
    400  # Allows a thin sliver of the 2nd color when viewed at an angle
)


def get_blob(mask, min_area):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    if area < min_area:
        return None

    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    bbox = cv2.boundingRect(biggest)  # (x, y, w, h)
    return {"center": (cx, cy), "area": area, "bbox": bbox}


def are_touching(box1, box2, max_gap=40):
    # Checks if two bounding boxes touch or sit within max_gap pixels of each other
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    return gap_x <= max_gap and gap_y <= max_gap


last_action = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filter Red and Blue pixels
    red_mask = cv2.inRange(
        hsv, np.array([0, 110, 70]), np.array([10, 255, 255])
    ) | cv2.inRange(hsv, np.array([170, 110, 70]), np.array([180, 255, 255]))
    blue_mask = cv2.inRange(
        hsv, np.array([95, 110, 70]), np.array([130, 255, 255])
    )

    # Detect blobs down to sliver size first
    red = get_blob(red_mask, MIN_SLIVER_AREA)
    blue = get_blob(blue_mask, MIN_SLIVER_AREA)

    red_is_main = red is not None and red["area"] >= MIN_MAIN_AREA
    blue_is_main = blue is not None and blue["area"] >= MIN_MAIN_AREA

    action = "no box detected"

    # CASE A: Both colors visible & touching (at least one is main size, other can be a sliver)
    if (
        (red_is_main or blue_is_main)
        and (red and blue)
        and are_touching(red["bbox"], blue["bbox"])
    ):
        rx, ry = red["center"]
        bx, by = blue["center"]

        # Draw yellow dividing line halfway between Red and Blue centers
        mid_x, mid_y = (rx + bx) // 2, (ry + by) // 2
        vx, vy = bx - rx, by - ry
        px, py = -vy, vx
        cv2.line(
            frame,
            (int(mid_x - px * 0.6), int(mid_y - py * 0.6)),
            (int(mid_x + px * 0.6), int(mid_y + py * 0.6)),
            (0, 255, 255),
            3,
        )
        cv2.circle(frame, (rx, ry), 6, (0, 0, 255), -1)
        cv2.circle(frame, (bx, by), 6, (255, 0, 0), -1)

        # Check where OUR team's color sits relative to the dividing line
        my_x, my_y = (bx, by) if TEAM == "BLUE" else (rx, ry)
        dx = my_x - mid_x
        dy = my_y - mid_y

        if abs(dx) > abs(dy):
            action = (
                "turn 90 degree clockwise"
                if dx < 0
                else "turn 90 degree anti-clockwise"
            )
        else:
            action = (
                "box is upside down, turn 180 degree" if dy > 0 else "keep"
            )

    # CASE B: Only ONE solid main color is visible
    elif red_is_main and not blue_is_main:
        rx, ry, rw, rh = red["bbox"]
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
        action = "move front to up" if TEAM == "RED" else "move front to down"

    elif blue_is_main and not red_is_main:
        bx, by, bw, bh = blue["bbox"]
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        action = "move front to up" if TEAM == "BLUE" else "move front to down"

    # Print to terminal only when the state changes
    if action != last_action:
        print(action)
        last_action = action

    cv2.putText(
        frame,
        f"TEAM {TEAM}: {action}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Sky Block Orientation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()