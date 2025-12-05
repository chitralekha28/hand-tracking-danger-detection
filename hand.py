import cv2
import numpy as np
import time
import math

def get_fingertip_point(contour):
    """
    Given a hand contour, find an approximate fingertip:
    - Compute centroid of contour
    - Find convex hull
    - Return hull point farthest from centroid
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None  # no centroid

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = (cx, cy)

    hull = cv2.convexHull(contour, returnPoints=True)
    if hull is None or len(hull) == 0:
        return centroid, None

    max_dist = 0
    fingertip = None
    for pt in hull:
        x, y = pt[0]
        dist = (x - cx) ** 2 + (y - cy) ** 2
        if dist > max_dist:
            max_dist = dist
            fingertip = (int(x), int(y))

    return centroid, fingertip

def distance_point_to_rect(point, rect):
    """
    Compute shortest Euclidean distance from a point to a rectangle.
    rect = (x1, y1, x2, y2)
    """
    if point is None:
        return None

    px, py = point
    x1, y1, x2, y2 = rect

    # dx is distance in x to the rectangle (0 if inside horizontally)
    if px < x1:
        dx = x1 - px
    elif px > x2:
        dx = px - x2
    else:
        dx = 0

    # dy is distance in y to the rectangle (0 if inside vertically)
    if py < y1:
        dy = y1 - py
    elif py > y2:
        dy = py - y2
    else:
        dy = 0

    return math.sqrt(dx * dx + dy * dy)

def classify_state(distance, safe_thresh=150, danger_thresh=70):
    """
    Classify state based on distance:
    - SAFE: distance > safe_thresh
    - WARNING: danger_thresh < distance <= safe_thresh
    - DANGER: distance <= danger_thresh
    """
    if distance is None:
        return "NO_HAND"
    if distance > safe_thresh:
        return "SAFE"
    elif distance > danger_thresh:
        return "WARNING"
    else:
        return "DANGER"

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame (like a selfie view)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Define a virtual rectangle (danger zone) on the right side
        rect_width = int(w * 0.2)
        rect_height = int(h * 0.4)
        rect_x1 = int(w * 0.65)
        rect_y1 = int(h * 0.3)
        rect_x2 = rect_x1 + rect_width
        rect_y2 = rect_y1 + rect_height
        virtual_rect = (rect_x1, rect_y1, rect_x2, rect_y2)

        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin color range (this is approximate and may need tuning)
        # You can adjust these values depending on your lighting & skin tone.
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Clean up the mask using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hand_centroid = None
        fingertip = None

        if contours:
            # Take the largest contour as the hand
            max_contour = max(contours, key=cv2.contourArea)

            # Ignore very small contours (noise)
            if cv2.contourArea(max_contour) > 2000:
                # Draw contour for visualization
                cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

                hand_centroid, fingertip = get_fingertip_point(max_contour)

                if hand_centroid is not None:
                    cv2.circle(frame, hand_centroid, 5, (255, 0, 0), -1)
                    cv2.putText(frame, "Hand Center", (hand_centroid[0] + 5, hand_centroid[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                if fingertip is not None:
                    cv2.circle(frame, fingertip, 8, (0, 0, 255), -1)
                    cv2.putText(frame, "Fingertip", (fingertip[0] + 5, fingertip[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Compute distance from fingertip to virtual rectangle
        dist = distance_point_to_rect(fingertip, virtual_rect)

        # Classify state
        state = classify_state(dist)

        # Draw the virtual rectangle with color based on state
        if state == "SAFE":
            color = (0, 255, 0)      # green
        elif state == "WARNING":
            color = (0, 255, 255)    # yellow
        elif state == "DANGER":
            color = (0, 0, 255)      # red
        else:
            color = (255, 255, 255)  # white / no hand

        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, 2)
        cv2.putText(frame, "Virtual Object", (rect_x1, rect_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show distance info
        if dist is not None:
            cv2.putText(frame, f"Distance: {dist:.1f}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show state label at top
        state_text = f"STATE: {state}"
        if state == "SAFE":
            state_color = (0, 255, 0)
        elif state == "WARNING":
            state_color = (0, 255, 255)
        elif state == "DANGER":
            state_color = (0, 0, 255)
        else:
            state_color = (255, 255, 255)

        cv2.putText(frame, state_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 2)

        # If DANGER, show big overlay text: "DANGER DANGER"
        if state == "DANGER":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, int(h * 0.4)), (w, int(h * 0.6)), (0, 0, 255), -1)
            alpha = 0.4
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.putText(frame, "DANGER DANGER", (int(w * 0.15), int(h * 0.55)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frames
        cv2.imshow("Hand Tracking - DANGER Demo", frame)
        cv2.imshow("Skin Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            # ESC or 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
