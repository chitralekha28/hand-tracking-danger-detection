📌 Hand Tracking Danger Detection System (Classical Computer Vision, No MediaPipe)

This project is a real-time hand tracking and danger zone detection system built using OpenCV + NumPy only, without MediaPipe, OpenPose, or any pose-detection APIs.

It was developed as part of the Arvyax internship assignment.

🚀 Objective

Detect a user's hand through a webcam and trigger warnings when the hand approaches a virtual object on the screen.

The system must:

Track the hand / fingertip in real time

Detect distance between the hand and a virtual boundary

Display interaction states:

SAFE

WARNING

DANGER

Show a big "DANGER DANGER" alert when the hand enters the danger zone

Run at ≥ 8 FPS using CPU only

🎯 Features
✔ Real-time hand detection

Uses classical computer vision techniques:

HSV color segmentation

Morphological filtering

Contour detection

Convex hull

Centroid + farthest hull point (fingertip estimation)

✔ Virtual Object Detection

A virtual rectangle acts as a boundary.
The distance from the fingertip → rectangle determines the state.

✔ Distance-Based State Logic
State	Condition
SAFE	Hand is far from virtual object
WARNING	Hand approaching boundary
DANGER	Hand extremely close / touching
✔ Visual Overlay

Hand contours

Centroid and fingertip marker

Color-coded virtual boundary (Green/Yellow/Red)

Big DANGER DANGER alert when triggered

FPS counter

📌 Limitations

Skin detection may vary with lighting

Works best with a single hand

Background should not contain skin-colored objects near the camera

🔮 Future Improvements

Add adaptive skin detection using K-means

Gesture recognition

Multi-hand tracking

Replace virtual rectangle with complex shapes

👤 Author

Chitralekha
Developed for Arvyax Internship Assignment
December 2025
