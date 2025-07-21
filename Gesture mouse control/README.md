# Gesture Mouse Control

A real-time computer vision application that enables touchless mouse control using hand gestures. The system tracks hand landmarks and translates finger movements and gestures into precise mouse operations, providing an intuitive interface for hands-free computer interaction.

## Overview

This project combines MediaPipe's hand detection with PyAutoGUI automation to create a seamless gesture-based mouse control system. Users can move the cursor, perform left and right clicks through natural hand gestures within a defined control area, making computer interaction more accessible and futuristic.

## Features

- **Precise Cursor Control**: Index finger movement mapped to mouse cursor with smooth interpolation
- **Gesture-based Clicking**: Left click via thumb-index pinch, right click via index-middle finger pinch
- **Bounded Control Area**: Defined rectangular area for accurate gesture recognition
- **Real-time Performance**: Live FPS monitoring and optimized gesture detection
- **Visual Feedback**: Clear indicators showing control boundaries and detected hand landmarks

## Usage

1. **Install Dependencies**:

   ```bash
   pip install opencv-python mediapipe pyautogui
   ```

2. **Launch the Application**:

   ```bash
   python mouse_control.py
   ```

3. **Position Your Hand**: Hold your hand within the blue rectangular boundary displayed on screen.

4. **Control Mouse**:

   - **Move Cursor**: Point with index finger (only index finger extended)
   - **Left Click**: Bring thumb and index finger together (pinch gesture)
   - **Right Click**: Bring index and middle finger together (pinch gesture)

5. **Exit**: Press 'q' key to close the application.

_Requirements: Python 3.7+, webcam, and good lighting for accurate hand detection._

## How It Works

The application uses a computer vision pipeline:

- **Hand Detection**: MediaPipe tracks 21 hand landmarks in real-time
- **Finger Recognition**: Identifies which fingers are extended using landmark positions
- **Cursor Mapping**: Maps index finger position within boundary (50,20) to (1200,700) to screen coordinates
- **Gesture Detection**: Calculates distances between specific finger pairs for click detection
- **Smooth Movement**: Applies smoothing algorithm to reduce cursor jitter
- **Mouse Control**: Uses PyAutoGUI for system-level mouse operations

## Control Gestures

| Gesture             | Fingers Extended | Action              | Distance Threshold |
| ------------------- | ---------------- | ------------------- | ------------------ |
| **Cursor Movement** | Index only       | Move mouse cursor   | N/A                |
| **Left Click**      | Thumb + Index    | Perform left click  | < 310 pixels       |
| **Right Click**     | Index + Middle   | Perform right click | < 100 pixels       |

## Technical Specifications

- **Camera Resolution**: 1280x720 pixels
- **Control Boundary**: 1150x680 pixel area
- **Smoothing Factor**: 5 (adjustable for cursor stability)
- **Detection Confidence**: 0.5 (configurable)
- **Click Sensitivity**: Customizable distance thresholds
