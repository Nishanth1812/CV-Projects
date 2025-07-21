# Hand Gesture Volume Control

A real-time computer vision application that enables hands-free volume control using finger distance detection. The system leverages MediaPipe hand tracking to provide an intuitive, touchless interface for audio control by mapping thumb-index finger distance to system volume levels.

## Overview

This project combines MediaPipe's hand landmark detection with Windows audio control systems, creating a seamless bridge between physical gestures and digital audio control. Users can naturally manage their computer's volume without physical controls, with real-time visual feedback showing how gestures translate to volume changes.

## Features

- **Advanced Hand Detection**: Google's MediaPipe framework with 21-point hand landmark tracking
- **Intuitive Gesture Control**: Volume adjustment through thumb-index finger distance mapping
- **Rich Visual Feedback**: Live volume bar, percentage indicator, and gesture visualization
- **Performance Monitoring**: Real-time FPS display for optimal responsiveness
- **Smart Detection**: Intelligent recognition distinguishing intentional gestures from casual movements

## Usage

1. **Install Dependencies**:

   ```bash
   pip install opencv-python mediapipe numpy pycaw comtypes
   ```

2. **Launch the Application**:

   ```bash
   python volume_control.py
   ```

3. **Position Your Hand**: Hold your hand in front of the camera with thumb and index finger visible.

4. **Control Volume**:

   - **Decrease**: Pinch thumb and index finger together
   - **Increase**: Spread thumb and index finger apart
   - **Visual Cues**: Green circles mark fingertips, connecting line shows distance

5. **Exit**: Press any key while the camera window is active.

_Requirements: Python 3.7+, Windows OS, webcam, and good lighting._

## How It Works

The application uses a computer vision pipeline:

- **Hand Detection**: MediaPipe tracks 21 hand landmarks in real-time
- **Gesture Recognition**: Calculates distance between thumb tip (landmark 4) and index finger tip (landmark 8)
- **Volume Mapping**: Maps finger distance (30-150 pixels) to Windows volume range (-65.25 to 0.0 dB)
- **Visual Feedback**: Renders finger positions, distance lines, volume bar, and percentage display
- **Audio Control**: Integrates with Windows Core Audio APIs through PyCaw for system volume adjustment
