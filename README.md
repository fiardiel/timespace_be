# Timespace Backend
## Overview
This is the backend where it runs the API for authentication and the machine learning pipeline
for the FaceTrack feature.

## Structure
Just the important stuff that needs to be known.
```
timespace_be/
├── authentication/ # Authentication logic (users, sessions, JWT, permissions)
├── facetrack/ # Facial recognition & ML pipeline integration
├── timespace_be/ # Core Django project (settings, urls, wsgi, asgi)
└── requirements.txt # Python dependencies
```
