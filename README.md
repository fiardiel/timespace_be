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

## Create New Admin User
To create an admin user, we must do the following steps:

1. Connect to the VM instance 
2. Navigate to the project directory
```
cd /path/to/timespace_be/
```
3. Run the command:
```
python manage.py createsuperuser
```

4. Follow the prompts to set up the username, email, and password for the new admin user.