# Django Core Imports
from django.shortcuts import render, redirect, get_object_or_404  # Rendering templates, redirecting, and fetching objects
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse, HttpResponseRedirect  # Handling HTTP responses
from django.contrib import messages  # Displaying success/error messages
from django.contrib.auth.decorators import login_required, user_passes_test  # Restricting views to logged-in users
from django.contrib.auth.models import User  # Accessing Django's built-in User model
from django.contrib.auth.hashers import make_password  # Hashing passwords securely
from django.contrib.auth import authenticate, login as auth_login  # Handling user authentication
from django.urls import reverse  # Generating dynamic URLs
from django.views.decorators.csrf import csrf_exempt  # Disabling CSRF protection for certain views (Use cautiously)
from django.utils.timezone import now  # Getting timezone-aware current time
from django.core.files.base import ContentFile  # Handling in-memory file storage
import cv2
import io
from PIL import Image
from django.conf import settings


# Models
from .models import Student, Exam, CheatingEvent, CheatingImage, CheatingAudio, Certificate  # Importing custom models

# External Library Imports
import os  # Operating system utilities (e.g., file handling)
import json  # JSON handling (e.g., parsing request data)
import threading  # Running concurrent tasks (e.g., real-time monitoring)
import base64  # Encoding and decoding base64 (used for image handling)
import numpy as np  # Numerical operations, especially for image processing
import cv2  # OpenCV for computer vision tasks (e.g., face recognition)
import logging  # Logging errors and system activity
import time  # Time-based operations (e.g., timestamps)
from PIL import Image, ImageDraw, ImageFont  # Image processing using the Pillow library
import io  # Handling in-memory file operations
import qrcode  # QR code generation for certificate authenticity

# Machine Learning Imports (Custom AI Models for Proctoring)
from .ml_models.object_detection import detectObject  # Detecting objects in the exam environment
from .ml_models.audio_detection import audio_detection  # Detecting external sounds for cheating detection
from .ml_models.gaze_tracking import gaze_tracking # Tracking eye gaze to detect focus and distractions 

# from .ml_models.gaze_tracking import gaze_tracking  # Tracking eye gaze to detect focus and distractions

# Fix: Import face_recognition (Previously missing)
import face_recognition  # Used for facial recognition, comparing student faces with stored images

# Fix: Proper datetime handling for Nepal Time Zone (Asia/Kathmandu)
import pytz  # For timezone handling
from datetime import datetime  # Standard date and time handling
def registration(request):
    return render(request, "registration.html")


# Define Nepal Time Zone
NEPAL_TZ = pytz.timezone('Asia/Kathmandu')

# Function to get Nepal's current time
def get_nepal_time():
    """
    Returns the current time in Nepal's timezone.
    This ensures all timestamps are consistent with the local time.
    """
    return datetime.now(NEPAL_TZ)


# Home page view
def home(request):
    """
    Renders the home page of the application.
    This is the entry point for users visiting the site.
    """
    return render(request, 'home.html')  # Render the home page


# Registration View
def registration(request):
    """
    Handles user registration, including:
    - Capturing form data (name, address, email, password, and photo)
    - Decoding and processing a base64-encoded image
    - Extracting face encoding using face recognition
    - Creating a new User and Student instance
    - Handling errors and displaying messages
    """
    if request.method == 'POST':  # Check if form is submitted
        # Retrieve form data
        name = request.POST['name']
        address = request.POST['address']
        email = request.POST['email']
        password = request.POST['password']
        captured_photo = request.POST.get('photo_data')  # Base64 image data

        try:
            # Decode the base64 image (photo_data comes in "data:image/png;base64,ENCODED_DATA")
            img_data = base64.b64decode(captured_photo.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)  # Convert to numpy array
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert to OpenCV image

            # Extract face encoding from the image
            face_encoding = get_face_encoding(image)  # Function should return a list or None
            if face_encoding is None:  # No face detected
                messages.error(request, "No face detected. Please try again.")
                return redirect('registration')
        except Exception as e:
            messages.error(request, f"Error processing image: {e}")
            return redirect('registration')

        # Check if the email is already registered
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return redirect('registration')

        try:
            # Create a new User instance
            user = User.objects.create(
                username=email,  # Use email as username for uniqueness
                email=email,
                first_name=name.split(' ')[0],  # Extract first name
                last_name=' '.join(name.split(' ')[1:]) if ' ' in name else '',  # Extract last name if available
                password=make_password(password),  # Hash password for security
            )

            # Create a linked Student instance
            student = Student(
                user=user,
                name=name,
                address=address,
                email=email,
                photo=ContentFile(img_data, name=f"{name}_photo.jpg"),  # Save the uploaded image
                face_encoding=face_encoding.tolist(),  # Convert NumPy array to list
            )
            student.save()

            # Store user session data
            request.session['user_id'] = user.id
            request.session['user_name'] = user.first_name

            messages.success(request, "Registration successful!")
            return redirect('login')  # Redirect to login page
        except Exception as e:
            messages.error(request, f"Error creating user: {e}")
            return redirect('registration')

    return render(request, 'registration.html')  # Render the registration page


# Helper function to extract face encoding
def get_face_encoding(image):
    """
    Extracts face encoding from an image using the face_recognition library.
    - Detects faces in the image.
    - Returns the encoding of the first face found.
    - Returns None if no faces are detected.
    """
    face_locations = face_recognition.face_locations(image)  # Detect faces in the image
    if not face_locations:
        return None  # Return None if no faces are detected
    return face_recognition.face_encodings(image, face_locations)[0]  # Return the first face encoding

# Helper function to match face encodings
def match_face_encodings(captured_encoding, stored_encoding):
    return face_recognition.compare_faces([stored_encoding], captured_encoding)[0]  # Compare encodings


#Login View
@csrf_exempt  # Allow POST requests without CSRF token (for simplicity, use proper CSRF handling in production)
def login(request):
    """
    Handles user login with email, password, and facial recognition.
    - Authenticates the user using email and password.
    - Compares the captured photo with the stored face encoding.
    - Logs the user in if all checks pass.
    - Returns JSON responses for success or failure.
    """
    if request.method == "POST":
        # Retrieve form data
        email = request.POST.get('email')
        password = request.POST.get('password')
        captured_photo_data = request.POST.get('captured_photo')

        # Validate required fields
        if not email or not password or not captured_photo_data:
            return JsonResponse({"success": False, "error": "Missing email, password, or captured photo."})

        try:
            # Decode the base64 image (remove the "data:image/png;base64," prefix)
            captured_photo_data = captured_photo_data.split(',')[1]
            captured_photo = base64.b64decode(captured_photo_data)

            # Convert the image to a NumPy array and decode it using OpenCV
            nparr = np.frombuffer(captured_photo, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Extract face encoding from the captured image
            captured_encoding = get_face_encoding(image)
            if captured_encoding is None:
                return JsonResponse({"success": False, "error": "No face detected in the captured photo."})

            # Authenticate the user using email and password
            user = authenticate(request, username=email, password=password)
            if user is None:
                return JsonResponse({"success": False, "error": "Invalid email or password."})

            try:
                # Fetch the associated student record
                student = user.student
                stored_encoding = np.array(student.face_encoding)

                # Compare the captured face encoding with the stored encoding
                if match_face_encodings(captured_encoding, stored_encoding):
                    # Log the user in
                    auth_login(request, user)

                    # Store student data in the session for future use
                    request.session['student_id'] = student.id
                    request.session['student_name'] = student.name

                    # Return a success response with redirect URL and student name
                    return JsonResponse({
                        "success": True,
                        "redirect_url": "/dashboard/",
                        "student_name": student.name
                    })
                else:
                    return JsonResponse({"success": False, "error": "Face does not match our records."})

            except Student.DoesNotExist:
                return JsonResponse({"success": False, "error": "No student record associated with this account."})

        except Exception as e:
            # Handle any unexpected errors during the login process
            return JsonResponse({"success": False, "error": f"Error processing image: {str(e)}"})

    # Render the login page for GET requests
    return render(request, "login.html")

# Logout View 
def logout_view(request):
    """
    Handles user logout.
    - Clears all session data.
    - Displays a success message.
    - Redirects the user to the home page.
    """
    request.session.flush()  # Clear all session data
    messages.success(request, "You have been logged out.")  # Display a success message
    return redirect('home')  # Redirect to the home page

# Video feed generation for the webcam
# Video feed generation for the webcam + cheating detection
def gen_frames(request):
    """
    Single webcam stream:
    - Reads frames from camera
    - Runs cheating detection on each frame
    - Streams frames to browser
    """
    global warning

    camera = cv2.VideoCapture(0)  # open default webcam
    if not camera.isOpened():
        logger.error("Could not open webcam in gen_frames().")
        return

    while not stop_event.is_set():
        success, frame = camera.read()
        if not success:
            logger.error("Failed to read frame from webcam.")
            break

        # Run cheating detection on this frame
        try:
            process_frame(frame, request)
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.error("Failed to encode frame.")
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

    camera.release()
    logger.info("Webcam released in gen_frames()")

#Dashboard View
@login_required
def dashboard(request):
    """
    Renders the dashboard page for authenticated users.
    - Retrieves the user's name from the session.
    - Displays personalized content on the dashboard.
    - Handles cases where the user is not logged in (defaults to 'Guest').
    """
    # Retrieve the user's name from the session (default to 'Guest' if not found)
    user_name = request.session.get('user_name', 'Guest')

    # Prepare context data to pass to the template
    context = {
        'user_name': user_name,  # Pass the user's name to the template
    }

    # Render the dashboard template with the context data
    return render(request, 'dashboard.html', context)



# -------------------------Video Detection Thread----------------------------------
from django.utils import timezone
import pytz

# Define Nepal Time Zone
NEPAL_TZ = pytz.timezone('Asia/Kathmandu')

# Helper function to get Nepal time
def get_nepal_time():
    return timezone.now().astimezone(NEPAL_TZ)

def get_nepal_time_str():
    return get_nepal_time().strftime('%Y-%m-%d %I:%M:%S %p %Z')


logger = logging.getLogger(__name__)

# Global variables for warnings and background processes
warning = None
last_audio_detected_time = time.time()
stop_event = threading.Event()  # To stop background threads

# Function to process each frame
def process_frame(frame, request):
    """Process a single frame for cheating detection."""
    global warning
    labels, processed_frame, person_count, detected_objects = detectObject(frame)
    cheating_event = None

    # Extract object names
    detected_labels = [label for label, _ in labels]
    # Check for cheating conditions
    if any(label in ["cell phone", "book"] for label in detected_labels):
        warning = f"ALERT: {', '.join(detected_labels)} detected!"  # Corrected formatting
        cheating_event, _ = CheatingEvent.objects.get_or_create(
            student=request.user.student,
            cheating_flag=True,
            event_type="object_detected"
        )
        save_cheating_event(frame, request, cheating_event, detected_objects)

    if person_count > 1:
        warning = "ALERT: Multiple persons detected!"
        cheating_event, _ = CheatingEvent.objects.get_or_create(
            student=request.user.student,
            cheating_flag=True,
            event_type="multiple_persons"
        )
        save_cheating_event(frame, request, cheating_event, detected_objects)

    gaze = gaze_tracking(frame)
    if gaze["gaze"] != "center":
        warning = "ALERT: Candidate not looking at the screen!"
        cheating_event, _ = CheatingEvent.objects.get_or_create(
            student=request.user.student,
            cheating_flag=True,
            event_type="gaze_detected"
        )
        save_cheating_event(frame, request, cheating_event, detected_objects)

def video_feed(request):
    """
    Streams the live video feed (with cheating detection) to the browser.
    """
    return StreamingHttpResponse(
        gen_frames(request),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


# Accept frames streamed from the browser camera
@login_required
def upload_frame(request):
    """
    Receives a base64-encoded frame from the browser, decodes it,
    runs cheating detection, and returns the latest warning (if any).
    """
    global warning

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method."}, status=405)

    if stop_event.is_set():
        return JsonResponse({"status": "stopped"})

    frame_data = request.POST.get("frame")
    if not frame_data:
        return JsonResponse({"error": "No frame data provided."}, status=400)

    try:
        # Strip data URL prefix if present and decode
        _, encoded = frame_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_frame = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Frame decode returned None")

        process_frame(frame, request)
    except Exception as exc:  # Broad catch to keep the stream alive
        logger.error(f"Failed to process uploaded frame: {exc}")
        return JsonResponse({"error": "Failed to process frame."}, status=400)

    return JsonResponse({"status": "ok", "warning": warning})


# Function to process audio
def process_audio(request):
    """Continuously process audio for cheating detection."""
    global last_audio_detected_time, warning

    while not stop_event.is_set():  # Check if stop_event is triggered
        audio = audio_detection()
        if audio["audio_detected"]:
            warning = "ALERT: Suspicious audio detected!"
            cheating_event, _ = CheatingEvent.objects.get_or_create(
                student=request.user.student,
                cheating_flag=True,
                event_type="audio_detected"
            )
            save_cheating_event(None, request, cheating_event, audio_data=audio["audio_data"])
            last_audio_detected_time = time.time()

        if time.time() - last_audio_detected_time > 5:
            warning = None

        time.sleep(2)  # Avoid excessive CPU usage

    print("Audio processing stopped.")  # Debugging to confirm the thread exits



# Background processing for video
def background_processing(request):
    """Runs video processing in the background."""
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 2 == 0:
            process_frame(frame, request)
        
        frame_count += 1
        time.sleep(0.5)
    
    cap.release()


# Helper function to create a WAV file from raw audio bytes
import io
import wave

def create_wav_bytes(raw_audio, channels=1, sampwidth=2, framerate=48000):
    """
    Wrap raw PCM audio bytes with a WAV header.
    
    :param raw_audio: The raw audio bytes (concatenated frames)
    :param channels: Number of audio channels (1 for mono)
    :param sampwidth: Sample width in bytes (2 for 16-bit audio)
    :param framerate: Frame rate (sample rate)
    :return: Audio data in WAV format as bytes
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(raw_audio)
    return wav_buffer.getvalue()

## Function to save cheating event
def save_cheating_event(frame, request, cheating_event, detected_objects=None, audio_data=None):
    """Save cheating event along with images and audio in the database."""
    try:
        
        # Save detected objects
        if detected_objects:
            cheating_event.detected_objects = detected_objects  # Save as JSON
            cheating_event.save()
        # Save up to 10 sample images per event
        if frame is not None and cheating_event.cheating_images.count() < 10:
            try:
                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_io = io.BytesIO()
                image_pil.save(image_io, format="JPEG", quality=85)
                image_content = image_io.getvalue()
                
                cheating_image = CheatingImage(event=cheating_event)
                cheating_image.image.save(
                    f"cheating_{time.time()}.jpg", 
                    ContentFile(image_content), 
                    save=True
                )
            except Exception as e:
                logger.error(f"Error processing image: {e}")
        
        # Save audio data
        if audio_data:
            try:
                # Convert raw audio bytes to a proper WAV file bytes.
                wav_data = create_wav_bytes(audio_data, channels=1, sampwidth=2, framerate=48000)
                cheating_audio = CheatingAudio(event=cheating_event)
                cheating_audio.audio.save(
                    f"cheating_audio_{time.time()}.wav", 
                    ContentFile(wav_data), 
                    save=True
                )
            except Exception as e:
                logger.error(f"Error processing audio: {e}")

        logger.info(f"Cheating event saved for student {request.user.student.id}")
    
    except Exception as e:
        logger.error(f"Error saving cheating event: {e}")

## Exam Page View
## Exam Page View
@login_required
def exam(request):
    """
    Renders the exam page and starts proctoring:
    - Loads questions from ai.json
    - Resets stop_event
    - Starts background audio detection thread
    - Video + visual cheating detection run inside video_feed/gen_frames
    """
    # Ensure student exists
    student = request.user.student

    # Path to questions JSON
    json_path = os.path.join(
        settings.BASE_DIR,        # futurproctor/
        "proctoring",             # proctoring/
        "dummy_data",             # dummy_data/
        "ai.json"                 # ai.json
    )

    print("DEBUG JSON PATH:", json_path)

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            questions = data.get("questions", [])

    except FileNotFoundError:
        return HttpResponse(f"Error: Questions file not found at {json_path}", status=404)

    except json.JSONDecodeError:
        return HttpResponse("Error: Invalid JSON format", status=500)

    # ✅ Start proctoring: reset stop_event and launch audio thread
    global stop_event
    stop_event.clear()
    threading.Thread(target=process_audio, args=(request,), daemon=True).start()
    # Note: video + process_frame run via /video_feed/, not a separate thread

    return render(request, "exam.html", {
        "questions": questions,
        "warning": warning,
    })

# Submit exam


@login_required
def submit_exam(request):
    if request.method == 'POST':
        # Stop proctoring threads
        global stop_event
        stop_event.set()
        user = request.user

        # ✅ Use BASE_DIR instead of hard-coded Windows path
        json_path = os.path.join(
            settings.BASE_DIR,
            "proctoring",
            "dummy_data",
            "ai.json",
        )

        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            return HttpResponse(f"Error: Questions file not found at {json_path}!", status=404)
        except json.JSONDecodeError:
            return HttpResponse("Error: Failed to parse the questions file!", status=400)

        questions = data.get('questions', [])
        total_questions = len(questions)
        correct_answers = 0

        # Check answers
        for question in questions:
            qid = question['id']
            user_answer = request.POST.get(f'answer_{qid}')
            if user_answer == question['correct_answer']:
                correct_answers += 1

        # Save exam result
        exam = Exam(
            student=user.student,
            total_questions=total_questions,
            correct_answers=correct_answers,
            timestamp=timezone.now()
        )
        exam.save()

        messages.success(request, 'You have successfully completed the exam!')
        return redirect('exam_submission_success')

    return HttpResponse("Invalid request method.", status=400)

@login_required
def record_tab_switch(request):
    if request.method == "POST":
        # Get the current student
        student = request.user.student
        logger.info(f"Student: {student}")

        # # Get the active exam for the student
        # active_exam = Exam.objects.filter(student=student, status='ongoing').first()
        # if not active_exam:
        #     logger.error("No active exam found for the student")
        #     return JsonResponse({"error": "No active exam found for the student"}, status=400)

        # logger.info(f"Active Exam: {active_exam}")

        # Get or create a CheatingEvent for the student and exam
        cheating_event, created = CheatingEvent.objects.get_or_create(
            student=student,
            # exam=active_exam,
            event_type='tab_switch',  # Specify the event type
            defaults={
                'cheating_flag': False,
                'tab_switch_count': 0,
            }
        )

        logger.info(f"Cheating Event: {cheating_event}, Created: {created}")

        # Increment the tab switch count
        cheating_event.tab_switch_count += 1
        logger.info(f"Updated Tab Switch Count: {cheating_event.tab_switch_count}")

        # Set cheating_flag based on tab_switch_count
        cheating_event.cheating_flag = cheating_event.tab_switch_count >= 1
        logger.info(f"Cheating Flag: {cheating_event.cheating_flag}")

        # Save the updated CheatingEvent
        cheating_event.save()
        logger.info("Cheating Event saved successfully")

        # If tab switches exceed 5, take action
        if cheating_event.tab_switch_count > 5:
            stop_event.set()  # Stop background threads (ensure stop_event is defined)
            logger.info("Tab switches exceeded 5, terminated from the exam")
            return JsonResponse({
                "status": "terminated",
                "message": "You have exceeded the allowed tab switches. Your exam is terminated."
            }, status=200)
        # Return a JSON response with the updated count and flag
        return JsonResponse({
            "status": "updated",
            "count": cheating_event.tab_switch_count,
            "cheating_flag": cheating_event.cheating_flag,
            "message": f"Tab switch detected! Total switches: {cheating_event.tab_switch_count}"
        }, status=200)

    return JsonResponse({"error": "Invalid request"}, status=400)


# Exam submission success page
def exam_submission_success(request):
    return render(request, 'exam_submission_success.html')

# Result page
@login_required
def result(request):
    user = request.user
    try:
        exam = Exam.objects.filter(student=user.student).latest('timestamp')
    except Exam.DoesNotExist:
        return HttpResponse("No exam found for this user", status=404)

    total_questions = exam.total_questions
    correct_answers = exam.correct_answers
    percentage = (correct_answers / total_questions) * 100

    context = {
        'user_name': user.username,
        'score': correct_answers,
        'total_questions': total_questions,
        'percentage': round(percentage, 2),
    }

    return render(request, 'result.html', context)



from django.http import JsonResponse

# Fetch warnings
@csrf_exempt
def get_warning(request):
    """Fetch real-time warnings for the exam page."""
    global warning
    return JsonResponse({'warning': warning})

# Streaming notifications to the proctor
def proctor_notifications(request):
    """Stream real-time cheating events to the proctor."""
    def event_stream():
        while True:
            events = CheatingEvent.objects.filter(cheating_flag=True).order_by('-timestamp')[:5]
            if events:
                yield f"data: {json.dumps([str(event) for event in events])}\n\n"
            time.sleep(5)
    
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


## Logout
def logout(request):
    return render(request,'home.html')

# ----------------------Admin Plus Report Page ---------------------------------------

# Admin views
from django.shortcuts import render, get_object_or_404
from django.db.models import Count, Sum
from django.contrib.admin.views.decorators import staff_member_required
from .models import Student, Exam, CheatingEvent, CheatingImage, CheatingAudio, Certificate


def _load_font(size, fallback=ImageFont.load_default()):
    """Try to load a nicer font if available, otherwise fallback."""
    try:
        # DejaVuSans is bundled with Pillow in most installs
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return fallback


def generate_certificate_image(student, exam, verification_url):
    """
    Build a more polished certificate image with a QR code for authenticity.
    """
    width, height = 1600, 1100
    bg_color = (255, 255, 255)
    primary = (40, 55, 90)          # Deep navy for headings
    accent = (191, 146, 38)         # Gold accent
    text_color = (45, 45, 45)

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Subtle border + corner accents
    border_thickness = 6
    draw.rectangle([(40, 40), (width - 40, height - 40)], outline=accent, width=border_thickness)
    draw.rectangle([(70, 70), (width - 70, height - 70)], outline=primary, width=2)

    # Decorative waves (simple lines)
    for offset in range(0, 140, 20):
        draw.arc([(80 - offset, 80 - offset), (200 + offset, 200 + offset)], start=20, end=340, fill=accent, width=2)
        draw.arc([(width - 200 - offset, height - 200 - offset), (width - 80 + offset, height - 80 + offset)],
                 start=200, end=520, fill=accent, width=2)

    # Fonts (name bold and large)
    title_font = _load_font(70)
    subtitle_font = _load_font(44)
    award_font = _load_font(30)
    name_font = _load_font(84)
    body_font = _load_font(28)
    small_font = _load_font(22)

    # Header text
    draw.text((width // 2, 170), "CERTIFICATE", fill=primary, anchor="ms", font=title_font)
    draw.text((width // 2, 240), "OF APPRECIATION", fill=accent, anchor="ms", font=subtitle_font)
    draw.text((width // 2, 320), "THE FOLLOWING AWARD IS GIVEN TO", fill=primary, anchor="ms", font=award_font)

    # Recipient
    draw.text((width // 2, 420), student.name, fill=primary, anchor="ms", font=name_font)

    # Summary line
    exam_name = exam.exam_name or "Exam"
    score = exam.percentage_score if exam.percentage_score is not None else 0
    summary = f"This certificate is given to {student.name} for completing the {exam_name} with a score of {score}%."
    draw.text((width // 2, 500), summary, fill=text_color, anchor="ms", font=body_font)

    # Metadata row
    draw.text((width // 2, 560), f"Issued on: {now().date().isoformat()}", fill=text_color, anchor="ms", font=small_font)
    draw.text((width // 2, 600), f"Student ID: {student.id}", fill=text_color, anchor="ms", font=small_font)

    # Signatures placeholders
    draw.line([(width * 0.22, 780), (width * 0.42, 780)], fill=accent, width=3)
    draw.text((width * 0.32, 810), "Head of Event", fill=text_color, anchor="ms", font=small_font)
    draw.line([(width * 0.58, 780), (width * 0.78, 780)], fill=accent, width=3)
    draw.text((width * 0.68, 810), "Mentor", fill=text_color, anchor="ms", font=small_font)

    # QR code for authenticity
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(verification_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color=primary, back_color="white").convert("RGB")
    qr_size = 220
    qr_img = qr_img.resize((qr_size, qr_size))
    img.paste(qr_img, (width - qr_size - 140, height - qr_size - 180))
    draw.text((width - qr_size - 140, height - qr_size - 200), "Scan to verify", fill=text_color, anchor="lb", font=small_font)

    return img


@staff_member_required(login_url='/admin/login/')
def admin_dashboard(request):
    # Fetch students with counts for exams and cheating events
    students = Student.objects.annotate(
        exam_count=Count('exams'),
        cheating_event_count=Count('cheating_events')
    ).prefetch_related('exams', 'cheating_events', 'certificates')
    
    # Calculate trust score and exam scores for each student
    for student in students:
        # Example: Trust score decreases 10 points per cheating event (with a floor of 0)
        student.trust_score = max(0, 100 - (student.cheating_event_count * 10))
        
        for exam in student.exams.all():
            if exam.total_questions and exam.total_questions > 0 and exam.percentage_score is None:
                exam.percentage_score = calculate_exam_score(exam)
                exam.save()

        # Attach latest certificate for easy access in the template
        student.latest_certificate = student.certificates.order_by('-issued_at').first()
    
    context = {
        'students': students,
    }
    return render(request, 'admin_dashboard.html', context)

## exam score
def calculate_exam_score(exam):
    """Calculate the exam score as a percentage."""
    if exam.total_questions and exam.total_questions > 0:
        return round((exam.correct_answers / exam.total_questions) * 100, 2)
    return 0.0


## Helper Function for aggregated detected objects
import json
def get_detected_objects_string(cheating_events):
    """Aggregate and convert the detected objects from all events into a list."""
    detected_objects_set = set()
    for event in cheating_events:
        # If detected_objects is not already a list, try converting it.
        objs = event.detected_objects
        if isinstance(objs, str):
            try:
                objs = json.loads(objs)
            except json.JSONDecodeError:
                objs = []
        # Now, objs should be a list so add each one to our set.
        if isinstance(objs, list):
            detected_objects_set.update(objs)
    return list(detected_objects_set)

### Report view
def report_page(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    exam = student.exams.first()  # Or however you want to choose the exam
    cheating_events = CheatingEvent.objects.filter(student=student)

    # Aggregate detected objects as a list
    detected_objects_list = get_detected_objects_string(cheating_events)
    detected_objects_str = ", ".join(detected_objects_list) if detected_objects_list else "No objects detected"

    # Sum up tab switch count from events
    total_tab_switch_count = cheating_events.aggregate(total=Sum('tab_switch_count'))['total'] or 0

    # Audio files: if you're using a FileField, Django automatically converts the stored file path
    # into a URL via the `.url` attribute once media is configured correctly.
    cheating_audios = CheatingAudio.objects.filter(event__student=student)
    audio_urls = [audio.audio.url for audio in cheating_audios if audio.audio]

    context = {
        'student': student,
        'exam': exam,
        'detected_objects': detected_objects_str,
        'total_tab_switch_count': total_tab_switch_count,
        # You can also add correct answer attempt and total questions:
        'correct_answers': exam.correct_answers,
        'total_questions': exam.total_questions,
        'cheating_status': any(
            event.event_type in ['object_detected', 'multiple_faces_detected', 'tab_switch']
            for event in cheating_events
        ),
        'cheating_images': [
            {
                'url': img.image.url,
                'event_type': img.event.event_type,
                'timestamp': img.timestamp
            }
            for img in CheatingImage.objects.filter(event__student=student)
        ],
        'audio_urls': audio_urls,
        'cheating_events': cheating_events,  # if you need to list them
    }
    return render(request, 'report_page.html', context)




from django.template.loader import get_template
from xhtml2pdf import pisa
# (Ensure you import any helper functions you might have, e.g., get_detected_objects_string)

def download_report(request, student_id):
    # Retrieve student and related data
    student = get_object_or_404(Student, id=student_id)
    exam = student.exams.first()  # Adjust as necessary
    cheating_events = CheatingEvent.objects.filter(student=student)
    
    # Process detected objects (assuming you have a helper function)
    detected_objects_list = get_detected_objects_string(cheating_events)
    detected_objects_str = ", ".join(detected_objects_list) if detected_objects_list else "No objects detected"

    # Sum up tab switch counts
    total_tab_switch_count = cheating_events.aggregate(total=Sum('tab_switch_count'))['total'] or 0

    # Audio URLs (xhtml2pdf might need absolute paths for images and other media,
    # but for simple cases it often works fine)
    cheating_audios = CheatingAudio.objects.filter(event__student=student)
    audio_urls = [audio.audio.url for audio in cheating_audios if audio.audio]

    # Prepare context for the template
    context = {
        'student': student,
        'exam': exam,
        'detected_objects': detected_objects_str,
        'total_tab_switch_count': total_tab_switch_count,
        'correct_answers': exam.correct_answers,
        'total_questions': exam.total_questions,
        'cheating_status': any(
            event.event_type in ['object_detected', 'multiple_faces_detected', 'tab_switch']
            for event in cheating_events
        ),
        'cheating_images': [
            {
                'url': img.image.url,
                'event_type': img.event.event_type,
                'timestamp': img.timestamp
            }
            for img in CheatingImage.objects.filter(event__student=student)
        ],
        'audio_urls': audio_urls,
        'cheating_events': cheating_events,
    }
    
    # Render the HTML template with context
    template = get_template('report_page.html')
    html = template.render(context)

    # Create a HttpResponse with PDF headers
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="report_{student.id}.pdf"'
    
    # Create PDF using xhtml2pdf (pisa)
    pisa_status = pisa.CreatePDF(html, dest=response)
    
    # Check for errors
    if pisa_status.err:
        return HttpResponse('We had some errors while generating the PDF', status=500)
    
    return response


def add_question(request):
    return render(request, 'add_question.html')  # Ensure you have this template


@staff_member_required(login_url='/admin/login/')
def issue_certificate(request, student_id):
    """Generate and store a certificate for the latest exam of a student."""
    student = get_object_or_404(Student, id=student_id)
    exam = student.exams.order_by('-timestamp').first()

    if not exam:
        messages.error(request, "No exam found for this student.")
        return redirect('admin_dashboard')

    if exam.percentage_score is None:
        exam.calculate_percentage()

    # Create certificate record first to generate a verification URL (file may be blank/null)
    certificate = Certificate.objects.create(student=student, exam=exam, issued_by=request.user)

    verification_url = request.build_absolute_uri(
        reverse('verify_certificate', args=[certificate.id])
    )

    # Build certificate image and save
    cert_image = generate_certificate_image(student, exam, verification_url)
    buffer = io.BytesIO()
    cert_image.save(buffer, format="PNG")
    image_content = buffer.getvalue()

    certificate.file.save(
        f"certificate_{student.id}_{int(time.time())}.png",
        ContentFile(image_content),
        save=True,
    )

    messages.success(request, f"Certificate issued for {student.name}.")
    return redirect('admin_dashboard')


@staff_member_required(login_url='/admin/login/')
def download_certificate(request, certificate_id):
    """Redirect to the certificate file for viewing/downloading."""
    certificate = get_object_or_404(Certificate, id=certificate_id)
    return HttpResponseRedirect(certificate.file.url)


def verify_certificate(request, certificate_id):
    """
    Simple verification page to confirm authenticity of a certificate.
    """
    certificate = get_object_or_404(Certificate.objects.select_related('student', 'exam', 'issued_by'), id=certificate_id)
    context = {
        "certificate": certificate,
        "student": certificate.student,
        "exam": certificate.exam,
    }
    return render(request, "verify_certificate.html", context)
