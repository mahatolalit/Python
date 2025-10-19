import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuration
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
MONKE_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Detection thresholds
FINGER_TO_LIPS_THRESHOLD = 0.08  # Distance threshold for index finger near lips (thinking pose)
HAND_RAISED_Y_THRESHOLD = 0.5    # Y coordinate threshold for raised hand (idea pose) - more lenient
FINGER_CURL_THRESHOLD = 0.08     # Threshold to detect if non-index fingers are curled - more lenient

# Load monke images
try:
    thinking_monke = cv2.imread("thinking.jpeg")
    idea_monke = cv2.imread("idea.jpeg")

    if thinking_monke is None:
        raise FileNotFoundError("thinking.jpeg not found")
    if idea_monke is None:
        raise FileNotFoundError("idea.jpeg not found")

    # Resize monke
    thinking_monke = cv2.resize(thinking_monke, MONKE_WINDOW_SIZE)
    idea_monke = cv2.resize(idea_monke, MONKE_WINDOW_SIZE)
    
except Exception as e:
    print("Error loading monke images!")
    print(f"Details: {e}")
    print("\nExpected files:")
    print("- thinking.jpeg (thinking/contemplating pose)")
    print("- idea.jpeg (idea/eureka moment)")
    exit()

blank_monke = np.zeros((MONKE_WINDOW_SIZE[0], MONKE_WINDOW_SIZE[1], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Monke Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Monke Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Monke Output', WINDOW_WIDTH + 150, 100)

print("Controls:")
print("  Press 'q' to quit")
print("  IDEA: Raise index finger up (other fingers closed) 💡")
print("  THINKING: Index finger near lips 🤔")
print("  Default: Neutral state")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "NEUTRAL"
        index_finger_up = False
        index_finger_near_lips = False

        # Process hand detection
        results_hands = hands.process(image_rgb)
        results_face = face_mesh.process(image_rgb)
        
        # Check hand gestures
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Get finger tip and base landmarks
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                
                # Check if index finger is extended (pointing up)
                index_extended = index_tip.y < index_mcp.y
                
                # Check if other fingers are curled (more lenient check)
                # Compare finger tips to their middle joints
                middle_curled = middle_tip.y > middle_mcp.y - FINGER_CURL_THRESHOLD
                ring_curled = ring_tip.y > ring_mcp.y - FINGER_CURL_THRESHOLD
                pinky_curled = pinky_tip.y > pinky_mcp.y - FINGER_CURL_THRESHOLD
                
                # Check if hand is raised (index finger pointing up - more lenient)
                hand_raised = index_tip.y < HAND_RAISED_Y_THRESHOLD
                
                # Count how many fingers are curled (at least 2 out of 3 should be curled for easier detection)
                curled_count = sum([middle_curled, ring_curled, pinky_curled])
                
                # IDEA pose: Index finger up, most other fingers curled, hand raised
                # Relaxed condition: index extended + hand raised + at least 2 fingers curled
                if index_extended and hand_raised and curled_count >= 2:
                    index_finger_up = True
                
                # THINKING pose: Index finger near lips
                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        # Get lips center
                        upper_lip = face_landmarks.landmark[13]
                        lower_lip = face_landmarks.landmark[14]
                        lips_center_x = (upper_lip.x + lower_lip.x) / 2
                        lips_center_y = (upper_lip.y + lower_lip.y) / 2
                        
                        # Distance from index finger tip to lips
                        distance_to_lips = ((index_tip.x - lips_center_x)**2 + (index_tip.y - lips_center_y)**2)**0.5
                        
                        if distance_to_lips < FINGER_TO_LIPS_THRESHOLD and index_extended:
                            index_finger_near_lips = True
        
        # Determine state based on detected gestures
        if index_finger_up:
            current_state = "IDEA"
        elif index_finger_near_lips:
            current_state = "THINKING"
        
        # Select monke based on state
        if current_state == "IDEA":
            monke_to_display = idea_monke
            monke_name = "�"
        elif current_state == "THINKING":
            monke_to_display = thinking_monke
            monke_name = "🤔"
        else:
            monke_to_display = blank_monke
            monke_name = "😐"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Monke Output', monke_to_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()