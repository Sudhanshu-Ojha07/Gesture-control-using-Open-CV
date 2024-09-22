import cv2
import mediapipe as mp
import os
import time
import math  # Added for distance calculation
import subprocess
import webbrowser as wb
# Define folder path (without dot)
folder_path = '/mnt/large_partition/Open CV'  # Change this to your folder path
drawing_mode = False 
drawing_point = []
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)  # Allow detection of 2 hands
mp_drawing = mp.solutions.drawing_utils


def open_folder():
    
    if os.path.exists(folder_path):
        subprocess.run(['xdg-open',folder_path])
     
        

        
    else:
        print("Folder is already locked or doesn't exist.")

def close_folder():
    
    
    try:
        window_id = subprocess.check_output(
            ["xdotool", "search", "--onlyvisible", "--class", "nautilus"]
        ).strip()

        # Close the window with the found window ID
        if window_id:
            os.system(f"xdotool windowclose {window_id.decode('utf-8')}")
        else:
            print("No window found for the specified folder path.")
    except subprocess.CalledProcessError:
        print("Error: Could not find the file manager window.")



def calculate_distance(landmark1, landmark2):
    """Calculate the Euclidean distance between two landmarks."""
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def is_lock_sign(landmarks1, landmarks2):
    """Detect if palms are touching each other (lock gesture)."""
    # Calculate the distance between the center of both palms (WRIST landmarks)
    palm_distance = calculate_distance(landmarks1.landmark[mp_hands.HandLandmark.WRIST], 
                                       landmarks2.landmark[mp_hands.HandLandmark.WRIST])
    return palm_distance < 0.3  # Palms close enough to lock (you may adjust the threshold)

def is_unlock_sign(landmarks1, landmarks2):
    """Detect if index fingers are moving away from each other (unlock gesture)."""
    # Calculate the distance between the tips of both index fingers
    index_finger_distance = calculate_distance(landmarks1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], 
                                               landmarks2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    return index_finger_distance > 0.5  # Index fingers are far enough to unlock (you may adjust the threshold)
def chrome_browser():
    
    subprocess.run(['xdg-open','http://www.google.com'])


def recognize_shape(drawing_point):
    if len(drawing_point) > 20:  # Just a simple check for the number of points
        return True
    return False



# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        # Draw hand landmarks
        mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)
        
        hand1 = result.multi_hand_landmarks[0]
        hand2 = result.multi_hand_landmarks[1]

        if open_folder and is_unlock_sign(hand1, hand2):
            open_folder()
            
            
        elif close_folder and is_lock_sign(hand1, hand2):
            close_folder()
    elif result.multi_hand_landmarks:
          for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
              mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the position of the index finger tip
              index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
              h, w, _ = frame.shape
              x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
              cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

              # Collect points for drawing
              if drawing_mode:
                  drawing_point.append((x, y))
                  for i in range(1, len(drawing_point)):
                      cv2.line(frame, drawing_point[i - 1], drawing_point[i], (255, 0, 0), 2)

    if len(drawing_point) > 0 and not drawing_mode:
               if recognize_shape(drawing_point):
                  chrome_browser()
                  drawing_point = []  # Clear points after action
 
    cv2.imshow('Hand Sign Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('d'):
        drawing_mode = not drawing_mode
        if not drawing_mode:
            drawing_point = []  # Clear points when stopping  


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
