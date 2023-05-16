import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from keypoint_classifier import KeyPointClassifier
from pynput.keyboard import Key, Controller

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
keypoint_classifier = KeyPointClassifier()

keyboard = Controller()
playing = False
mode = 0
number = 4 # number of poses
default = False
hand_sign_id = 0
#pose index
#0 - right
#1 - left
#2 - rotate
#3 - harddrop
#4 - hold
#5 - Default Position

start_time = 0

def pre_process_landmark(results):
    land_mark_list = []
    
    for data_points in results.pose_world_landmarks.landmark:
      land_mark_list.append(data_points.x)
      land_mark_list.append(data_points.y)
      land_mark_list.append(data_points.z)
    return land_mark_list
       

    

def logging_csv(results, number, mode):
    land_marks = pre_process_landmark(results)
    if mode == 0:
       pass
    if mode == 1 and (0 <= number <= 9):
       csv_path = 'keypoint.csv'
       with open(csv_path, 'a', newline="") as f:
          writer = csv.writer(f)
          writer.writerow([number,*land_marks])
    return
# Pose Classification, 0_x, 0_y, 0_z, 1_x, 1_y, 1_z, ....

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.8) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    


    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    collecting = False
    
    key=cv2.waitKey(1)
    if key == 110: #n
      mode = 0
    if key == 107: #k
      mode = 1
      start_time = time.time()
    if key == 104: #h
      if playing :
          playing = False
      elif playing == False:
          print("playing")
          playing = True
    if key == 27:
      break
    if time.time()-start_time > 5 and mode == 1:
      collecting = True
      print("collecting is true")
    if collecting == True:   
      logging_csv(results, number, mode)
      if time.time()-start_time > 30:
          mode = 0
          collecting = False
          print("done collecting")
    if mode == 0:
      try:  
        hand_sign_id = keypoint_classifier(pre_process_landmark(results))
        
      except:
        print("No Pose Detected")

      if playing: # PLAYING TETRIS
#pose index
#0 - right
#1 - left
#2 - rotate
#3 - harddrop
#4 - hold
#5 - Default Position
        if hand_sign_id == 5:
           default = True
           print(hand_sign_id)
        if default == True:
          if (hand_sign_id == 1):
            print(hand_sign_id)
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            default = False
          elif (hand_sign_id == 0):
            keyboard.press(Key.right)
            keyboard.release(Key.right)    
            default = False        
          elif (hand_sign_id == 2):
            keyboard.press('x')
            keyboard.release('x')
            default = False
          elif (hand_sign_id == 3):
            keyboard.press(Key.space)
            keyboard.release(Key.space)
            default = False
        if (hand_sign_id == 4):
          keyboard.press(Key.shift_l)
          keyboard.release(Key.shift_l)
          default = False
          


    
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
cap.release()