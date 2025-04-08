#https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      #print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
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
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:

      L_Max_Left = 0
      L_Max_Right = 0
      L_Max_Up = 0
      L_Max_Down = 0

      for face_landmarks in results.multi_face_landmarks:
        #mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
      
        left_iris = [face_landmarks.landmark[i] for i in range(474, 478)]
        right_iris = [face_landmarks.landmark[i] for i in range(469, 473)]

        LEFT_EYE_Points =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
        RIGHT_EYE_Points=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] #x:246 [15]/173 [9] y:  159 [12]/ 145 [4]
        left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_Points]#range(370,385)]#range(33, 133)]
        right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_Points]#range(144,161)]#range(362, 463)]

        L_Max_Left = left_eye[0].x
        L_Max_Right = left_eye[0].x
        L_Max_Up = left_eye[0].y
        L_Max_Down = left_eye[0].y

        for point in left_eye:
          if point.x <L_Max_Left:
            L_Max_Left = point.x
          if point.x >L_Max_Right:
            L_Max_Right = point.x
          if point.y <L_Max_Up:
            L_Max_Up = point.y
          if point.y >L_Max_Down:
            L_Max_Down = point.y
          pass


        #print('-----------------------------------------------------------------------------------------------')
      
      
      # Draw iris landmarks - was for testing
#      for landmark in left_iris + right_iris:
#        try:
#          x = int(landmark.x * image.shape[1])
#          y = int(landmark.y * image.shape[0])
#          cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
#        except:
#          pass
      for landmark in right_eye + left_eye: 
        try:
          x = int(landmark.x * image.shape[1])
          y = int(landmark.y * image.shape[0])
          cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        except:
          pass

      #center of right iris # [0] right; [1] up ;[2] left; [3] down
      right_iris_center_x = (right_iris[0].x + right_iris[2].x)/2
      right_iris_center_y = (right_iris[3].y + right_iris[1].y)/2
    
      cv2.circle(image, (int(right_iris_center_x * image.shape[1]), int(right_iris_center_y * image.shape[0])), 2, (0, 255, 0), -1)      

      #center of left iris # [0] right; [1] up ;[2] left; [3] down
      left_iris_center_x = (left_iris[0].x + left_iris[2].x)/2
      left_iris_center_y = (left_iris[3].y + left_iris[1].y)/2
      cv2.circle(image, (int(left_iris_center_x * image.shape[1]), int(left_iris_center_y * image.shape[0])), 2, (0, 255, 0), -1)  

      #center of right eye
      right_eye_center_x = (right_eye[9].x + right_eye[15].x)/2 # x: 246 [15]/173 [9]
      right_eye_center_y = (right_eye[4].y + right_eye[12].y)/2 # y: 159 [12]/145 [4]
      cv2.circle(image, (int(right_eye_center_x * image.shape[1]), int(right_eye_center_y * image.shape[0])), 2, (0, 0, 255), -1)

      #detect for mouse control
      right_eye_correction = 0.30
      right_eye_width = abs ((right_eye[9].x - right_eye[15].x) * right_eye_correction)
      right_eye_distance = abs(right_iris_center_x - right_eye_center_x)

      temp = right_eye_width / (2 * 3) # 2 = half of eye; 3 = 3 parts of eye
      if temp > right_eye_distance:
        print('idle')
      elif temp * 2 > right_eye_distance:
        print('Speed I - slow')
      else:
        print('Speed II fast')

      '''image = cv2.flip(image,1)
      cv2.putText(image, str(left_iris), (200,200), cv2.FONT_HERSHEY_COMPLEX, 1, 50, 2, cv2.LINE_AA, False)
      image = cv2.flip(image,1)'''

      '''
      for face_landmarks in results.multi_face_landmarks:
        #print(face_landmarks)
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
      '''
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    #if cv2.waitKey(5) & 0xFF == 27:
    #  break
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()