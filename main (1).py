import cv2
import mediapipe as mp
import pyautogui

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()
cam = cv2.VideoCapture(0)

index_y = 0
pndex_y = 0
andex_y = 0

hand_detected = False

while True:
    _, frame = cam.read()

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[274:276]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1 and not hand_detected:  # Only move cursor if hand is not detected
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)

        lip_eye = [landmarks[13], landmarks[14]]
        for landmark in lip_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
        if cv2.norm((lip_eye[0].x, lip_eye[0].y), (lip_eye[1].x, lip_eye[1].y)) > 0.04:
            pyautogui.scroll(200)
            pyautogui.sleep(0)

        scroll_up = [landmarks[12], landmarks[15]]
        for landmark in scroll_up:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
        if cv2.norm((scroll_up[0].x, scroll_up[0].y), (scroll_up[1].x, scroll_up[1].y)) < 0.01:
            pyautogui.scroll(-200)
            pyautogui.sleep(0)

        right_eye = [landmarks[374], landmarks[386]]
        for landmark in right_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
        if cv2.norm((right_eye[0].x, right_eye[0].y), (right_eye[1].x, right_eye[1].y)) < 0.01:
            right_eye_distance = cv2.norm((right_eye[0].x, right_eye[0].y), (right_eye[1].x, right_eye[1].y))
            pyautogui.click(button='right')
            pyautogui.sleep(0)

        left_eye = [landmarks[145], landmarks[159]]
        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))
        if cv2.norm((left_eye[0].x, left_eye[0].y), (left_eye[1].x, left_eye[1].y)) < 0.01:
            pyautogui.click(button='left')
            pyautogui.sleep(0)

    hand_output = hand_detector.process(rgb_frame)
    hands = hand_output.multi_hand_landmarks
    if hands:
        hand_detected = True
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                if id == 8:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    index_x = screen_w / frame_w * x
                    index_y = screen_h / frame_h * y

                if id == 4:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    thumb_x = screen_w / frame_w * x
                    thumb_y = screen_h / frame_h * y
                    print('up', abs(index_y - thumb_y))
                    if abs(index_y - thumb_y) < 80:
                        pyautogui.scroll(200)

                        pyautogui.sleep(0)

                if id == 12:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    pndex_x = screen_w / frame_w * x
                    pndex_y = screen_h / frame_h * y

                if id == 3:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    shumb_x = screen_w / frame_w * x
                    shumb_y = screen_h / frame_h * y
                    print('down', abs(pndex_y - shumb_y))
                    if abs(pndex_y - shumb_y) < 50:
                        pyautogui.scroll(-200)
                        pyautogui.sleep(0)

                if id == 16:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    andex_x = screen_w / frame_w * x
                    andex_y = screen_h / frame_h * y

                if id == 2:
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    zhumb_x = screen_w / frame_w * x
                    zhumb_y = screen_h / frame_h * y
                    print('click', abs(andex_y - zhumb_y))
                    if abs(andex_y - zhumb_y) < 30:
                        pyautogui.click(button='left')
                        pyautogui.sleep(0)

                if id == 9:
                    screen_x = screen_w / frame_w * x
                    screen_y = screen_h / frame_h * y
                    pyautogui.moveTo(screen_x, screen_y)

    cv2.imshow('Combined Interactions', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()