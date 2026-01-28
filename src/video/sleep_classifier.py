import cv2
import mediapipe as mp
import os

# ================= MEDIAPIPE =================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133]
RIGHT_EYE = [362, 385, 387, 263]

def eye_aspect_ratio(landmarks, eye):
    v1 = abs(landmarks[eye[1]].y - landmarks[eye[2]].y)
    v2 = abs(landmarks[eye[0]].x - landmarks[eye[3]].x)
    return v1 / v2 if v2 != 0 else 0


def analyze_uploaded_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)

    EAR_SLEEP_THRESHOLD = 0.20
    MIN_STATE_DURATION_SEC = 3      # minimum continuous state
    MIN_STATE_FRAMES = fps * MIN_STATE_DURATION_SEC

    sleep_segments = 0
    wake_segments = 0
    sleep_frames = 0

    current_state = "WAKE"
    state_counter = 0

    total_frames = 0

    print("\n🟢 ANALYZING UPLOADED VIDEO\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        detected_state = "WAKE"

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark
            ear = (
                eye_aspect_ratio(face, LEFT_EYE) +
                eye_aspect_ratio(face, RIGHT_EYE)
            ) / 2

            if ear < EAR_SLEEP_THRESHOLD:
                detected_state = "SLEEP"

        # ---------- STATE TRANSITION ----------
        if detected_state == current_state:
            state_counter += 1
        else:
            if state_counter >= MIN_STATE_FRAMES:
                if current_state == "SLEEP":
                    sleep_segments += 1
                else:
                    wake_segments += 1

            current_state = detected_state
            state_counter = 1

        if detected_state == "SLEEP":
            sleep_frames += 1

    # ---------- FINAL STATE COUNT ----------
    if state_counter >= MIN_STATE_FRAMES:
        if current_state == "SLEEP":
            sleep_segments += 1
        else:
            wake_segments += 1

    cap.release()

    total_sleep_time_sec = sleep_frames / fps
    total_video_time_sec = total_frames / fps

    final_state = "SLEEP" if total_sleep_time_sec > (0.5 * total_video_time_sec) else "WAKE"

    # ---------- SUMMARY ----------
    summary = {
        "sleep_segments": sleep_segments,
        "wake_segments": wake_segments,
        "total_sleep_seconds": round(total_sleep_time_sec, 2),
        "total_video_seconds": round(total_video_time_sec, 2),
        "final_state": final_state
    }

    return summary


# ================= MAIN =================
if __name__ == "__main__":
    VIDEO_PATH = "C:\\Users\\HAI\\Music\\acceleromete_main_project\\4880318-uhd_3840_2160_25fps.mp4"   # 👈 upload video here

    result = analyze_uploaded_video(VIDEO_PATH)

    print("\n📊 SLEEP ANALYSIS REPORT")
    print("------------------------")
    print(f"Sleep Segments      : {result['sleep_segments']}")
    print(f"Wake Segments       : {result['wake_segments']}")
    print(f"Total Sleep (sec)   : {result['total_sleep_seconds']}")
    print(f"Total Video (sec)   : {result['total_video_seconds']}")
    print(f"\n✅ FINAL RESULT → {result['final_state']}")
