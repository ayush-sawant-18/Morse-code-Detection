import cv2
import mediapipe as mp
import numpy as np
import time

# --- Morse Code Dictionary ---
MORSE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
    '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7',
    '---..': '8', '----.': '9'
}

def morse_to_text(code):
    return MORSE_DICT.get(code, '?')

# --- Eye Aspect Ratio Calculation ---
def eye_aspect_ratio(landmarks, eye_indices):
    eye = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    vertical1 = np.linalg.norm(eye[1] - eye[5])
    vertical2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    if horizontal == 0: return 0.0
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# --- EMA Filter Function ---
def apply_ema(current_value, prev_ema_value, alpha):
    if prev_ema_value is None: return current_value
    return alpha * current_value + (1 - alpha) * prev_ema_value

# --- Constants ---
# Timing & Thresholds
EAR_THRESHOLD = 0.21
NATURAL_BLINK_MAX_DURATION = 0.25
DOT_DURATION_RANGE = (0.15, 0.40)
DASH_DURATION_RANGE = (0.40, 1.0)
EYE_COOLDOWN_PERIOD = 0.4
DOT_DISTANCE = 0.05
DASH_DISTANCE = 0.15
HOLD_TIME = 0.25
CHARACTER_PAUSE = 1.8
WORD_PAUSE = 2.0
ERASE_HOLD_TIME = 1.5 # Longer hold for erase
ERASE_DISTANCE_THRESHOLD = 0.15 # Example threshold for erase gesture
# Landmark Indices
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# Filtering
EMA_ALPHA = 0.3

# --- UI Styling ---
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR_MODE = (0, 180, 220)
ACCENT_COLOR_RESULT = (100, 255, 150)
ACCENT_COLOR_INSTR = (255, 180, 120)
ACCENT_COLOR_FEEDBACK = (255, 255, 0)
ACCENT_COLOR_PAUSED = (255, 100, 100) # Color for paused indicator
ACCENT_COLOR_RUNNING = (100, 255, 100) # Color for running indicator
PANEL_COLOR = (50, 50, 50)
BUTTON_COLOR = (80, 80, 80)
BUTTON_TEXT_COLOR = (230, 230, 230)
BUTTON_HOVER_COLOR = (110, 110, 110)
ALPHA = 0.75
FEEDBACK_EYE_COLOR = (0, 255, 255)
FEEDBACK_HAND_DOT_COLOR = (255, 255, 0)
FEEDBACK_HAND_DASH_COLOR = (0, 165, 255)
FEEDBACK_HAND_LINE_COLOR = (0, 255, 0)
FEEDBACK_HAND_ERASE_COLOR = (255, 0, 0)

# --- Button Definitions & Mouse State ---
BUTTON_RECTS = {}
BUTTON_PADDING = 10
BUTTON_HEIGHT = 35
trigger_switch_mode = False
trigger_clear = False
trigger_quit = False
trigger_toggle_detection = False # New trigger flag for Start/Stop
trigger_erase_button = False # Trigger for the Erase button
mouse_pos = None

# --- Mouse Callback Function ---
def handle_mouse_click(event, x, y, flags, param):
    global trigger_switch_mode, trigger_clear, trigger_quit, trigger_toggle_detection, trigger_erase_button, mouse_pos
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        for name, rect in BUTTON_RECTS.items():
            bx, by, bw, bh = rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                print(f"Button Clicked: {name}")
                if name == "Switch Mode": trigger_switch_mode = True
                elif name == "Clear": trigger_clear = True
                elif name == "Quit": trigger_quit = True
                elif name == "Start" or name == "Stop": # Check for dynamic text
                    trigger_toggle_detection = True
                elif name == "Erase": trigger_erase_button = True
                break

# --- UI Drawing Function ---
def draw_ui(frame, morse_code, word_buffer, translated_text, mode, w, h,
            is_detecting,
            avg_ear=None, distance=None, is_blinking=False,
            is_eye_cooldown=False, current_gesture_feedback=None,
            is_erasing_gesture_feedback=False): # Add feedback for erase gesture
    global BUTTON_RECTS, mouse_pos

    overlay = frame.copy()
    top_panel_h = 180
    bottom_panel_h = 60
    padding = 15

    # --- Draw Panels ---
    cv2.rectangle(overlay, (0, 0), (w, top_panel_h), PANEL_COLOR, -1)
    cv2.rectangle(overlay, (0, h - bottom_panel_h), (w, h), PANEL_COLOR, -1)
    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)

    # --- Top Panel Content ---
    y_pos = 30
    # Mode and Detection Status Indicator
    status_text = "[RUNNING]" if is_detecting else "[PAUSED]"
    status_color = ACCENT_COLOR_RUNNING if is_detecting else ACCENT_COLOR_PAUSED
    mode_str = f"Mode: {mode.upper()} DETECTION "
    cv2.putText(frame, mode_str, (padding, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, ACCENT_COLOR_MODE, 2, cv2.LINE_AA)
    # Get text size to position status text next to mode text
    (mode_w, _), _ = cv2.getTextSize(mode_str, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(frame, status_text, (padding + mode_w, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
    y_pos += 35

    # --- Dynamic Feedback Area ---
    feedback_text = ""
    if not is_detecting:
        feedback_text = "Detection Paused"
    elif mode == "eye":
        if avg_ear is not None: feedback_text += f"EAR: {avg_ear:.3f} "
        if is_blinking: feedback_text += "[Blinking] "
        if is_eye_cooldown: feedback_text += "[Cooldown]"
    elif mode == "hand":
        if distance is not None: feedback_text += f"Dist: {distance:.3f} "
        if current_gesture_feedback == 'dot': feedback_text += "[DOT]"
        elif current_gesture_feedback == 'dash': feedback_text += "[DASH]"
        elif current_gesture_feedback == 'neutral': feedback_text += "[Neutral]"
        if is_erasing_gesture_feedback: feedback_text += "[ERASING]"

    cv2.putText(frame, feedback_text, (padding, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_COLOR_FEEDBACK if is_detecting else ACCENT_COLOR_PAUSED, 2, cv2.LINE_AA)
    y_pos += 35

    # --- Morse/Text Output ---
    cv2.putText(frame, f"Input: {morse_code}", (padding, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_COLOR_MODE, 2, cv2.LINE_AA)
    y_pos += 30
    cv2.putText(frame, f"Word: {word_buffer}", (padding, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
    y_pos += 30
    cv2.putText(frame, f"Result: {translated_text}", (padding, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.75, ACCENT_COLOR_RESULT, 2, cv2.LINE_AA)

    # --- Bottom Panel Content (Buttons) ---
    button_y = h - bottom_panel_h + (bottom_panel_h - BUTTON_HEIGHT) // 2
    # Define button texts dynamically
    start_stop_text = "Stop" if is_detecting else "Start"
    button_texts = [start_stop_text, "Switch Mode", "Clear", "Erase", "Quit"] # Add Erase button

    button_widths = []
    for text in button_texts:
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        button_widths.append(text_w + 2 * BUTTON_PADDING)
    button_total_width = sum(button_widths) + (len(button_texts) - 1) * BUTTON_PADDING
    current_x = max(padding, (w - button_total_width) // 2) # Ensure buttons start within frame padding

    BUTTON_RECTS.clear()

    for i, text in enumerate(button_texts):
        bw = button_widths[i]
        bx = current_x
        # Prevent button going off-screen if window is too narrow
        if bx + bw > w - padding:
            bw = max(0, w - padding - bx)
            if bw == 0: continue # Skip button if no space

        rect = (bx, button_y, bw, BUTTON_HEIGHT)
        BUTTON_RECTS[text] = rect # Use text as key, handles dynamic Start/Stop text

        button_bg_color = BUTTON_COLOR
        # Hover Effect
        if mouse_pos:
            mx, my = mouse_pos
            if bx <= mx <= bx + bw and button_y <= my <= button_y + BUTTON_HEIGHT:
                button_bg_color = BUTTON_HOVER_COLOR

        cv2.rectangle(frame, (bx, button_y), (bx + bw, button_y + BUTTON_HEIGHT), button_bg_color, -1)
        cv2.rectangle(frame, (bx, button_y), (bx + bw, button_y + BUTTON_HEIGHT), TEXT_COLOR, 1)

        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Center text only if button width is sufficient
        if bw >= text_w:
            text_x = bx + (bw - text_w) // 2
        else:
            text_x = bx + 2 # Left align if too small
        text_y = button_y + (BUTTON_HEIGHT + text_h) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, BUTTON_TEXT_COLOR, 2, cv2.LINE_AA)
        current_x += bw + BUTTON_PADDING

# --- Main Function ---
def main():
    global trigger_switch_mode, trigger_clear, trigger_quit, trigger_toggle_detection, trigger_erase_button

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: Cannot open camera."); return

    # MediaPipe Setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # --- State Variables ---
    morse_code, translated_text, word_buffer = "", "", ""
    last_input_time = time.time()
    mode = "eye"
    is_detecting = True # Start in detecting state
    is_erasing_gesture = False
    erase_start_time = None
    # Detection specific states
    blinking = False
    blink_start_time = None
    last_blink_registered_time = 0
    last_gesture, gesture_start_time = None, None
    # EMA Filter states
    prev_thumb_pt_filt = None
    prev_index_pt_filt = None
    # UI State Variables
    avg_ear_ui, distance_ui, is_eye_cooldown_ui, current_gesture_feedback_ui, is_erasing_gesture_feedback_ui = None, None, False, None, False

    window_name = "Morse Code Translator (Eye + Hand)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handle_mouse_click)

    print("Starting Morse Code Translator...")
    print("Modes: Eye Blink Detection, Hand Gesture Detection")
    print("Use UI Buttons to Start/Stop, Switch Mode, Clear Text, Erase, or Quit.")

    while True:
        # --- Handle Button Actions ---
        if trigger_quit: print("Quit button pressed. Exiting..."); break
        if trigger_clear:
            print("Clear button pressed.")
            morse_code, translated_text, word_buffer = "", "", ""
            last_input_time = time.time(); trigger_clear = False
        if trigger_switch_mode:
            mode = "hand" if mode == "eye" else "eye"
            print(f"Switch Mode button pressed. Switched to {mode.upper()} detection.")
            morse_code = ""; word_buffer = ""
            last_gesture, gesture_start_time = None, None
            blinking, blink_start_time = False, None
            last_input_time = time.time()
            prev_thumb_pt_filt = None; prev_index_pt_filt = None
            avg_ear_ui, distance_ui, is_eye_cooldown_ui, current_gesture_feedback_ui, is_erasing_gesture_feedback_ui = None, None, False, None, False
            trigger_switch_mode = False
        if trigger_toggle_detection:
            is_detecting = not is_detecting # Toggle detection state
            print(f"Detection {'Started' if is_detecting else 'Paused'}.")
            if not is_detecting: # Reset intermediate states when pausing
                blinking, blink_start_time = False, None
                last_gesture, gesture_start_time = None, None
                prev_thumb_pt_filt = None; prev_index_pt_filt = None
                is_erasing_gesture = False
            trigger_toggle_detection = False # Reset flag
        if trigger_erase_button:
            print("Erase button pressed.")
            if word_buffer:
                word_buffer = word_buffer[:-1]
                print(f"Erased last letter. Word: '{word_buffer}'")
            trigger_erase_button = False

        # --- Frame Capture ---
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_time = time.time()

        # Reset UI feedback variables for this frame
        avg_ear_ui, distance_ui, current_gesture_feedback_ui, is_erasing_gesture_feedback_ui = None, None, None, False
        is_eye_cooldown_ui = (current_time - last_blink_registered_time < EYE_COOLDOWN_PERIOD) if mode == "eye" else False

        # --- DETECTION BLOCK (Conditional) ---
        if is_detecting:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            if mode == "hand":
                results_hand = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True
                if results_hand.multi_hand_landmarks:
                    hand_landmarks = results_hand.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                    thumb_tip = hand_landmarks.landmark[4]; index_tip = hand_landmarks.landmark[8]
                    thumb_pt = np.array([thumb_tip.x * w, thumb_tip.y * h]); index_pt = np.array([index_tip.x * w, index_tip.y * h])
                    thumb_pt_filt = apply_ema(thumb_pt, prev_thumb_pt_filt, EMA_ALPHA); index_pt_filt = apply_ema(index_pt, prev_index_pt_filt, EMA_ALPHA)
                    prev_thumb_pt_filt = thumb_pt_filt; prev_index_pt_filt = index_pt_filt
                    distance_normalized = np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])
                    distance_ui = distance_normalized

                    current_gesture = None
                    if distance_normalized < DOT_DISTANCE: current_gesture = '.'; current_gesture_feedback_ui = 'dot'
                    elif distance_normalized > DASH_DISTANCE: current_gesture = '-'; current_gesture_feedback_ui = 'dash'
                    elif distance_normalized > ERASE_DISTANCE_THRESHOLD:
                        is_erasing_gesture = True
                        is_erasing_gesture_feedback_ui = True
                        if erase_start_time is None:
                            erase_start_time = current_time
                        elif current_time - erase_start_time > ERASE_HOLD_TIME:
                            if word_buffer:
                                word_buffer = word_buffer[:-1]
                                print(f"Hand Gesture Erase: Removed last letter. Word: '{word_buffer}'")
                                last_input_time = current_time # Reset input time to avoid accidental new letters
                            erase_start_time = None
                            is_erasing_gesture = False
                    else:
                        current_gesture_feedback_ui = 'neutral'
                        is_erasing_gesture = False
                        erase_start_time = None

                    if not is_erasing_gesture:
                        if current_gesture != last_gesture:
                            gesture_start_time = current_time
                            last_gesture = current_gesture
                        if current_gesture and gesture_start_time and (current_time - gesture_start_time > HOLD_TIME):
                            if current_time - last_input_time > 0.2:
                                morse_code += current_gesture
                                last_input_time = current_time
                                print(f"Hand Gesture Registered: {current_gesture}")
                            last_gesture, gesture_start_time = None, None # Reset after register

                    # Drawing uses filtered points
                    line_color = FEEDBACK_HAND_LINE_COLOR
                    if current_gesture_feedback_ui == 'dot': line_color = FEEDBACK_HAND_DOT_COLOR
                    elif current_gesture_feedback_ui == 'dash': line_color = FEEDBACK_HAND_DASH_COLOR
                    elif is_erasing_gesture_feedback_ui: line_color = FEEDBACK_HAND_ERASE_COLOR
                    thumb_pt_draw = tuple(np.round(thumb_pt_filt).astype(int)); index_pt_draw = tuple(np.round(index_pt_filt).astype(int))
                    cv2.line(frame, thumb_pt_draw, index_pt_draw, line_color, 3)
                    cv2.circle(frame, thumb_pt_draw, 8, (255, 0, 0), -1); cv2.circle(frame, index_pt_draw, 8, (0, 0, 255), -1)
                else: # No hand detected
                    last_gesture, gesture_start_time = None, None
                    prev_thumb_pt_filt = None; prev_index_pt_filt = None
                    distance_ui = None
                    is_erasing_gesture = False
                    erase_start_time = None

            elif mode == "eye":
                results_face = face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True
                if results_face.multi_face_landmarks:
                    face_landmarks = results_face.multi_face_landmarks[0]; landmarks = face_landmarks.landmark
                    for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX: pt = landmarks[idx]; center = (int(pt.x * w), int(pt.y * h)); cv2.circle(frame, center, 1, FEEDBACK_EYE_COLOR, -1)
                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX); right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
                    avg_ear_ui = (left_ear + right_ear) / 2.0

                    if avg_ear_ui < EAR_THRESHOLD:
                        if not blinking: blinking = True; blink_start_time = current_time
                    else: # Eyes open
                        if blinking:
                            blink_duration = current_time - blink_start_time
                            blinking = False # Reset blink state *here*
                            if not is_eye_cooldown_ui:
                                blink_type = None
                                if DOT_DURATION_RANGE[0] <= blink_duration <= DOT_DURATION_RANGE[1]: morse_code += '.'; blink_type = f"DOT (.) Duration={blink_duration:.2f}s"
                                elif DASH_DURATION_RANGE[0] <= blink_duration <= DASH_DURATION_RANGE[1]: morse_code += '-'; blink_type = f"DASH (-) Duration={blink_duration:.2f}s"
                                if blink_type:
                                    print(f"Blink Registered: {blink_type}"); last_input_time = current_time; last_blink_registered_time = current_time
                                    # is_eye_cooldown_ui = True # Let next frame update this based on time check
                else: # No face detected
                    blinking, blink_start_time = False, None
                    avg_ear_ui = None
        # --- END OF CONDITIONAL DETECTION BLOCK ---

        # --- Morse Code Translation (Always runs if morse_code has content) ---
        time_since_last = current_time - last_input_time
        if morse_code and time_since_last > CHARACTER_PAUSE:
            letter = morse_to_text(morse_code)
            word_buffer += letter
            print(f"Character Decoded: '{morse_code}' -> '{letter}' | Word: '{word_buffer}'")
            morse_code = ""; last_input_time = current_time
        if word_buffer and time_since_last > WORD_PAUSE:
            translated_text += word_buffer + " "; print(f"Word Finished: '{word_buffer}' | Full Text: '{translated_text}'")
            word_buffer = ""; last_input_time = current_time

        # --- Draw UI ---
        draw_ui(frame, morse_code, word_buffer, translated_text, mode, w, h,
                is_detecting=is_detecting,
                avg_ear=avg_ear_ui, distance=distance_ui, is_blinking=blinking if is_detecting else False,
                is_eye_cooldown=is_eye_cooldown_ui if is_detecting else False,
                current_gesture_feedback=current_gesture_feedback_ui if is_detecting else None,
                is_erasing_gesture_feedback=is_erasing_gesture_feedback_ui if is_detecting and mode == "hand" else False)

        # --- Display ---
        cv2.imshow(window_name, frame)

        # --- WaitKey ---
        if cv2.waitKey(1) & 0xFF == 27: print("ESC key pressed. Exiting..."); break

    # --- Cleanup ---
    cap.release(); cv2.destroyAllWindows()
    if 'hands' in locals(): hands.close()
    if 'face_mesh' in locals(): face_mesh.close()
    print("Application Closed.")

if __name__ == "__main__":
    main()    