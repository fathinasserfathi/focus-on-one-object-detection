from ultralytics import YOLO
import cv2




model = YOLO("yolo26m.pt")
cap = cv2.VideoCapture(0)

selected_id = None
current_results = None
tracker = None
selected_class = None
def select_object(event, x, y, flags, param):
    global selected_id, current_results, tracker, frame,selected_class
    if event == cv2.EVENT_LBUTTONDOWN and current_results is not None:
        for box in current_results[0].boxes:
            if box.id is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id)
            class_id = int(box.cls)
            if x1 < x < x2 and y1 < y < y2:
                selected_id = track_id
                selected_class = class_id
                bbox = (x1, y1, x2-x1, y2-y1)
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, bbox)
                print("Started tracking ID:", selected_id)
                break

cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Tracking", select_object)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)
    current_results = results

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id) if box.id is not None else -1
        class_id= int(box.cls)
        label = model.names[class_id]
        if selected_id is None:
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if selected_id is not None and tracker is not None:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            label = model.names[selected_class]
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "Lost target!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        tracker = None
        selected_id = None
        print("Reset")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()