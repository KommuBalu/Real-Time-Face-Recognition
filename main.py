try:
    import cv2  
    import ultralytics  
except ImportError as exc:
    raise SystemExit(
        "Missing required packages. Install them with: pip install opencv-python ultralytics"
    ) from exc

model = ultralytics.YOLO("yolo11n.pt")
# model = YOLO("image_of_Headphones.pt")
# model = YOLO("my_objects.pt")
# results = model(frame)
camera = cv2.VideoCapture(0)


camera.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)


camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("Camera could not be opened")
    exit()

print("Camera started")
print("Press Q to quit")

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera")
        break
    results = model(
        frame,
        imgsz=320,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("Q"): 
        break

camera.release()
cv2.destroyAllWindows()