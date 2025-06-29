import os, zipfile, cv2

# 1) Configure your paths. 
#    Adjust these to wherever you really have your .zip files.
crops_zip_dir = r"C:\Users\PRAGYA\Desktop\PROGRAMMING\4th sem\Hand_Gestures\A Video Dataset of the Hand Gestures of ISL\Cropped_Data"
videos_dir    = r"C:\Users\PRAGYA\Desktop\PROGRAMMING\4th sem\Hand_Gestures\video_dataset"
frames_dir    = r"C:\Users\PRAGYA\Desktop\PROGRAMMING\4th sem\Hand_Gestures\gesture_frames"

os.makedirs(videos_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

# 2) Unzip each ZIP into videos_dir/<gesture_label>/
for z in os.listdir(crops_zip_dir):
    if not z.lower().endswith(".zip"):
        continue
    label = z.replace("_Cropped.zip", "")
    dest  = os.path.join(videos_dir, label)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(os.path.join(crops_zip_dir, z), 'r') as zf:
        zf.extractall(dest)
    print(f"✅ Unzipped {z} → {dest}")

# 3) Now extract frames from each video into frames_dir/<gesture_label>/
for label in os.listdir(videos_dir):
    class_vids   = os.path.join(videos_dir, label)
    class_frames = os.path.join(frames_dir, label)
    os.makedirs(class_frames, exist_ok=True)

    for vid in os.listdir(class_vids):
        if not vid.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        cap   = cv2.VideoCapture(os.path.join(class_vids, vid))
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # save every 5th frame to keep dataset size manageable
            if count % 5 == 0:
                fn = f"{os.path.splitext(vid)[0]}_f{count:03d}.jpg"
                cv2.imwrite(os.path.join(class_frames, fn), frame)
            count += 1
        cap.release()

    print(f"✅ Extracted frames for {label} → {class_frames}")
