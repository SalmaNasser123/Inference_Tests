import subprocess


input_video_path = "/home/roboticscorner/Python/people_with_helmet_working_1920_1080_50fps.mp4"

weights_path = "/home/roboticscorner/Downloads/best.pt"

Yolo_Detect_file = "/home/roboticscorner/Python/yolov7/detect.py"

command = [
    "python3", Yolo_Detect_file,
    "--source", input_video_path,
    "--weights", weights_path,
    "--name", "video_1"
    "--view-img"
]

subprocess.run(command)
