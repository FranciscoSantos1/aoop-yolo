from roboflow import Roboflow

rf = Roboflow(api_key="buKD5rrRQcnQBMbhAVwu")

project = rf.workspace("aoop-vgwve").project("driver-behavior-wc25q")
version = project.version(4)
dataset = version.download("yolov8")
                