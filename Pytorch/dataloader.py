from roboflow import Roboflow
rf = Roboflow(api_key="FQKF006V72uaXh4LP6Sa")
project = rf.workspace("rdd-o05aa").project("rdd-model-29dpv")
version = project.version(5)
dataset = version.download("yolov11")
                