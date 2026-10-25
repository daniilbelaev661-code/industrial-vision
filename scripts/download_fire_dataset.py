from roboflow import Roboflow


rf = Roboflow(api_key="qzYFawjIO952sSmvUm7F")


project = rf.workspace("proyectosnic").project("fire-aonru-0kgon")
version = project.version(1)


dataset = version.download("yolov8")

print(f"Датасет успешно скачан в: {dataset.location}")