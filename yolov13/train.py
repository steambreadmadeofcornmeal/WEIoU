from ultralytics import YOLO
model = YOLO(
    # "O:\\anaconda\\envs\\for_sota\\yolov13\\yolov13n.yaml"
    # "O:\\anaconda\\envs\\for_sota\\yolov13\\runs\\detect\\train3\\weights\\best.pt"
    "O:\\anaconda\\envs\\for_sota\\yolov13\\runs\\detect\\train198\\weights\\last.pt"
    )

results = model.train(
    data='O:\\anaconda\\envs\\for_sota\\yolov13\\crowdhuman.yaml',
    # data='O:\\anaconda\\envs\\for_sota\\yolov13\\widerperson.yaml',
    epochs=90,
    batch=24,
    imgsz=640,
    scale=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.1,
    device='0',
    workers=0,
    val=True,
    resume=True,
    lr0=0.01,
    lrf=0.00001,
    dfl=0,
    seed=4096,
    # box=6,
    # cls=1,
    # pretrained=False,
)
# metrics = model.val(
#     data='O:\\anaconda\\envs\\for_sota\\yolov13\\widerperson.yaml',
#     batch=1,
#     workers=0,
#     )
