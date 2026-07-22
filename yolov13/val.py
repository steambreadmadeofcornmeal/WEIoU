from ultralytics import YOLO
model = YOLO(
    # "O:\\anaconda\\envs\\for_sota\\yolov13\\yolov13n0722.pt"
    "O:\\anaconda\\envs\\for_sota\\yolov13\\runs\\detect\\train90\\weights\\best.pt"
    )

results = model.train(
    data='O:\\anaconda\\envs\\for_sota\\yolov13\\widerperson.yaml',
    epochs=1,
    batch=16,
    imgsz=640,
    scale=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.1,
    device='0',
    workers=0,
    val=True,
    # resume=True,
    lr0=0.01,
    lrf=0.00001,
    # pretrained=False,
    freeze=[i for i in range(32)]
)
# metrics = model.val(
#     data='O:\\anaconda\\envs\\for_sota\\yolov13\\widerperson.yaml',
#     # data='O:\\anaconda\\envs\\for_sota\\yolov13\\crowdhuman.yaml',
#     batch=1,
#     workers=0,
#     # save=True,
#     # plots=True,
#     )
