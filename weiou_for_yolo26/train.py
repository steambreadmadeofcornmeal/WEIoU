from ultralytics import YOLO
model = YOLO(
    # "O:\\anaconda\\envs\\for_sota\\yolov13\\yolov13n.yaml"
    # "O:\\anaconda\\envs\\for_sota\\yolov13\\runs\\detect\\train3\\weights\\best.pt"
    # "O:\\anaconda\\envs\\for_sota\\yolov13\\runs\\detect\\train114\\weights\\last.pt"
    "yolo26n.pt"
    # "O:\\anaconda\\envs\\for_sota\\yolo26\\ultralytics\\runs\\detect\\train44\\weights\\last.pt"
    
    )

#仅对从配置文件训练有用
# model.model.args["reg_max"] = 16

results = model.train(
    # data='O:\\anaconda\\envs\\for_sota\\yolov13\\crowdhuman.yaml',
    data='O:\\anaconda\\envs\\for_sota\\yolov13\\widerperson.yaml',
    epochs=20,
    batch=24,
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
    # dfl=0,
    # box=6,
    # cls=1,
    # pretrained=False,
    # seed=64,
)
# metrics = model.val(
#     data='O:\\anaconda\\envs\\for_sota\\yolov13\\widerperson.yaml',
#     batch=1,
#     workers=0,
#     )
