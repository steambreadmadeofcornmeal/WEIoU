from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO("O:\\anaconda\\envs\\for_sota\\yolov13\\runs\\detect\\train75\\weights\\best.pt")

    #标准预测
    results = model.predict(
        source="O:\\WiderPerson\\yolo\\test\\images1", 
        # source="O:\\crowdhuman\\images_test1",
        save=True, 
        # boxes=False,
        # visualize=True,
        show_conf=False,
                            )

    #整个文件夹逐图片预测
    # imgpath = "O:\\anaconda\\envs\\for_sota\\yolov13\\runs\\nanbengimgs\\crowdnanbenghuman"
    # imgpathorg = "O:\\WiderPerson\\yolo\\test\\images1"
    # imgpathorg = "O:\\crowdhuman\\images_test1"
    # imgpath = "O:\\anaconda\\envs\\for_sota\\yolov13\\runs\\nanbengimgs\\widernanbengperson"
    # imgs = os.listdir(imgpath)
    # for img in imgs:
    #     results = model.predict(
    #     # source="O:\\WiderPerson\\yolo\\test\\images1", 
    #     source=os.path.join(imgpathorg,img),
    #     save=True, 
    #     # boxes=False,
    #     visualize=True,
    #     show_labels=False,
    #     show_conf=False,
    #                         )