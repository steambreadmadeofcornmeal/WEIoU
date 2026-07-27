# WEIoU

## Overview

The results summary table is `results.xlsx`.  
Training parameters, results, and model weights are saved in `weiou_for_yolo26/results` and `weiou_for_yolov13/results`.  
Gradient data and plotting programs are saved in `weiou_for_yolov13/loss_grad`.

---

## Quick Start

### 1. Install Dependencies

Install YOLOv13 and Ultralytics:

- YOLOv13: [https://github.com/iMoonLab/yolov13](https://github.com/iMoonLab/yolov13)
- Ultralytics: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

### 2. Clone This Repository

```bash
git clone https://github.com/steambreadmadeofcornmeal/WEIoU.git
conda create -n weiou python=3.11
conda activate weiou
```

Replace the original files with the modified ones:

- Use `weiou_for_yolov13/loss.py` and `weiou_for_yolov13/metrics.py` to replace `yolov13/ultralytics/utils/loss.py` and `yolov13/ultralytics/utils/metrics.py`.
- Use `weiou_for_yolo26/loss.py` and `weiou_for_yolo26/metrics.py` to replace `ultralytics/ultralytics/utils/loss.py` and `ultralytics/ultralytics/utils/metrics.py`.

### 3. Prepare Datasets

Download the CrowdHuman and WiderPerson datasets:

- CrowdHuman: [https://www.crowdhuman.org/](https://www.crowdhuman.org/)
- WiderPerson: [http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/)

Or download the YOLO-format version we prepared (Baidu Netdisk):  
[https://pan.baidu.com/s/1xVlCOqUEPgZsjMk_whU6YQ?pwd=bpkb](https://pan.baidu.com/s/1xVlCOqUEPgZsjMk_whU6YQ?pwd=bpkb)

Modify the `path` field in `crowdhuman.yaml` and `widerperson.yaml` to point to your local dataset directory.

### 4. Validation

```python
from ultralytics import YOLO

model = YOLO('path/to/local/model/weight.pt')
metrics = model.val('path/to/crowdhuman.yaml/or/widerperson.yaml')
```

### 5. Training

```python
from ultralytics import YOLO

model = YOLO('path/to/coco_200epoch_pretrain.pt')

results = model.train(
    data='path/to/crowdhuman.yaml/or/widerperson.yaml',
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
    # resume=True,
    lr0=0.01,
    lrf=0.00001,
    dfl=0,
    seed=4096,
    # box=6,
    # cls=1,
)
```

### 6. Prediction

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("path/to/local/model/weight.pt")
    results = model.predict(
        source="path/to/local/image/folder",
        save=True,
        # boxes=False,
        # visualize=True,
        # show_conf=False,
    )
```
