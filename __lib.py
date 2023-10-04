import glob
import os
import math
import sys
import time
import random
import time

import matplotlib.pyplot as plt
from tqdm import tqdm # progress bar
import cv2
import pandas as pd
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
#
import warnings
warnings.filterwarnings("ignore")
#

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir,transform=None,target_transform =None ):
        self.imgsz = 320

        self.transform = transform

        self.transform = transform
        self.target_transform = target_transform

        self.image_files = []
        self.annotation_files = []
        self.image_files , self.annotation_files = CustomDataset.find_pair_files(root_dir,"jpg","txt")

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_in = CustomDataset.read_image(image_file)
        image_tensor , width_new , height_new = CustomDataset.image_process(image_in,self.imgsz)
        class_ids , boxes = CustomDataset.read_annotation_yolov8_idxyxy(self.annotation_files[index],width_new,height_new)

        boxes = torch.tensor(np.array(boxes, dtype=np.float32), dtype=torch.float32)
        labels = torch.tensor(np.array(class_ids, dtype=np.int64), dtype=torch.int64)

        target = {
            "boxes": boxes.cpu(),
            "labels": labels.cpu()
        }
        return image_tensor.cpu(), target

    def __len__(self):
        return len(self.image_files)
    
    def image_process(image_in,imgsz):
        [height, width, _] = image_in.shape
        if height > width:
            image_rect = np.zeros((height, height, 3), np.uint8)
            image_rect[0:height, 0:width] = image_in
            scale = height / imgsz
        else:
            image_rect = np.zeros((width, width, 3), np.uint8)
            image_rect[0:height, 0:width] = image_in
            scale = width / imgsz

        width_new  = int(width / scale)
        height_new = int(height / scale)

        image_rect = cv2.resize(image_rect, (imgsz,imgsz))
        image_normalize = image_rect / 255.0
        image_3nn = image_normalize.transpose((2, 0, 1))

        image_3nn = image_3nn.astype(np.float32)
        image_3nn = torch.FloatTensor(image_3nn)
        return image_3nn , width_new , height_new

    def read_image(path_image:str):
        image = cv2.imread(path_image)
        if image is None or image.size == 0:
            raise IOError('failed to load ' + str(path_image))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def read_annotation_yolov8_idxyxy(annotation_path:str,image_width:int, image_height:int):
        with open(annotation_path, 'r') as label_file:
            label_data = label_file.readlines()
        _boxes = []
        _class_ids = []
        for line in label_data:
            r= line.split(" ")

            # One number is added for the background category
            id_class = int(r[0])+1
            xc = float(r[1]) * image_width
            yc = float(r[2]) * image_height
            w =  float(r[3]) * image_width
            h =  float(r[4]) * image_height

            x0 = int( xc - (w/2) )
            y0 = int( yc - (h/2) )
            x1 = int( xc + (w/2) )
            y1 = int( yc + (h/2) )

            _class_ids.append( id_class)
            _boxes.append( [x0,y0,x1,y1])

        return _class_ids , _boxes

    def get_all_files(directory_path:str,extension:str):
        if os.path.isdir(directory_path):
            _files = []
            for root, dirs, files in os.walk(directory_path):
                _files.extend(glob.glob(os.path.join(root, f"*.{extension}")))
            return _files
        else:
            return []

    def find_pair_files(directory_path:str,extension1:str,extension2:str):
        files1 = CustomDataset.get_all_files(directory_path,extension1)
        files2 = CustomDataset.get_all_files(directory_path,extension2)

        list_pair1 = []
        list_pair2 = []

        for f1 in files1:
            for f2 in files2:
                basename_f1 = os.path.splitext(os.path.basename(f1))[0]
                basename_f2 = os.path.splitext(os.path.basename(f2))[0]
                if basename_f1 == basename_f2:
                    list_pair1.append(f1)
                    list_pair2.append(f2)
                    break
        print(f"number files pair in folder --: {directory_path} : --is-- :{len(list_pair1)} ")
        return list_pair1,list_pair2

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class TimeExe:
  def __init__(self,TAG):
    self.TAG = TAG
    pass
  def begin(self):
    self.start_time = time.time()

  def end(self):
    end_time = time.time()
    elapsed_time = end_time - self.start_time
    print(self.TAG," : Elapsed time : ",elapsed_time,"seconds")
    return elapsed_time

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def predict(image, model,imgsz, detection_threshold):
    image_tensor , width_new , height_new = CustomDataset.image_process(image,imgsz)
    image_tensor = [image_tensor.cpu()]
    scale = image.shape[0]/height_new

    outputs = model(image_tensor)

    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    pred_ids = outputs[0]['labels'].detach().cpu().numpy()

    mask = pred_scores >= detection_threshold

    pred_scores = pred_scores[mask].astype(np.float32)
    pred_bboxes = pred_bboxes[mask].astype(np.int32)
    pred_ids    = pred_ids[mask].astype(np.int32)

    boxes_out = []
    for box in pred_bboxes:
            x1 = round(box[0] * scale)
            y1 = round(box[1] * scale)
            x2 = round(box[2] * scale)
            y2 = round(box[3] * scale)
            boxes_out.append([int(x1),int(y1),int(x2),int(y2)])

    return boxes_out , pred_ids , pred_scores

def draw_boxes(image , boxes , ids , classes_names , colors , scors):
    for i, box in enumerate(boxes):
        id  = ids[i]
        if id == 0:
            break
        color = colors[id]
        name = classes_names[id]
        scor = scors[i]
        x0  = int(box[0])
        y0  = int(box[1])
        x1  = int(box[2])
        y1  = int(box[3])
               
        cv2.rectangle( image,(x0, y0), (x1, y1),  color, 2  ) 
        cv2.putText(image,"{} {:.0f} %".format(name,scor * 100),(x0+5,y0 + 20),cv2.FONT_HERSHEY_COMPLEX,(1),color,2)

def test_tensor(images, targets,names):
    img_int = imageTensor2imageRGB(images)
    boxes = torch.tensor(targets["boxes"].cpu(), dtype=torch.float32).numpy().tolist()
    labels = torch.tensor(targets["labels"].cpu(), dtype=torch.int64).numpy().tolist()
    for i , box in enumerate(boxes) :
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        id  = labels[i]
               
        cv2.rectangle(img_int, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(img_int,"{}".format(names[id]),(x0,y0+20),cv2.FONT_HERSHEY_COMPLEX,(1),(0, 255, 255),1)

    return img_int

def imageTensor2imageRGB(images):
    img_int = torch.tensor(images.cpu() * 255, dtype=torch.uint8).numpy()
    img_int = img_int.transpose((1, 2, 0))
    img_int = cv2.cvtColor(img_int, cv2.COLOR_BGR2RGB)
    return img_int
  
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def plot_cost(_costs , key):
    title = f"Epoch vs {key}"
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(np.arange(len(_costs)), _costs, 'r')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(key)
    ax.set_title(title)
    # plt.show()
    plt.savefig(title)

def plot_csv(path):
    map_dict_all = pd.read_csv(path)
    df_map = pd.DataFrame(map_dict_all)
    keys = df_map.keys()
    for key in keys:
        try:
            plot_cost(df_map[key],key)
        except:
            pass

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def validate_model(model,loader,device):
  model.to(device)
  model.eval()
  # #https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
  metric = MeanAveragePrecision(iou_type="bbox")
  with torch.no_grad():
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

        preds = model(images)

        metric.update(preds, targets)
  # from pprint import pprint
  map_result = metric.compute()
  keys = map_result.keys()
  dict_matric = {}
  for key in keys:
    dict_matric[key] = map_result[key].numpy()
  # fig_, ax_  = metric.plot()

  return dict_matric

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def save_list_dict(_data,_path):
  timestr = time.strftime("%Y%m%d-%H%M%S")
  random_str = str(random.randint(0, 999999999)).zfill(10)
  _path_gen = f"{_path}_{timestr}_{random_str}.csv"
  print("save :",_path_gen)
  losses_dict_all = pd.DataFrame(_data)
  losses_dict_all.to_csv(_path_gen,index=False)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def train_one_epoch(model, optimizer, data_loader, device, epoch , scaler=None):
    model.to(device)
    model.train()

    _lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        _lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    all_losses = []
    all_losses_dict = []
    print(f" \n -------------------- epoch {epoch}")
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()
        
        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if _lr_scheduler is not None:
              _lr_scheduler.step()

    all_losses_dict = pd.DataFrame(all_losses_dict)
    dict_r = all_losses_dict.mean(axis=0).to_dict()
    dict_r["lr"] = optimizer.param_groups[0]["lr"]
    return dict_r

