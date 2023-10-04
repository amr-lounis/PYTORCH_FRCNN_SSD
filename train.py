import os
import functools
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# from google.colab.patches import cv2_imshow
def cv2_imshow(img:np.ndarray):
    cv2.imshow("img",img)
    cv2.waitKey(0)

from __lib import CustomDataset , TimeExe , train_one_epoch , validate_model ,save_list_dict ,plot_cost ,test_tensor

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
model_name = "SSD"
classes_names = ["background","0","1","2","3","4","5","6","7","8","9"]
# classes_names = ["background","car","plate"]
imgsz = 320
batch_size = 16
num_epochs = 50

dataset_path = "data"
model_saved ="model.pth"

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("device:",device)

COLORS = np.random.uniform(0, 255, size=(len(classes_names), 3))
num_classes = len(classes_names)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
sub_folder_train = "train"
sub_folder_valid = "valid"
sub_folder_test = "test"

dataset_train = CustomDataset(os.path.join(dataset_path,sub_folder_train))
dataset_test = CustomDataset(os.path.join(dataset_path,sub_folder_test))
dataset_validate = CustomDataset(os.path.join(dataset_path,sub_folder_valid))

print( len(dataset_train) ," --:-- ", len(dataset_test) ," --:-- ", len(dataset_validate) )
image,target = dataset_train[0]
image_out = test_tensor(image,target,classes_names)
cv2_imshow(image_out)

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def collate_fn(batch):
    return tuple(zip(*batch))

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)

loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,  collate_fn=collate_fn)

loader_valid = DataLoader(dataset_validate, batch_size=batch_size, shuffle=False,  collate_fn=collate_fn)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if model_name == "FRCNN":
    print("FRCNN")
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    print("created new model")

elif model_name == "SSD":
    print("SSD")
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    in_channels = torchvision.models.detection._utils.retrieve_out_channels(model.backbone, (imgsz, imgsz))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)

    model.head.classification_head = torchvision.models.detection.ssdlite.SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.05, momentum=0.9, weight_decay=0.0005)
# learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
t_all = TimeExe("t_all")
t_all.begin()

losses_dict_all = []
map_dict_all = []


for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    losses_dict_this = train_one_epoch(model, optimizer, loader_train, device, epoch )
    # print(losses_dict_this)
    losses_dict_all.append(losses_dict_this)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    print("---------- validate ")
    dict_valid = validate_model(model, loader_valid, device=device)
    print(dict_valid)
    map_dict_all.append(dict_valid)

    t_all.end()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
torch.save(model, model_saved)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
save_list_dict(losses_dict_all,"loss")
print(losses_dict_all)

save_list_dict(map_dict_all,"map")
print(map_dict_all)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
df_losses = pd.DataFrame(losses_dict_all)
keys = df_losses.keys()
for key in keys:
  plot_cost(df_losses[key],key)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df_map = pd.DataFrame(map_dict_all)
keys = df_map.keys()
for key in keys:
  try:
    plot_cost(df_map[key],key)
  except:
    pass
  
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++