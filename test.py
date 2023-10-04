import os
import cv2
import numpy as np
import torch
from __lib import CustomDataset ,predict,draw_boxes,TimeExe
# from google.colab.patches import cv2_imshow
def cv2_imshow(img:np.ndarray):
    cv2.imshow("img",img)
    cv2.waitKey()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# classes_names = ["background","0","1","2","3","4","5","6","7","8","9"]
classes_names = ["background","car","plate"]
COLORS = np.random.uniform(0, 255, size=(len(classes_names), 3))

# # # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 0.7
# dataset_path = "data_car"
# model_path =  "model_FRCNN+CAR.pth" 
# # # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 0.11
dataset_path = "data_car"
model_path =  "model_SSD+CAR.pth" 
# # # # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 0.7
# dataset_path = "data_ocr"
# model_path =  "model_FRCNN+OCR.pth" 
# # # # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 0.11
# dataset_path = "data_ocr"
# model_path =  "model_SSD+OCR.pth" 


out_path = os.path.join("out",dataset_path,model_path)
print(out_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

model_loaded = torch.load(model_path , map_location=torch.device('cpu') )
images_path = CustomDataset.get_all_files(dataset_path,"jpg")
t = TimeExe("TIME")

for i, p in enumerate(images_path):
    img = CustomDataset.read_image(p)

    t.begin()
    boxes, labels ,scors = predict(img,model_loaded,320,0.3)
    t.end()

    draw_boxes(img,boxes,labels,classes_names,COLORS,scors)
    cv2.imwrite(os.path.join(out_path , os.path.basename(p)),img)

    # cv2_imshow(img)
    if i >= 20:
        break

