import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import CascadeRCNN, Cascade_ROIPredictor
from backbone import resnet50_fpn_backbone
from tools import draw_objs
import numpy as np


# image_path = r'E:\DataSet1\multi_objec_tracking\data_tracking_image_2\training\image_02\0000'  # 图片文件夹路径
image_path = r'./data/object/image_2'  # 图片文件夹路径
# image_save_path = r'E:\DataSet1\save_image'  # 检测后图片保存的路径
image_save_path = r'./data/object/detection'  # 检测后图片保存的路径


def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = CascadeRCNN(backbone=backbone, num_classes=21)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = Cascade_ROIPredictor(in_features, num_classes)
    model.roi_heads.ready()  # 初始化各个 stage 网络

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=9)

    # load train weights
    weights_path = "./save_weights/Cascade-resNet50-fpn-model-KITTI-14.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './kitti_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])

    img_list = []  # 储存 Image 对象转化的 Tensor
    original_img_list = []  # 储存 Image 对象
    img_filepath_list = os.listdir(image_path)
    for val in img_filepath_list:
        img = Image.open(os.path.join(image_path, val))
        original_img_list.append(img)
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        img_list.append(img)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img_list[0].shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)
        boxes_list = {}
        for i in range(len(img_list)):
            predictions = model(img_list[i].to(device))[0]
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            #  获取目标检测框的信息
            # idxs = np.greater(predict_scores, 0.5)
            # boxes = predict_boxes[idxs]
            # classes = predict_classes[idxs]
            #
            # boxes_list[i] = {'boxes': boxes.tolist(), 'classes': classes.tolist()}

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            plot_img = draw_objs(original_img_list[i],
                                 predict_boxes,
                                 predict_classes,
                                 predict_scores,
                                 category_index=category_index,
                                 box_thresh=0.5,
                                 line_thickness=3,
                                 font='arial.ttf',
                                 font_size=20)

            # # 保存预测的图片结果
            plot_img.save(f"{image_save_path}/{i:06d}.png")
        # 存储目标检测框的信息
        # with open('./image.json', 'w') as fw:
        #     json.dump(boxes_list, fw)


if __name__ == '__main__':
    main()
