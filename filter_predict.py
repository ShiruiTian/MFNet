import os
import time
import json

import torch
import torchvision
import numpy as np
from PIL import Image

from torchvision import transforms
from detector.detectorimage.network_files import Cascade_ROIPredictor, CascadeRCNN
from detector.detectorimage.backbone import resnet50_fpn_backbone
from detector.detectorimage.tools import draw_objs_new, draw_objs

from detector.detectorimage.filter_files import Kalman, UKF, GHFilter
from detector.detectorimage.filter_files import utils

# image_path = r'E:\DataSet1\multi_objec_tracking\data_tracking_image_2\training\image_02\0000'  # 图片文件夹路径
image_path = r'E:/code/Cascade_RCNN/data/object/image_2'  # 图片文件夹路径
# image_save_path = r'E:\DataSet1\save_image\0001'  # 检测后图片保存的路径
image_save_path = r'./data/object/filter/detection'  # 检测后图片保存的路径
label_save_path = r'E:/code/Cascade_RCNN/data/object/KITTI-10/det/det.txt'


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


def main(filter):  # filter: 滤波器
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
    data_transform = transforms.Compose([transforms.ToTensor()])

    img_name_list = os.listdir(image_path)  # 获取图像名
    state_list = []  # 目标状态信息，存 filter 对象
    model.eval()  # 进入验证模式
    frame_number = 1
    saved_objects = set()

    with torch.no_grad():
        for index, value in enumerate(img_name_list):
            orignal_img = Image.open(os.path.join(image_path, value))  # Image object
            img = data_transform(orignal_img)  # Image Tensor
            img = torch.unsqueeze(img, dim=0)

            predictions = model(img.to(device))[0]
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            # 筛选目标检测框
            idxs = np.greater(predict_scores, 0.5)
            predict_boxes = predict_boxes[idxs]
            predict_scores = predict_scores[idxs]
            predict_classes = predict_classes[idxs]

            # 预测
            for target in state_list:
                target.predict()

            # 关联
            mea_list = [(utils.box2meas(box1), class1, score1) for box1, class1, score1 in
                        zip(predict_boxes, predict_classes, predict_scores)]

            state_rem_list, mea_rem_list, match_list = filter.association(state_list, mea_list)

            # 状态未匹配上的，做删除处理
            state_del = []
            for idx in state_rem_list:
                if state_list[idx].delete():
                    state_del.append(state_list[idx])
            state_list = [val for val in state_list if val not in state_del]

            # 量测没匹配上的，作为新生目标进行航迹起始
            for idx in mea_rem_list:
                state_list.append(filter(utils.mea2state(mea_list[idx][0]), mea_list[idx][1], mea_list[idx][2]))

            # -----------------------------------------------可视化-----------------------------------

            # plot_img = draw_objs(orignal_img.copy(),
            #                      predict_boxes,
            #                      predict_classes,
            #                      predict_scores,
            #                      category_index=category_index,
            #                      box_thresh=0.5,
            #                      line_thickness=3,
            #                      font='arial.ttf',
            #                      font_size=20)
            # # # 保存预测的图片结果
            # plot_img.save(f"{image_save_path}/{index:06d}.png")

            # 显示所有的state到图像上
            predict_boxes = []
            predict_classes = []
            predict_scores = []


            for filter1 in state_list:
                predict_boxes.append(utils.state2box(filter1.X_posterior[:4]))
                predict_classes.append(filter1.class1)
                predict_scores.append(filter1.score1)

                # 保存结果到文本文件
                saved_objects_in_frame = set()
                '''
                with open("./data/object/filter/label.txt", "a") as file:
                    object_number = 1
                    for box, cls, score in zip(predict_boxes, predict_classes, predict_scores):
                        if (frame_number, object_number) not in saved_objects and (frame_number, object_number) not in saved_objects_in_frame:
                            saved_objects.add((frame_number, object_number))
                            saved_objects_in_frame.add(
                                (frame_number, object_number))  # Add the object to the set for the current frame
                            file.write(str(frame_number) + ",")
                            file.write(str(object_number) + ",")
                            file.write(str(int(cls)) + ",")
                            file.write(str(box[0]) + "," + str(box[1]) + "," + str(box[2] - box[0]) + "," + str(box[3] - box[1]) + ",")
                            file.write(str(round(score * 100, 1)) + ",")
                            file.write(str(-1) + "," + str(-1) + "," + str(-1) + "\n")
                        object_number += 1
'''
                with open(label_save_path, "a") as file:
                    object_number = 1
                    for box, cls, score in zip(predict_boxes, predict_classes, predict_scores):
                        if (frame_number, object_number) not in saved_objects and (frame_number, object_number) not in saved_objects_in_frame:
                            saved_objects.add((frame_number, object_number))
                            saved_objects_in_frame.add(
                                (frame_number, object_number))  # Add the object to the set for the current frame
                            file.write(str(frame_number) + ",")
                            file.write("-1" + ",")
                            # file.write(str(int(cls)) + ",")
                            # file.write(str(box[0]) + "," + str(box[1]) + "," + str(box[2] - box[0]) + "," + str(box[3] - box[1]) + ",")
                            file.write(
                                "{:.3f},{:.3f},{:.3f},{:.3f},".format(box[0], box[1], box[2] - box[0], box[3] - box[1]))
                            # file.write(str(round(score * 100, 1)) + ",")
                            file.write("{:.3f},".format(round(score * 100, 1)))
                            file.write(str(-1) + "," + str(-1) + "," + str(-1) + "\n")
                        object_number += 1


            # print(predict_boxes, predict_classes, predict_scores)
            # predict_boxes.save(f"{label_save_path}/{index:06d}.txt")


            orignal_img = draw_objs(orignal_img,
                                    np.array(predict_boxes),
                                    np.array(predict_classes),
                                    np.array(predict_scores),
                                    category_index=category_index,
                                    box_thresh=0.5,
                                    line_thickness=3,
                                    font='arial.ttf',
                                    font_size=20)
            orignal_img.save(f"{image_save_path}/{index:06d}.png")
            frame_number += 1


if __name__ == '__main__':
    main(Kalman)
