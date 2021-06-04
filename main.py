
import common
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from model.DBFace import DBFace

HAS_CUDA = torch.cuda.is_available()
print(f"HAS_CUDA = {HAS_CUDA}")


def nms(objs, iou=0.5):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(model, image, threshold=0.4, nms_iou=0.5):

    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    if HAS_CUDA:
        torch_image = torch_image.cuda()

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


def detect_image(model, file):

    image = common.imread(file)
    objs = detect(model, image)

    for obj in objs:
        # 增加对五官的检测
        frame = organ_detection(image, obj)
        common.drawbbox(image, obj)

    cv2.imshow('demo', image)
    cv2.waitKey()
    # common.imwrite("detect_result/" + common.file_name_no_suffix(file) + ".draw.jpg", image)


def image_demo():

    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    # detect_image(dbface, "datas/selfie.jpg")
    detect_image(dbface, "datas/1.jpg")


def camera_demo():

    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("model/dbface.pth")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    cap.set(cv2.CAP_PROP_FPS, 2)
    ok, frame = cap.read()


    while ok:
        # frame = cv2.resize(frame, (480, 640))

        objs = detect(dbface, frame)

        for obj in objs:
            # 增加对五官的检测
            frame = organ_detection(frame, obj)

            common.drawbbox(frame, obj)


        cv2.imshow("demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        ok, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()

# 载入人脸识别和眼睛识别的两个xml文件
eye_xml = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')# 效果更好
# eye_xml = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_xml = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

def organ_detection(frame, bbox):
    # 灰度处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 在人脸的基础上识别眼睛
    x, y, r, b = common.intv(bbox.box)
    w = r - x + 1
    h = b - y + 1
    face_gray = gray[y:y + h, x:x + w]
    face_color = frame[y:y + h, x:x + w]
    # 眼睛识别
    position = eye_xml.detectMultiScale(face_gray)
    # 绘制出识别到的眼睛
    for (p_x, p_y, p_w, p_h) in position:
        cv2.rectangle(face_color, (p_x, p_y), (p_x + p_w, p_y + p_h), (255, 255, 0), 2)  # 绘制眼睛方框

    # 嘴巴识别
    if bbox.haslandmark:
        x_r, y_r = common.intv(bbox.landmark[3][:2])
        x_l, y_l = common.intv(bbox.landmark[4][:2])
    mouth_w = x_l - x_r + 40
    mouth_h = x_l - x_r
    x = x_r - 20
    y = y_r - mouth_h // 2

    # 嘴部的检测区域
    cv2.rectangle(frame, (x, y), (x + mouth_w, y + mouth_h), (0, 50, 0), 2)

    face_gray = gray[y:y + mouth_h, x:x + mouth_w]
    face_color = frame[y:y + mouth_h, x:x + mouth_w]
    position = mouth_xml.detectMultiScale(face_gray)
    # 绘制出识别到的嘴巴
    for (p_x, p_y, p_w, p_h) in position:
        cv2.rectangle(face_color, (p_x, p_y), (p_x + p_w, p_y + p_h), (0, 255, 0), 2)  # 绘制嘴巴

    return frame

if __name__ == "__main__":
    # image_demo()
    camera_demo()




