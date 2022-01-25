import glob
import os
import cv2
import numpy as np
from skimage import transform


def read_video(video_path):
    # 等间隔采样法
    need_number = 8
    images = []
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7) - 1
    step = frames_num // need_number
    print("总帧数: ", frames_num, "步长:", step)
    i = 0
    frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        i = i+1
        if frames == need_number:
            cap.release()
            break
        if (i % step == 0):
            frames = frames + 1
            frame = transform.resize(frame, (150, 80))
            images.append(frame)
    return np.asarray(images, np.float32)

def get_data(class_paths):
    """
    数据标签
    biking 0
    diving 1
    """
    # 获得数据集 
    train = []
    label = []
    for index, paths in enumerate(class_paths):
        class_images = []
        for path in paths:
            img = read_video(path)
            class_images.append(img)
        class_labels = np.ones(len(paths)) * index
        train.extend(class_images)
        label.extend(class_labels)
    return np.array(train),np.array(label)

if __name__ == "__main__":
    print("开始收集训练集数据")
    biking = glob.glob("videos/train/biking/*/*")
    diving = glob.glob("videos/train/diving/*/*")
    images, labels = get_data([biking, diving])
    np.savez('data/train.npz', images=images, labels=labels)

    print("开始收集测试集数据")
    biking = glob.glob("videos/test/biking/*/*")
    diving = glob.glob("videos/test/diving/*/*")
    images, labels = get_data([biking, diving])
    np.savez('data/test.npz', images=images, labels=labels)
