import cv2
from math import *
import numpy as np

def piano_bound_decision(frame):
    piano_img = cv2.imread('image/cat.jpg')
    # 使用近邻算法 nearest neighbour
    frame_l, frame_h, _ = frame.shape
    piano_l, piano_h, _ = piano_img.shape
    lowest_error = [-1, 0, 0] # 用以存储最小误差的起始点
    # 起始点位置
    for i in range(0, frame_l - piano_l + 2):
        for j in range(0, frame_h - piano_h + 2):
            init_l, init_h = i, j
            test_error = 0
            for k1 in range(init_l, init_l + piano_l):
                for k2 in range(init_h, init_h + piano_h):
                    # L2范式距离
                    test_error =test_error + sqrt((frame[k1][k2][0] - piano_img[k1][k2][0]) ** 2
                                                       + (frame[k1][k2][1] - piano_img[k1][k2][1]) ** 2
                                                       + (frame[k1][k2][2] - piano_img[k1][k2][2]) ** 2)
            if lowest_error[0]==-1:
                lowest_error = [test_error, init_l, init_h]
            elif test_error < lowest_error[0]:
                lowest_error = [test_error, init_l, init_h]
    # 绘制框图 draw_box
    _, init_x, init_y = lowest_error
    frame_process1 = cv2.rectangle(frame, (init_x, init_y), (init_x + piano_l - 1, init_y + piano_h -1), (0, 0, 255), 3)
    frame_list_param = [frame, frame_process1, init_x, init_y, init_x + piano_l - 1, init_y + piano_h -1]
    return frame_list_param

def split_piano(frame):
    piano_national_xy = [200, 200, 2, 1] # 国标数据：长length，高height，白键键程，黑键键程
    x1, y1, x2, y2 = frame[2 : ]
    ration = (x2 - x1) / piano_national_xy[0]  # 计算比例，保证摄像头水平
    xy_split_pose = [piano_national_xy[2] * ration, piano_national_xy[3] * ration]
    return xy_split_pose

def r_feature_extraction(frame, xy_split):
    commodity_img = frame[0]
    x1, y1, x2, y2 = frame[2:]
    x_white_sp, x_black_sp = xy_split  # 白键键程，黑键键程
    x1_param = []
    x2_param = []
    x1_start = x1 - x_black_sp
    for i in range(88):
        if (i % 2 == 0):  # 单个琴键起止位置
            x1_start = x1_start + x_black_sp
            x1_end = x1_start + x_white_sp
        else:
            x1_start = x1_start + x_white_sp
            x1_end = x1_start + x_black_sp
        r = []
        for s in range(x1_start, x1_end + 1):
            for k in range(y1, y2 + 1):
                # 确认红线位置
                if (commodity_img[s][k][0] > 200 and (commodity_img[s][k][1] + commodity_img[s][k][2]) / 2 < 50):
                    r.append(commodity_img[s][k][0])
        # 计算特征值，r通道的均值、方差
        x1_param.append(np.mean(r))
        x2_param.append(np.var(r, ddof = 1))
    return x1_param, x2_param

def anomaly_detection_Gauss(x1_param, x2_param):
    mu =[np.mean(x1_param), np.mean(x2_param)]
    sigma = [np.var(x1_param), np.var(x2_param)]
    model_con = [mu, sigma]
    return model_con

def anomaly_detection_prediction(model, x):
    p = 1
    mu, sigma = model
    for j in range(2):
        p = p * (1 / (sqrt(2 * pi) * sigma[j])) * (e ** (- (x[j] - mu[j]) ** 2 / (2 * (sigma[j] ** 2))))
    return p

if __name__ == "__main__":
    # 处理视屏
    video = cv2.VideoCapture('image/test.mp4')  # 捕捉摄像头，用数字来控制不同的设备，例如外接通常为400
    # 检查是否正确打开
    if video.isOpened():
        open, frame = video.read()
    else:
        open = False

    pd = True
    while open:
        ret, frame = video.read()
        if frame is None:
            break
        if ret == True:
            # 在保证摄像头水平后，执行一次调整与高斯模型构建
            if pd == True:
                pd = False
                # 寻找88个琴键的位置，决策边界
                frame_piano = piano_bound_decision(frame)
                # 根据国标详细划分88个琴键
                xy_split = split_piano(frame_piano)
                # r通道均值、方差的特征提取
                x1_param, x2_param = r_feature_extraction(frame_piano, xy_split)
                # 构建正常情况下的高斯模型
                model = anomaly_detection_Gauss(x1_param, x2_param)

            # 红外畸变时，再对88琴键提取r通道的均值、方差特征值
            frame_piano[0] = frame
            x1, x2 = r_feature_extraction(frame_piano, xy_split)
            # 使用建立好的高斯模型进行异常检测
            threhold = 0.5 # 阈值
            tot = 0
            for i in range(88):
                x = [x1[i], x2[i]]
                p = anomaly_detection_prediction(model, x)
                if p > threhold:
                    print(i)
            #cv2.imshow('result', gray)
            if cv2.waitKey(10) & 0xFF == 27:  # waitkey代表视屏播放时间，0xFF==27相当于键盘上的Esc键退出
                break
    video.release()
    cv2.destroyAllWindows()