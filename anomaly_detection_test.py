import cv2
from math import *
import numpy as np

def cvshow(filename, img):
    cv2.imshow(filename, img)
    cv2.waitKey(0)  # 窗口停止时间，毫秒级，0表示任意键结束
    cv2.destroyAllWindows()

def r_feature_extraction(commodity_img, x1, x2):
    r = []
    h, r_1, r_2 = commodity_img.shape
    for s in range(x1, x2):
        for k in range(h):
            # 确认红线位置
            if (commodity_img[k][s][2] > 170 and (commodity_img[k][s][1] + commodity_img[k][s][0]) / 2 < 50):
                r.append(commodity_img[k][s][2])
    # print(r)
    xparam = [np.mean(r), np.var(r, ddof=1)]
    return xparam

def anomaly_detection_prediction(model, x):
    p = 1
    mu, sigma = model
    for j in range(2):
        p = p * (1 / (sqrt(2 * pi) * sigma[j])) * (e ** (- (x[j] - mu[j]) ** 2 / (2 * (sigma[j] ** 2))))
    return p

if __name__ == "__main__":
    piano_img = cv2.imread('image/piano_static.jpg')
    x1, y1, x2, y2 = [120, 100, 620, 500]
    #piano_img = cv2.rectangle(piano_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    piano_img = piano_img[100:500, 120:620, :]
    frame = [piano_img, piano_img, x1, y1, x2, y2]
    xysplit = [50, 30]
    # draw_box
    x1 = -30
    x_param = []
    for i in range(5):
        if i%2==0:
            x1 = x1 + xysplit[1]
            x2 = x1 + xysplit[0]
            x_param.append(r_feature_extraction(piano_img, x1, x2))
            #piano_img = cv2.rectangle(piano_img, (x1, 0), (x2, 400), (255, 0, 0), 1)
        else:
            x1 = x1 + xysplit[0]
            x2 = x1 + xysplit[1]
            x_param.append(r_feature_extraction(piano_img, x1, x2))
            #piano_img = cv2.rectangle(piano_img, (x1, 0), (x2, 400), (0, 255, 0), 1)
    x1 = x1 + 20
    for i in range(7):
        if i%2==0:
            x1 = x1 + xysplit[1]
            x2 = x1 + xysplit[0]
            x_param.append(r_feature_extraction(piano_img, x1, x2))
            #piano_img = cv2.rectangle(piano_img, (x1, 0), (x2, 400), (255, 0, 0), 1)
        else:
            x1 = x1 + xysplit[0]
            x2 = x1 + xysplit[1]
            x_param.append(r_feature_extraction(piano_img, x1, x2))
            #piano_img = cv2.rectangle(piano_img, (x1, 0), (x2, 400), (0, 255, 0), 1)
    print(x_param)
    x1_param = []
    x2_param = []
    for i in range(len(x_param)):
        x1_param.append(x_param[i][0])
        x2_param.append(x_param[i][1])
    mu = [np.mean(x1_param), np.mean(x2_param)]
    sigma = [np.var(x1_param), np.var(x2_param)]
    model = [mu, sigma]
    # print(model)
    test_img = cv2.imread('image/test.png')
    x2, x1, _ = test_img.shape
    x = r_feature_extraction(test_img, 0, x1)
    p = anomaly_detection_prediction(model, x)
    print(p)
    test_img = cv2.imread('image/test2.png')
    x2, x1, _ = test_img.shape
    x = r_feature_extraction(test_img, 0, x1)
    p = anomaly_detection_prediction(model, x)
    print(p)
    # cvshow('piano', piano_img)