import numpy as np
import cv2

def cv_show(img):
    cv2.imshow('test', img)
    cv2.waitKey(0)  # 窗口停止时间，毫秒级，0表示任意键结束
    cv2.destroyAllWindows()

def r_mu(commodity_img, x):
    r = []
    x1, x2 = x
    h, r_1, r_2 = commodity_img.shape
    for s in range(x1, x2):
        if s >= r_1:
            continue
        for k in range(h):
            # 确认红线位置
            if commodity_img[k][s][2] > 170 and (commodity_img[k][s][1] + commodity_img[k][s][0]) / 2 < 50:
                r.append(k)
                # r.append(commodity_img[k][s][2])
    return np.mean(r)

def r_feature_extraction(piano_img, xysplit):
    mu = []
    x1 = - xysplit[1]
    for i in range(5):
        if i%2==0:
            x1 = x1 + xysplit[1]
            x2 = x1 + xysplit[0]
            mu.append(r_mu(piano_img, [x1, x2]))
            piano_img = cv2.rectangle(piano_img, (x1, 0), (x2, 300), (255, 0, 0), 1)
        else:
            x1 = x1 + xysplit[0]
            x2 = x1 + xysplit[1]
            mu.append(r_mu(piano_img, [x1, x2]))
            piano_img = cv2.rectangle(piano_img, (x1, 0), (x2, 300), (0, 255, 0), 1)
    x1 = x1 + (xysplit[0] - xysplit[1])
    for i in range(7):
        if i%2==0:
            x1 = x1 + xysplit[1]
            x2 = x1 + xysplit[0]
            mu.append(r_mu(piano_img, [x1, x2]))
            piano_img = cv2.rectangle(piano_img, (x1, 0), (x2, 300), (255, 0, 0), 1)
        else:
            x1 = x1 + xysplit[0]
            x2 = x1 + xysplit[1]
            mu.append(r_mu(piano_img, [x1, x2]))
            piano_img = cv2.rectangle(piano_img, (x1, 0), (x2, 300), (0, 255, 0), 1)
    # cv_show(piano_img)
    return piano_img, mu

if __name__ == "__main__":
    # 处理视屏
    video = cv2.VideoCapture('image/piano_video.MP4')  # 捕捉摄像头，用数字来控制不同的设备，例如外接通常为400
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
            frame = cv2.resize(frame, (900, 675))
            frame = frame[100:400, 250:570, :]
            if pd == True:
                pd = False
                xysplit = [30, 23]
                piano_frame, mu = r_feature_extraction(frame, xysplit)
                print("mu: ")
                print(mu)
            else:
                piano_frame, mu_1 = r_feature_extraction(frame, xysplit)
                print("mu_1: ")
                print(mu_1)
                threhold = 10
                # for i in range(len(mu_1)):
                #     if mu[i] - mu_1[i] > threhold:
                #         print(i)
                # cv_show(frame)
            cv2.imshow('result', piano_frame)
            if cv2.waitKey(10) & 0xFF == 27:  # waitkey代表视屏播放时间，0xFF==27相当于键盘上的Esc键退出
                break
    video.release()
    cv2.destroyAllWindows()