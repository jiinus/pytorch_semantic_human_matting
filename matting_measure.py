import numpy as np
import scipy.ndimage
import cv2


def SAD_error(alpha_pre, alpha_truth):
    return np.mean(np.abs(alpha_pre - alpha_truth))


def MSE_error(alpha_pre, alpha_truth):
    return np.mean(np.power(alpha_pre - alpha_truth, 2))


def Gradient_error(alpha_pre, alpha_truth):
    Gradient_pre = scipy.ndimage.filters.gaussian_filter(
        alpha_pre, 1.4, order=1)
    Gradient_truth = scipy.ndimage.filters.gaussian_filter(
        alpha_truth, 1.4, order=1)
    return np.mean(np.power(Gradient_pre - Gradient_truth, 2))


def connectivity(alpha, omega, steps=100):
    dist = np.zeros((alpha.shape[0], alpha.shape[1], steps))
    l = np.ones(alpha.shape)
    connect = np.zeros(alpha.shape)

    for step in range(steps + 1):
        threshold = 1 - step / steps
        alpha_ = alpha.copy()
        alpha_[alpha_ < threshold] = 0
        alpha_[alpha_ >= threshold] = 1
        alpha_ = np.array(alpha_, np.uint8)

        _, labels = cv2.connectedComponents(alpha_, connectivity=4)

        label = labels[alpha_ == omega][0]
        alpha_[labels != label] = 0

        contours, _ = cv2.findContours(
            alpha_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                if alpha_[i][j] != 1:
                    dist[i][j][step] = cv2.pointPolygonTest(
                        contours, (i, j), True)
                elif step != 0:
                    if dist[i][j][step - 1] != 0:
                        l[i][j] = threshold

    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            zero_num = np.bincount(dist[i][j])[0]
            lamta = dist[i][j] / (step - zero_num)
            d = alpha[i][j] - l[i][j]
            if d >= 0.15:
                connect[i][j] = 1
            else:
                connect[i][j] = 1 - lamta * d

    return connect


def Connectivity_error(alpha_pre, alpha_truth):
    alpha_pre_ = alpha_pre.copy()
    alpha_truth_ = alpha_truth.copy()

    alpha_pre_[alpha_pre_ != alpha_truth_] = 0
    alpha_pre_[alpha_pre_ != 1] = 0
    omega = np.array(alpha_pre_, np.uint8)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        omega, connectivity=4)

    for row in stats:
        if alpha_pre_[row[0]][row[1]] == 0:
            row[-1] = 0

    max_label = np.argmax(stats, axis=0)[-1]
    omega[labels != max_label] = 0

    con_pre = connectivity(alpha_pre, omega)
    con_truth = connectivity(alpha_truth, omega)

    return np.sum(np.power(con_pre-con_truth, 1))


def matting_measure(alpha_pres, alpha_truthes):
    SAD = 0
    MSE = 0
    Gradient = 0
    Connectivity = 0

    num = alpha_pres.shape[0]

    for alpha_pre, alpha_truth in zip(alpha_pres, alpha_truthes):
        SAD += SAD_error(alpha_pre, alpha_truth)
        MSE += MSE_error(alpha_pre, alpha_truth)
        Gradient += Gradient_error(alpha_pre, alpha_truth)
        #Connectivity += Connectivity_error(alpha_pre, alpha_truth)

    return SAD/num, MSE/num, Gradient/num, Connectivity/num
