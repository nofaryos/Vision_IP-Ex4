import sys

import numpy as np
import matplotlib.pyplot as plt


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """

    disparityMap = np.zeros_like(img_l)
    minR = disp_range[0]
    maxR = disp_range[1]
    rowsL, colsL = img_l.shape
    rowsR, colsR = img_r.shape

    for r in range(k_size, rowsL - k_size):
        for c in range(k_size, colsL - k_size):
            win = img_l[r - k_size: 1 + k_size + r, c - k_size: 1 + k_size + c]
            maxSSD = 10000
            for j in range(max(c - maxR, k_size), min(maxR + c, colsL - k_size)):
                if abs(c - j) < minR:
                    continue
                compareWin = img_r[r - k_size: 1 + k_size + r, j - k_size: 1 + k_size + j]
                ssd = ((win - compareWin) ** 2).sum()
                if ssd < maxSSD:
                    maxSSD = ssd
                    disparityMap[r, c] = abs(j - c)
    disparityMap *= (255 // maxR)
    return disparityMap


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)
    return: Disparity map, disp_map.shape = Left.shape
    """
    disparityMap = np.zeros_like(img_l)
    rowsL, colsL = img_l.shape
    rowsR, colsR = img_r.shape
    for row in range(k_size, rowsL - k_size):
        for col in range(k_size, colsL - k_size):
            maxNC = -1000000
            disparity = 0
            winL = img_l[row - k_size: 1 + k_size + row, col - k_size: 1 + k_size + col]
            normL = np.linalg.norm(winL)
            sumL = np.sqrt(np.sum((winL - normL) ** 2))

            for j in range(disp_range[0], disp_range[1]):
                if (j + col - k_size >= 0) and (1 + k_size + col + j < colsL) and \
                        (col - j - k_size >= 0) and (1 + k_size + col - j < colsL):

                    # move left:
                    winR = img_r[row - k_size: 1 + k_size + row, col - j - k_size: 1 + k_size + col - j]
                    normR = np.linalg.norm(winR)
                    sumR = np.sqrt(np.sum((winR - normR) ** 2))
                    NNC = np.sum((winL - normL) * (winR - normR)) / (sumL * sumR)
                    if maxNC < NNC:
                        maxNC = NNC
                        disparity = j
                    # move right: 
                    winR = img_r[row - k_size: 1 + k_size + row, col + j - k_size: 1 + k_size + col + j]
                    normR = np.linalg.norm(winR)
                    sumR = np.sqrt(np.sum((winR - normR) ** 2))
                    NNC = ((winL - normL) * (winR - normR)).sum() / (sumL * sumR)
                    if maxNC < NNC:
                        maxNC = NNC
                        disparity = j
            disparityMap[row, col] = disparity
    disparityMap *= (255 // disp_range[1])
    return disparityMap


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))
    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]
    return: (Homography matrix shape:[3,3], Homography error)
    """
    n = 0
    a = np.zeros((8, 9))
    lenSrc = len(src_pnt)
    for i in range(lenSrc):
        xSrc = src_pnt[i][0]
        xDst = dst_pnt[i][0]
        ySrc = src_pnt[i][1]
        yDst = dst_pnt[i][1]
        a[n + 1] = [0, 0, 0, xSrc, ySrc, 1, -yDst * xSrc, -yDst * ySrc, -yDst]
        a[n] = [xSrc, ySrc, 1, 0, 0, 0, -xDst * xSrc, -xDst * ySrc, -xDst]
        n += 2
    a = np.asarray(a)
    u, d, v = np.linalg.svd(a)
    h = v[-1, :] / v[-1, -1]
    H = h.reshape(3, 3)

    # padding src_pnt
    col = [[1], [1], [1], [1]]
    paddedSrc = np.append(src_pnt, col, axis=1)
    old = H.dot(paddedSrc.T).T

    # calculate the error:
    new = np.zeros((4, 2))
    for i in range(4):
        for j in range(2):
            new[i][j] = old[i][j] / old[i][2]
    error = np.sqrt(np.square(new - dst_pnt).mean())

    return H, error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.
       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.
       output:
        None.
    """

    dstP = []
    fig = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dstP.append([x, y])

        if len(dstP) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dstP = np.array(dstP)

    # The corners of the image
    point_1 = [0, 0]
    point_2 = [0, src_img.shape[1] - 1]
    point_3 = [src_img.shape[0] - 1, src_img.shape[1] - 1]
    point_4 = [src_img.shape[0] - 1, 0]

    srcP = np.array([point_1, point_2, point_3, point_4])

    h, e = computeHomography(dstP, srcP)
    for i in range(dst_img.shape[0] - 1):
        for j in range(dst_img.shape[1] - 1):
            newXY = h.dot(np.array([[i], [j], [1]]))
            norm = h[2, :].dot(np.array([[i], [j], [1]]))
            newXY /= norm

            if 0 < newXY[0] < src_img.shape[0] and 0 < newXY[1] < src_img.shape[1]:
                dst_img[j][i][:] = src_img[int(newXY[0]), int(newXY[1]), :]

    plt.matshow(dst_img)
    plt.colorbar()
    plt.show()
