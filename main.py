import os
from ex4_utils import *
import cv2


def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd)
    plt.colorbar()
    plt.show()


def main():
    # Read images
    L = cv2.imread('pair0-L.png', cv2.COLOR_BGR2GRAY)
    R = cv2.imread('pair0-R.png', cv2.COLOR_BGR2GRAY)

    # Display depth SSD
    min_r, max_r = 0, 5
    displayDepthImage(L, R, (min_r, max_r), method=disparitySSD)

    # ~Display depth NC~
    displayDepthImage(L, R, (min_r, max_r), method=disparityNC)
    # L = cv2.imread('pair1-D_L.png')
    # R = cv2.imread('pair1-D_R.png')
    # displayDepthImage(L, R, (50, 150), method=disparitySSD)

    # Homography
    src = np.array([[279, 552],
                    [372, 559],
                    [362, 472],
                    [277, 469]])
    dst = np.array([[24, 566],
                    [114, 552],
                    [106, 474],
                    [19, 481]])
    h, error = computeHomography(src, dst)
    print("homography:\n", h)
    print("error:", error)

    # warping
    dst = cv2.imread('billBoard.jpg', cv2.COLOR_BGR2RGB)
    src = cv2.imread('car.jpg', cv2.COLOR_BGR2RGB)
    warpImag(src, dst)


if __name__ == '__main__':
    print("id: 208583476")
    main()