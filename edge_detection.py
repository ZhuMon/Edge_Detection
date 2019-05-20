# file name: edge_detection.py
# usage:
#    python3 edge_detection.py [file] [threshold]
# ex: 
#    python3 edge_detection.py plane.png 50

import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np
import sys
import argparse
from scipy import signal

def rgb2gray(rgb, file_type):
    " Turn RGB image to gray image "
    if(file_type == "jpg"):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114 ])
    elif(file_type == "png"):
        return np.dot(rgb[...,:3], [0.299*255, 0.587*255, 0.114*255])

def histogram_equalization(original):
    " Compute histogram equalization"

    # make gray scale value to integer
    original = np.around(original).astype(int)

    r = [] # 儲存 value(0 - 255) 在原圖(original)的數量
    for i in range(0, 256):
        r.append((original == i).sum())


    MN = original.shape[0] * original.shape[1]
    # CDF 
    cdf_min = MN
    for i in range(1, 256):
        r[i] = r[i] + r[i-1]
        if r[i] != 0 and r[i] <= cdf_min:
            cdf_min = r[i]


    # turn gray value to histogram equalization
    cdf = np.vectorize(lambda x : r[x])
    
    rlt = np.around((cdf(original)-cdf_min)/(MN-cdf_min)*255)
    

    return rlt

def Laplacian_Mask(x, x1y, xy1):
    "Return 1 if x * x1y < 0 and x * xy1 < 0 and x-x1y > T and x-xy1 > T"
    if x * x1y < 0 and abs(x-x1y) > T :
        return 1
    elif x * xy1 < 0 and abs(x-xy1) > T:
        return 1
    else:
        return 0

def Laplacian(a):
    size = a.shape
    # x: row, y: column

    # xy1(x, y) = f(x, y+1) = a[x][y+1]
    b = np.zeros((1,size[0])).T # a all zero column
    xy1 = np.delete(a,0,1)
    xy1 = np.c_[xy1,b]

    # xy_1(x, y) = f(x, y-1) = a[x][y-1]
    xy_1 = np.delete(a, size[1]-1, 1)
    xy_1 = np.c_[b,xy_1]

    # x1y(x, y) = f(x+1, y) = a[x+1][y]
    c = np.zeros((1, size[1])) # a all zero row
    x1y = np.delete(a, 0, 0)
    x1y = np.r_[x1y, c]

    # x_1y(x, y) = f(x-1, y) = a[x-1][y]
    x_1y = np.delete(a, size[0]-1, 0)
    x_1y = np.r_[c, x_1y]

    # Δ^2f = Δ_x^2f + Δ_y^2f 
    lp_op = xy1 + xy_1 + x1y + x_1y - 4*a
    
    lp_x1y = np.delete(lp_op, 0, 0)
    lp_x1y = np.r_[lp_x1y, c]

    lp_xy1 = np.delete(lp_op, 0, 1)
    lp_xy1 = np.c_[lp_xy1, b]

    LM = np.vectorize(Laplacian_Mask)
    return LM(lp_op, lp_x1y, lp_xy1)

def Sobel(a):
    #x = np.array([[-1, 0, 1],[-2,0,2],[-1,0,1]])
    #y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    # ppt:
    x = np.array([[-1, -2, -1],[0,0,0],[1,2,1]])
    y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    # 沙爾算子
    # x = np.array([[3, 0, -3],[-10,0,10],[3,0,-3]])
    # y = np.array([[3,10,3],[0,0,0],[-3,-10,-3]])

    # convolution
    gx = signal.convolve2d(a, x, boundary='symm', mode='same')
    gy = signal.convolve2d(a, y, boundary='symm', mode='same')

    g = np.sqrt(np.square(gx)+np.square(gy))
    return g

def Gaussian_Blur(a, sigma):
    # 3*3 Gassian filter
    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = 1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))

    # Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    print(gaussian_kernel)
    g = signal.convolve2d(a, gaussian_kernel, boundary='symm', mode='same')
    return g

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "-image", required=True, help = "Input image", dest="img")
    ap.add_argument("-s", "-sobel", required=False, help = "Use Sobel operator",action='store_true', default=False)                                             
    ap.add_argument("-l", "-laplacian", required=False, help = "Use Laplacian operator, input threshold", dest="lp_T", default ="-1", type=float)
    ap.add_argument("-g", "-gaussion-blur", required=False, help = "Use Gaussion Blur", default=-1, type=float)
    ap.add_argument("-he", "-histogram-equalization", required=False, help = "Use Histogram Equalization", action='store_true', default=False)

    args = ap.parse_args()

    print(args)

    image = mpimg.imread(args.img)
    gray = rgb2gray(image, args.img[len(args.img)-3:len(args.img)])

    out = gray

    if args.g != -1:
        out = Gaussian_Blur(out, args.g)

    if args.he == True:
        out = histogram_equalization(out)

    if args.s == True and args.lp_T != -1:
        print("choose sobel or laplacian")
        sys.exit(1)
    elif args.s == True:
        out = Sobel(out)
    elif args.lp_T != -1:
        T = args.lp_T
        out = Laplacian(out)


    plt.imshow(out, cmap='Greys_r')
    plt.axis('off')
    plt.show()       
# T = float(sys.argv[2]) # threshold for Laplacian Mask
# inImg = mpimg.imread(image)  # read file 

# turn rgb to gray
# gray = rgb2gray(inImg, image[len(image)-3:len(image)]) 


# print gray level image
# plt.figure(figsize=(15,8))
# plt.subplot(2,2,1)
# plt.imshow(gray, cmap= 'Greys_r')
# plt.title("Gray level")
# plt.axis('off')

# plt.subplot(2,2,3)
# Gaussian blur
# gb = Gaussian_Blur(gray, float(sys.argv[2]))
# so = Sobel(gb)
# plt.imshow(so, cmap= 'Greys_r')
# plt.title("Gausssian_Blur")
# plt.axis('off')


# print gray level image that used histogram equalization
# plt.subplot(2,2,2)
# his = histogram_equalization(gray)
# so = Sobel(his)
# plt.imshow(so, cmap= 'Greys_r')
# plt.title("histogram equalization")
# plt.axis('off')


# plt.subplot(2,2,4)
# his = histogram_equalization(gb)
# so = Sobel(his)
# plt.imshow(so, cmap= 'Greys_r')
# plt.title("gb+he")
# plt.axis('off')
# plt.show()
# print the image use Laplacian operator without histogram equalization
# plt.subplot(3,2,3)
# la = Laplacian(gray)
# plt.imshow(la, cmap= 'Greys_r')
# plt.title("Laplacian without histogram equalization")
# plt.axis('off')

# print the image use Laplacian operator with histogram equalization
# plt.subplot(3,2,4)
# la = Laplacian(his)
# plt.imshow(la, cmap= 'Greys_r')
# plt.title("Laplacian with histogram equalization")
# plt.axis('off')

# print the image use Sobel operator without histogram equalization
# plt.subplot(2,2,3)
# so = Sobel(gray)
# plt.imshow(so, cmap= 'Greys_r')
# plt.title("Sobel no HE")
# plt.axis('off')

# # print the image use Sobel operator with histogram equalization
# plt.subplot(3,2,6)
# so = Sobel(his)
# plt.imshow(so, cmap= 'Greys_r')
# plt.title("Sobel HE")
# plt.axis('off')
# plt.show()
