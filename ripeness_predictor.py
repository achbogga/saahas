
# coding: utf-8

# In[6]:

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
get_ipython().magic(u'matplotlib inline')
def is_ripe(img, threshold=40): #input img as an nd array
    #img = cv2.imread('t1.png')
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #get rid of very bright and very dark regions
    delta=30
    lower_gray = np.array([delta, delta,delta])
    upper_gray = np.array([255-delta,255-delta,255-delta])
    # Threshold the image to get only selected
    mask = cv2.inRange(img, lower_gray, upper_gray)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    #Convert to HSV space
    HSV_img = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    hue = HSV_img[:, :, 0]

    #select maximum value of H component from histogram
    hist = cv2.calcHist([hue],[0],None,[256],[0,256])
    hist= hist[1:, :] #suppress black value
    elem = np.argmax(hist)
    hist_new = deepcopy(hist)
    hist_new[elem] = 0
    new_elem = np.argmax(hist_new)
    #elem = hist.argsort()[-3:][::-1][1]
    elem = new_elem
    print np.max(hist), np.argmax(hist)
    print np.max(hist_new), np.argmax(hist_new)

    tolerance=10
    lower_gray = np.array([elem-tolerance, 0,0])
    upper_gray = np.array([elem+tolerance,255,255])
    # Threshold the image to get only selected
    mask = cv2.inRange(HSV_img, lower_gray, upper_gray)
    # Bitwise-AND mask and original image
    res2 = cv2.bitwise_and(img,img, mask= mask)

    titles = ['Original Image', 'Selected Gray Values', 'Hue', 'Result']
    images = [img, res, hue, res2]
    for i in xrange(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    res3 = images[0]-images[3]
    res4 = cv2.cvtColor(res3, cv2.COLOR_HSV2BGR)
    res5 = cv2.cvtColor(res4, cv2.COLOR_BGR2GRAY)

    print res5.shape
    hist, bins = np.histogram(res5)
    #hist, bins = np.histogram(x, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    ripeness = (np.mean(res5))
    print ('ripeness: ')
    print(ripeness)
    plt.show()
    plt.imshow(res5)
    #res5[res5>(np.mean(res5)-5)]=0
    threshold = 40
    tomato_color_param = np.mean(res5)
    print (tomato_color_param)
    if (tomato_color_param) > threshold:
        print ('Tomatoes are ripe!\n')
        return True, ripeness
    else:
        print ('Tomatoes are not ripe!\n')
        return False, ripeness


# In[7]:

is_ripe(cv2.imread('t1.png'))


# In[ ]:



