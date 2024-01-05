import math
import os
from tkinter import filedialog

import cv2
import numpy as np
from scipy.spatial import KDTree


def scalemax(img,maxallow):
    h,w = img.shape[:2]
    maxdim = max([h,w])
    scale = maxallow/maxdim
    resized = cv2.resize(img, (int(round(w*scale)), int(round(h*scale))))
    return resized

def selectcolors(img, maxlen = 1000, output = 'colors'):
    resized = scalemax(img,maxlen)
    landslide_points = []
    mask_points = []
    window_name = 'select points'

    def mouse_callback(event, x, y,flags,params):
        nonlocal updated, circolor,counter
        if event == cv2.EVENT_LBUTTONDOWN:
            landslide_points.append((x, y))
            updated = True
            circolor =[0,255,0]
            counter += 1
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            mask_points.append((x, y))
            updated = True
            circolor =[0,0,255]
            counter -= 1
            return
        
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    updated = False
    circolor = [0,255,0]
    toshow = scalemax(img,maxlen)
    witex = toshow.copy()
    counter = 0
    while True:
        cv2.imshow(window_name, witex)
        key = cv2.waitKey(1)
        if updated:
            if circolor == [0,255,0]:
                center = landslide_points.copy()[-1]
            else:
                center = mask_points.copy()[-1]
            updated = False
            toshow = cv2.circle(toshow, center, 3, circolor, 2)
            witex = cv2.putText(toshow.copy(), str(counter), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,140,252], 2, cv2.LINE_AA)

        if key in [27,13]:
            cv2.destroyAllWindows()
            break
    if output == 'points':
        return landslide_points, mask_points
    if output == 'colors':
        landslide_colors = []
        mask_colors = []
        for i in landslide_points:
            landslide_colors.append(list(resized[i[1],i[0]]))
        for i in mask_points:
            mask_colors.append(list(resized[i[1],i[0]]))
        return landslide_colors, mask_colors

def genmask(img,landslide_colors,mask_colors, maxlen = 500):
    def saturcolor(color):
        rescale = 255/max(color)
        for i in range(len(color)):
            color[i]= min(round(color[i]*rescale),255)
        return color
    def saturated(colorlist):
        for j in range(len(colorlist)):
            colorlist[j]= saturcolor(colorlist[j])
        return colorlist
    h,w = img.shape[:2]
    scalefactor = max([h,w])/maxlen
    # kernel_size = math.ceil(scalefactor)
    kernel_size = 2
    filtered = scalemax(img,maxlen)
    mask = np.zeros(filtered.shape[:2],dtype = np.uint8)
    allcolors = np.concatenate((np.array(saturated(landslide_colors)), np.array(saturated(mask_colors))))
    tree = KDTree(allcolors)
    allcolors = allcolors.tolist()
    hc,wc = filtered.shape[:2]
    print('assigning pixel values in mask...')
    for i in range(hc):
        for j in range(wc):
            lsl = 0
            no_lsl = 0
            _, indices = tree.query(saturcolor(filtered[i,j]), k=len(allcolors)/2)
            for k in indices:
                if allcolors[k] in landslide_colors:
                    lsl +=1
                elif allcolors[k] in mask_colors:
                    no_lsl +=1
                else:
                    print('Silvio Berlusconi va direttamente in paradiso')
            if no_lsl < lsl:
                mask[i,j]+=255
    print('done!')
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    mask =cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),iterations = 4)
    mask = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),iterations = 4)
    scaled_mask = cv2.resize(mask,(w,h))
    return scaled_mask

def mergemask(img,mask, export = True, exportname = 'trial.png'):
    h,w = img.shape[:2]
    output = img.copy()
    for i in range(h):
        for j in range(w):
            if mask[i,j] != 255:
                output[i,j] = np.array([0,255,0])
    cv2.imshow('filtered image',scalemax(output,1000))
    cv2.waitKey(0)
    if export:
        cv2.imwrite(exportname,output)
    return output

if __name__ == '__main__':
    pathfile = filedialog.askopenfilename()
    imagename = os.path.basename(pathfile)
    img = cv2.imread(pathfile)
    

    landslide_colors,mask_colors = selectcolors(img)
    mask = genmask(img,landslide_colors,mask_colors)
    cv2.imshow('mask',scalemax(mask,1000))
    cv2.waitKey(0)
    # mergemask(img,mask,exportname = imagename)



