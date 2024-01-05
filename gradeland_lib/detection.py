# -*- coding:utf-8 -*-
#本程序用于大津算法的实现
'''
author: binbinlan, marcolovati
data: 2022.05.10
email:787250087@qq.com, mlov@cdut.edu.cn
'''

import csv
import multiprocessing as mtp
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog

import cv2  # 导入opencv模块
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy as sp
from scipy import ndimage
import skimage
from skimage.morphology import reconstruction

from gradeland_lib import maskmkr as mkmk


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def adaptiveOtsu(gray,div_pixel,fraction,blur = True):
    """
    This function efficiently implements an adaptive version of the Otsu threshold, particularly useful for images with large shadows. Normal Otsu thresholding could result in large unsegmented areas, while traditional adaptive thresholds require manual calibration.
    """
    thresholded = np.zeros((gray.shape[0],gray.shape[1]), dtype=np.uint8)
    hei = int(gray.shape[0]/div_pixel)
    wid = int(gray.shape[1]/div_pixel)
    otsumat = np.zeros((hei,wid),dtype=np.uint8)
    for y in range(hei):
        for x in range(wid):
            otsumat[y][x] = cv2.threshold(gray[y*div_pixel:y*div_pixel+div_pixel,x*div_pixel:x*div_pixel+div_pixel],0 ,255, cv2.THRESH_OTSU)[0]
    if blur:
        otsumat = ndimage.gaussian_filter(otsumat, [3,3], mode='constant')
    for y in range(hei):
        for x in range(wid):
            thresholded[y*div_pixel:y*div_pixel+div_pixel,x*div_pixel:x*div_pixel+div_pixel]=cv2.threshold(gray[y*div_pixel:y*div_pixel+div_pixel,x*div_pixel:x*div_pixel+div_pixel],otsumat[y][x]*fraction,255,cv2.THRESH_BINARY)[1]
        if gray.shape[1]%div_pixel!=0:
            thresholded[y*div_pixel:y*div_pixel+div_pixel,wid*div_pixel:gray.shape[1]]=cv2.threshold(gray[y*div_pixel:y*div_pixel+div_pixel,wid*div_pixel:gray.shape[1]],otsumat[y][wid-1]*fraction,255,cv2.THRESH_BINARY)[1]
    if gray.shape[0]%div_pixel!=0:
        for x in range(wid):
            thresholded[hei*div_pixel:gray.shape[0],x*div_pixel:x*div_pixel+div_pixel]=cv2.threshold(gray[hei*div_pixel:gray.shape[0],x*div_pixel:x*div_pixel+div_pixel],otsumat[hei-1][x]*fraction,255,cv2.THRESH_BINARY)[1]
    return thresholded

def confirmedThreshold(to_b_conf,confirmed):
    """
    This function takes in two thresholds, a more permissive one and a stricter one. It then removes any areas in the permissive threshold that do not intersect with the stricter threshold. Essentially, this function expands the stricter threshold to the extent of the permissive threshold, while also eliminating any false positives that may have resulted from the permissive threshold.
    """
    cleaningmask = cv2.addWeighted(to_b_conf, 0.8, confirmed, 0.2, 0)
    dst = cv2.addWeighted(to_b_conf, 0.5, confirmed, 0.5, 0)
    mask = np.zeros((cleaningmask.shape[0],cleaningmask.shape[1]), dtype=np.uint8)
    mask = cv2.copyMakeBorder(mask,top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=[0, 0, 0])
    for y in range(cleaningmask.shape[0]):
        for x in range(cleaningmask.shape[1]):
            if cleaningmask[y][x]== 0:
                cleaningmask = cv2.floodFill( cleaningmask, mask, (x,y), 0, 0, 100)[1]
    cleaningmask = cv2.threshold(cleaningmask,0,255,cv2.THRESH_BINARY)[1]
    for y in range(cleaningmask.shape[0]):
        for x in range(cleaningmask.shape[1]):
            if cleaningmask[y][x]== 255:
                dst[y][x] = 255
    return dst

def DoubleOtsu(input_img,big_t = 0.9,small_t = 0.5):
    """
    This function extracts a stringent threshold and a more permissive one using the adaptiveOtsu() method, and then merges them using the confirmedThreshold() method.
    """
    dst1 = adaptiveOtsu(input_img,3,big_t)
    dst2 = adaptiveOtsu(input_img,3,small_t)
    dst = confirmedThreshold(dst1,dst2)
    return dst

def bottom_hat(input_img,doubleotsu_img,closings = 5):
    """
    also known as black top-hat transform, see https://en.wikipedia.org/wiki/Top-hat_transform
    """
    bottom_hat = np.zeros((input_img.shape[0],input_img.shape[1]), dtype=np.uint8)
    closed = cv2.dilate(cv2.erode(input_img,1,iterations=closings),1,iterations = closings)
    for i in range(bottom_hat.shape[0]):
        for j in range(bottom_hat.shape[1]):
            bottom_hat[i][j] = int(closed[i][j])-int(input_img[i][j])
    confirmer = cv2.threshold(doubleotsu_img,200,255,cv2.THRESH_BINARY)[1]
    bottom_hat = confirmedThreshold(bottom_hat,confirmer)
    return bottom_hat

def find_centroid(bwimage):
    """
    For every identified grain, this function computes the geometrical centroid as well as two orthogonal axes: one along the longest side and the other along the shortest side.
    """
    def calclen(pointlist,centroid):
        if len(pointlist)==0:
            return 1, [[int(centroid[0]),int(centroid[1])],[int(centroid[0]),int(centroid[1])]]
        else:
            sortedp = sorted(pointlist, key=lambda point: point[0])
            if sortedp[0][0]==sortedp[-1][0]:
                sortedp = sorted(pointlist, key=lambda point: point[1])
        return ((sortedp[-1][0]-sortedp[0][0])**2+(sortedp[-1][1]-sortedp[0][1])**2)**0.5, [[sortedp[0][0],sortedp[0][1]],[sortedp[-1][0],sortedp[-1][1]]]
    
    h2,w2 = bwimage.shape
    arealen = 0
    xsum = 0
    ysum = 0
    for i in range(h2):
        for j in range(w2):
            if bwimage[i,j]==255:
                xsum+=j
                ysum+=i
                arealen +=1
    centroid = [xsum/arealen,ysum/arealen]
    slopes = [[0.5,1,2],[-2,-1,-0.5]]
    lines = [[],[]]
    for i in range(len(slopes)):
        for j in range(len(slopes[i])+1):
            lines[i].append([])
    for i in range(h2):
        for j in range(w2):
            if bwimage[i,j]==255:
                dx = j-centroid[0]
                dy = i-centroid[1]
                relativept = [j,i]
                if abs(dx) <= 1:
                    lines[1][0].append(relativept)
                elif abs(dy) <= 1:
                    lines[0][0].append(relativept)
                else:
                    for h in range(len(slopes)):
                        for k in range(len(slopes[h])):
                            heightline = int(slopes[h][k]*dx)
                            if int(dy) == heightline:
                                lines[h][k+1].append(relativept)
    ratios = []
    axelens = {'Lmax': [], 'Lmin': []}
    for i in range(len(lines[0])):
        axe1,lines[0][i]= calclen(lines[0][i],centroid)
        axe2,lines[1][i] = calclen(lines[1][i],centroid)
        if min(axe1,axe2)!= 0:
            ratios.append(max(axe1,axe2)/min(axe1,axe2))
        else:
            ratios.append(0)
        axelens['Lmax'].append(max(axe1,axe2))
        axelens['Lmin'].append(min(axe1,axe2))
    realaxe = ratios.index(max(ratios))
    lines[0] = lines[0][realaxe]
    lines[1] = lines[1][realaxe]
    axelens['Lmax'] = axelens['Lmax'][realaxe]
    axelens['Lmin'] = axelens['Lmin'][realaxe]
    return centroid,lines, arealen, axelens

def rerangemarkers(markers,h,w):
    """
    Given the output of cv2.watershed(), this function converts it into a black and white image where each identified grain is entirely isolated.
    """
    for i in range(h):
        for j in range(w):
            currentpix = markers[i,j]
            if currentpix in [1,-1]:
                markers[i,j] = 0
            else:
                markers[i,j]=255
    return markers.astype(np.uint8)

def separate_img(insidefunc, markers, img,h,w,parallel=True):
    """
    This function accepts the entire image containing separate grains as input, and returns a list of smaller sub-images, each one cropped around a single grain. It is intended to be used within the FragAnalFg() method.
    """
    whitecoords = []
    maskmarkers = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
    for i in range(h):
        for j in range(w):
            if markers[i,j] == 255:
                whitecoords.append([i,j])
                markers = cv2.floodFill(markers,cv2.copyMakeBorder(np.zeros((h,w),dtype=np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=0),[j,i],0,10,10)[1]
    if parallel:
        print('number of cores',mtp.cpu_count())
        pool = mtp.Pool(processes = mtp.cpu_count()-1)
        outputlist = pool.starmap(insidefunc,[(whitecoord,maskmarkers,img,h,w) for whitecoord in whitecoords])
        pool.close()
        pool.join()
        return outputlist
    else:
        outputlist = []
        for i in whitecoords:
            outputlist.append(insidefunc(i,maskmarkers,img,h,w))
        return outputlist

def DistanceBasedInside(whitecoord,maskmarkers,img,h,w,erode=3):
    """
    this function finds the sure foreground 'surefg' by performing a distance-transform 
    """
    def is_max_pixel(image, x, y):
        max_pixel_value = image[y][x]  # Assuming image is a 2D list representing the greyscale image
        for i in range(-1, 2):
            for j in range(-1, 2):
                if image[y + j][x + i] > max_pixel_value:
                    return 0
        if max_pixel_value ==0:
            return 0
        else:
            return max_pixel_value
        
    temporary = np.zeros((h,w), dtype=np.uint8)
    temporary, _, rect = cv2.floodFill(temporary,maskmarkers,[whitecoord[1],whitecoord[0]],255,10,10)[1:]
    mask = temporary[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    img= img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    if erode != 0:
        mask = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_CROSS, (erode,erode)), iterations = 1)
    distance = cv2.distanceTransform(mask,cv2.DIST_L2,5).astype(np.uint8)
    # surefg = np.zeros([distance.shape[0],distance.shape[1]],dtype=np.uint8)
    # if min(surefg.shape)<5:
    #     return [surefg,rect]
    # else:
    #     structuring_edge = 5
    #     for wi in range(1,distance.shape[1]-1):
    #         for hi in range(1,distance.shape[0]-1):
    #             surefg [hi][wi]=is_max_pixel(distance,wi,hi)
    #     surefg =cv2.dilate(surefg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structuring_edge,structuring_edge)),iterations = 1)
    #     surefg = cv2.threshold(surefg,0 ,255, cv2.THRESH_OTSU)[1]

    surefg = cv2.threshold(distance,0 ,255, cv2.THRESH_OTSU)[1]
    return [surefg,rect]

def reunite_img(outputlist,h,w):
    """
    This function performs the reverse operation of separate_img(). It takes in a list of separate images and collates them into a unified picture.
    """
    toreunite = np.zeros((h,w),dtype=np.uint8)
    for i in outputlist:
        h2,w2 = i[0].shape[:2]
        for a in range(h2):
            for b in range(w2):
                    if i[0][a,b]!=0:
                        toreunite[a+i[1][1],b+i[1][0]]=i[0][a,b]
    return toreunite

def FragAnalFg(func,dividing,img,gray,h,w):
    countmask = cv2.countNonZero(cv2.inRange(img, np.array([0, 250, 0]), np.array([10, 255, 10])))
    if countmask ==0:
        markers = gray
    else:
        markers = img
    outputlist = separate_img(func,dividing,markers,h,w,parallel = False)
    return reunite_img(outputlist,h,w), outputlist

def exportGSD(markers,img, basename, centroidcolor = [197,97,255], min_area = 16,showimg = False, export = False):
    """
    This function takes the segmented grains as input, filters out smaller ones, superimposes their axes and borders onto the original image, and then saves the resulting output in a CSV file.
    """
    csvdata = {"centroid":[],"area":[],"Lmax":[],"Lmin":[],"axes":[]}
    h,w = markers.shape
    centrimg = img.copy()
    centrimg[markers == -1] = centroidcolor
    markers = markers.astype(np.uint8)
    for i in range(h):
        for j in range(w):
            if markers[i,j] == 255:
                temporary = np.zeros((h,w), dtype=np.uint8)
                mask = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
                retval, temporary, mask, rect = cv2.floodFill(temporary,mask,[j,i],255,10,10)
                subimage = temporary.copy()[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                centroid_loc,centr_lines, arealen, axelens = find_centroid(subimage)
                if arealen >= min_area:
                    csvdata["area"].append(arealen)
                    csvdata["Lmax"].append(axelens["Lmax"])
                    csvdata["Lmin"].append(axelens["Lmin"])
                    centroid = [int(centroid_loc[0])+rect[0],int(centroid_loc[1])+rect[1]]
                    csvdata["centroid"].append(centroid)
                    centrimg = cv2.rectangle(centrimg, [centroid[0]-1,centroid[1]-1], [centroid[0]+1,centroid[1]+1], centroidcolor, 2)
                    csvdata['axes'].append([[],[]])
                    for k in range(len(centr_lines)):
                        cv2.line(centrimg, np.add(np.array(centr_lines[k][0]),np.array(rect[:2])), np.add(np.array(centr_lines[k][1]),np.array(rect[:2])), centroidcolor, 2)
                        for a in range(len(centr_lines[k])):
                            csvdata['axes'][-1][k].append(list(np.add(np.array(centr_lines[k][a]),np.array(rect[:2]))))
                markers = cv2.floodFill(markers,cv2.copyMakeBorder(np.zeros((h,w),dtype=np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=0),[j,i],0,10,10)[1]
    df = pd.DataFrame.from_dict(csvdata)
    if export:
        df.to_csv(basename+'.csv', index=False)
    if showimg:
        cv2.imshow("detection",centrimg)
        cv2.waitKey(5000)
    return csvdata

def grain_size_dis(x):
    """
    this is the legacy version of exportGSD
    """
    x_sort = sorted(x)
    y_label=[]
    for enu,i in enumerate(x_sort):
        y = (enu+1)/len(x_sort)
        y = '{:.0%}'.format(y)
        y_label.append(y)
    #print(y_label)
    plt.Figure()
    fig,ax = plt.subplots(1,1)
    plt.xscale('log')
    plt.plot(x_sort,y_label)
    plt.xlabel('grain size')
    plt.ylabel('ratio')
    plt.title('Particle size curve')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=9))
    plt.savefig('颗粒级配曲线.png')
    plt.show()

def colordetect(markers,img, rerange = True):
    """
    takes the output of the watershed operation and creates a colorful representation of the different grains
    """
    h,w = markers.shape
    colorful = np.zeros((h,w,3), dtype=np.uint8)
    counter = 0
    colorpool = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[0,170,255],[0,255,200],[0,255,106],[0,140,255],[115,0,255],[255,0,166],[145,149,255],[153,255,145]]
    if rerange:
        markers = rerangemarkers(markers,h,w)
    for i in range(h):
        for j in range(w):
            if markers[i,j] == 255:
                temporary = np.zeros((h,w), dtype=np.uint8)
                mask = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
                temporary, mask= cv2.floodFill(temporary,mask,[j,i],255,10,10)[1:3]
                markers = cv2.floodFill(markers,cv2.copyMakeBorder(np.zeros((h,w),dtype=np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=0),[j,i],0,10,10)[1]
                colorful[temporary==255] = colorpool[counter]
                counter+=1
                if counter == len(colorpool):
                    counter = 0
    return cv2.addWeighted(colorful, 0.25, img, 0.75, 0)

def merge_split(window_name,markers,surefg,surebg,img,loadpic):
    #loadpic = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),"loading.png"))

    def mouse_callback(event, x, y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            toMerge[-1].append((x, y))
            if len(toMerge[-1]) == 2:
                toMerge.append([])
                return
        if event == cv2.EVENT_RBUTTONDOWN:
            toSplit[-1].append((x, y))
            if len(toSplit[-1]) == 2:
                toSplit.append([])
                return
        
    def merge_procedure(img,markers,toMerge):
        del toMerge[-1]
        print("uniting...",toSplit)
        mask = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),1,1,1,1,cv2.BORDER_CONSTANT,value = 0))
        difference = np.zeros(markers.shape,np.uint8)
        for i in toMerge:
            for j in i:
                _,difference,_,_= cv2.floodFill(difference, mask, j, 255, 10, 10)
        # Define the kernel for morphological closure
        kernel = np.ones((3,3),np.uint8)
        # Perform morphological closure
        difference= cv2.morphologyEx(difference, cv2.MORPH_CLOSE, kernel)
        difference = cv2.erode(difference,kernel,iterations = 1)
        markers[difference == 255] = 255
        colored = colordetect(markers.copy(),img, rerange= False)
        toMerge = [[]]
        print("done")
        return markers,toMerge, colored
    
    def split_procedure(img,markers,toSplit):
        del toSplit[-1]
        print("dividing...",toSplit)
        mask = np.zeros(markers.shape, np.uint8)
        mask[markers.astype(np.uint8) != 0] = 255
        mask = cv2.bitwise_not(cv2.copyMakeBorder(mask,top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
        # mask = cv2.bitwise_not(cv2.copyMakeBorder(markers.astype(np.uint8),top=1,bottom=1,left=1,right=1,borderType=cv2.BORDER_CONSTANT,value=255))
        marcopy = markers.copy()
        for i in range(len(toSplit)):
            marcopy = cv2.floodFill(marcopy,mask,toSplit[i][0],100,10,10)[1]
        markers[marcopy==100]= 0 
        surefg = np.zeros(markers.shape, np.uint8)
        for i in range(len(toSplit)):
            for j in range(len(toSplit[i])):
                cv2.circle(surefg, toSplit[i][j], 5, 255, 10)
        surebg = np.zeros(markers.shape, np.uint8)
        surebg[marcopy == 100] = 255
        unknown = cv2.subtract(surebg,surefg)
        temporary = cv2.connectedComponents(surefg, ltype=cv2.CV_32S)[1]
        temporary = temporary+1
        temporary[unknown==255] = 0
        temporary = cv2.watershed(cv2.GaussianBlur(img, (5, 5), 1),temporary)
        markers[temporary.astype(np.uint8)>1]=255
        markers[temporary.astype(np.uint8)==255]=0
        colored = colordetect(markers.copy(),img, rerange= False)
        toSplit = [[]]
        print("done")
        return markers, toSplit, colored

    
    colored = colordetect(markers,img)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    instructions = np.zeros((200,650),dtype=np.uint8)
    cv2.putText(instructions, "left click 2 points and press enter to merge them in a single area ", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(instructions, "right click 2 points and press enter to split them into 2 separate areas", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(instructions, "press esc when done", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("instructions",instructions)
    toMerge = [[]]
    toSplit = [[]]
    while True:
        cv2.imshow(window_name, colored)
        cv2.imshow("original",img)
        key = cv2.waitKey(1)
        if key == 13:
            cv2.imshow('loading', loadpic)
            cv2.waitKey(1)
            if len(toMerge[0])!= 0:
                markers,toMerge, colored = merge_procedure(img,markers,toMerge)
            if len(toSplit[0])!=0:
                markers,toSplit, colored = split_procedure(img,markers,toSplit)
            if len(toMerge[0])==0 and len(toSplit[0])==0:
                cv2.destroyWindow('loading')
        if key == 27:
            break
    unknown = cv2.subtract(surebg,surefg)
    cv2.destroyAllWindows()
    return surefg,unknown,markers,colored

def empty_borders(markers):
    h,w = markers.shape
    for i in range(w):
        for j in range(h):
            if any([i==1,i==w-2,j==1,j==h-2]):
                markers[markers==markers[j,i]] =0
    return cv2.threshold(markers.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)[1]

def exec(mode = 'automatic',roiselect = False, invert = False):
    """
    this is the main function of the program, later i will think if i want to transform it into a class
    """
    #type = sys.getfilesystemencoding()
    pathfile = filedialog.askopenfilename()
    basename = os.path.splitext(pathfile)[0]
    img = cv2.imread(pathfile)

    if roiselect:
        # 选择roi区域
        if max(img.shape[:2])>1000:
            scaled = mkmk.scalemax(img,1000)
        else:
            scaled = img.copy()
        factor = img.shape[0]/scaled.shape[0]
        roi = cv2.selectROI(windowName="select ROI", img=scaled, showCrosshair=True, fromCenter=False)  # 选择ROI区域
        roi = list(roi)
        for i in range(len(roi)):
            roi[i] = int(roi[i]*factor)
        x, y, w, h = roi  # 将选择的roi区域转换成坐标，x,y为ROI的坐标，w,h为ROI的大小
        print('roi区域的坐标是 ',roi)  # 输出roi区域的坐标

        # 显示ROI并保存图片
        if roi != [0, 0, 0, 0]:
            img = img[y:y + h, x: x+ w]  # 切片获取roi区域
            print(img.shape)
            cv2.imshow('crop', img)  # 显示roi区域
            # cv2.imwrite('crop.jpg', crop)  # 保存roi区域
            # print('Saved!')  # 输出保存成功
        cv2.destroyAllWindows() 

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape[:2]

    if invert:
        def on_tracebar_changed(args):
            pass
        def on_buttom_changed(args):
            pass
        print('选择好阈值后按q退出')
        toshow = gray.copy()
        cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
        cv2.createTrackbar('inverse','Image',0,1,on_tracebar_changed)
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        while True:
            time.sleep(0.1)
            inverse = cv2.getTrackbarPos('inverse', 'Image')
            #min_rec = cv2.getTrackbarPos('min_rec', 'Image')
            if inverse == 1:
                toshow = cv2.bitwise_not(gray)
            else:
                toshow = gray.copy()
            cv2.imshow('Image',toshow)
            key = cv2.waitKey(1)
            if key == ord('q') :
                break
        if inverse == 1:
                gray = cv2.bitwise_not(gray)
        cv2.destroyAllWindows()

    loadpic = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),"loading.png")) 
    cv2.imshow("loading",loadpic)
    cv2.waitKey(1)
    start0= start = time.time()
    print('pre-processing image...')
    fordst = cv2.equalizeHist(cv2.GaussianBlur(gray, (5, 5), 0))
    print('finding edges...')
    dst = DoubleOtsu(fordst)
    #bottom_hat = bottom_hat(gray,dst)
    # canny = confirmedThreshold(cv2.bitwise_not(cv2.Canny(gray,50,150)),cv2.threshold(dst,200,255,cv2.THRESH_BINARY)[1])
    # canny = cv2.threshold(canny,200,255,cv2.THRESH_BINARY)[1]
    # dst = cv2.addWeighted(dst, 0.5, canny, 0.5, 0)
    end = time.time()
    print('completed',end-start0,'seconds')
    print('finding foreground and background...')
    start0= time.time()
    surefg = FragAnalFg(DistanceBasedInside,cv2.threshold(dst,200,255,cv2.THRESH_BINARY)[1],img,fordst,h,w)[0]
    cv2.imwrite(r'C:\Users\lovam\Documents\sklgp_lit\3d_drone\paper\foreground.png',surefg)#-------------------------------
    # cv2.imshow('surefg',cv2.equalizeHist(surefg))
    # cv2.waitKey(0)
    surebg =  cv2.threshold(dst,0,255,cv2.THRESH_BINARY)[1]
    unknown = cv2.subtract(surebg,surefg)
    end = time.time()
    print('completed',end-start0,'seconds')
    markers = cv2.connectedComponents(surefg, ltype=cv2.CV_32S)[1]
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)


    print("entire process lasted", end-start)
    cv2.destroyAllWindows()
    if mode == 'automatic':
        #cv2.imwrite('initial_colored.jpg',colordetect(markers.copy(),img, rerange= True))
        def export_to_csv(data, file_path):
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(data)
        export_to_csv(markers,'prova.csv')
    elif mode == 'manual correction':
        surefg,unknown,markers,colored = merge_split("manual correction",markers,surefg,surebg,img,loadpic)
        cv2.imwrite(r'C:\Users\lovam\Documents\sklgp_lit\3d_drone\paper\colored.png',colored)#-------------------------------
        gsdata = exportGSD(markers,img, basename)
    return

if __name__ == '__main__':
    exec(mode = 'manual correction', roiselect= True)