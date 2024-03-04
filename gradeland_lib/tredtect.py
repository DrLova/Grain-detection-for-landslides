import json
import math
import multiprocessing
import os
import random as rnd
import threading
import time

import cv2
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from gradeland_lib import detection as dtct
from gradeland_lib import maskmkr as mkmk
from gradeland_lib import name_generator as nmgen
from gradeland_lib import nvm_to_obj as nto
from gradeland_lib import AIfilter as ai



def roiselect(photoanal,maxdim =1000):
    drawing = False  # True if mouse button is pressed
    ix, iy = -1, -1  # Initial coordinates of the rectangle
    current_x, current_y = -1, -1  # Current coordinates of the rectangle
    existing = False

    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, ix, iy, current_x, current_y,existing
        if event == cv2.EVENT_LBUTTONDOWN:
            existing = False
            drawing = True
            ix, iy = x, y
            current_x, current_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_x, current_y = x, y
            existing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                current_x, current_y = x, y

    cv2.namedWindow('Rectangle Drawing')
    cv2.setMouseCallback('Rectangle Drawing', draw_rectangle)
    image = cv2.imread(photoanal)
    # Display the image and rectangle
    rectcolor = nmgen.makeacolor()
    modcoords = []
    while True:
        if not existing:
            if max(image.shape[:2])<=maxdim:
                temp_image = image.copy()
                factor = 1
            else:
                temp_image = mkmk.scalemax(image.copy(),maxdim)
                factor = max(image.shape[:2])/max(temp_image.shape[:2])
        if drawing:
            cv2.rectangle(temp_image, (ix, iy), (current_x, current_y), rectcolor, 1)
        cv2.imshow('Rectangle Drawing', temp_image)
        # Exit on ESC key press
        choice = cv2.waitKey(1)
        if choice in [27,13,32]:
            if choice == 32:
                cv2.rectangle(image, (round(factor*ix), round(factor*iy)), (round(factor*current_x), round(factor*current_y)), rectcolor, round(factor*1))
                rectcolor = nmgen.makeacolor()
                modcoords.append([[factor*ix,factor*iy],[factor*current_x,factor*current_y]])
            else:
                modcoords.append([[factor*ix,factor*iy],[factor*current_x,factor*current_y]])
                break
    cv2.destroyAllWindows()
    return modcoords

def find_bounds(pointcloud):
    allx = []
    ally = []
    for i in pointcloud:
        allx.append(i[0])
        ally.append(i[1])
    return [min(allx),max(allx),min(ally),max(ally)]

def find_bounds_dict(clustered_dictionary):
    detectionpoints = []
    for multipledetect in clustered_dictionary:
        for instance in multipledetect['axes']:
            for axe in instance:
                for point in axe:
                    detectionpoints.append(point)
    return find_bounds(detectionpoints)

def change_coordinate_system(sourcedistances,sourcecoord,targetcoord, reference_edge = 0):
    
    def trilateration(points, distances, integers = True):
        d1,d2,d3 = distances
        x1,y1 = [float(i) for i in points[0]]
        x2,y2 = [float(i) for i in points[1]]
        x3,y3 = [float(i) for i in points[2]]
        A = 2 * (x2 - x1)
        B = 2 * (y2 - y1)
        C = d1**2 - d2**2 - x1**2 + x2**2 - y1**2 + y2**2
        D = 2 * (x3 - x2)
        E = 2 * (y3 - y2)
        F = d2**2 - d3**2 - x2**2 + x3**2 - y2**2 + y3**2
        x = (C*E - F*B) / (E*A - B*D)
        y = (C*D - A*F) / (B*D - A*E)
        if integers:
            return round(x),round(y)
        return x,y
    referedges = [[0,1],[1,2],[2,0]]
    reference = pdist([[float(i) for i in sourcecoord[referedges[reference_edge][0]]],[float(i) for i in sourcecoord[referedges[reference_edge][1]]]])[0]
    target_reference = pdist([targetcoord[referedges[reference_edge][0]],targetcoord[referedges[reference_edge][1]]])[0]
    targetdistances = [(sourcedistances[i]/reference)*target_reference for i in range(len(targetcoord))]
    return trilateration(targetcoord,targetdistances,integers=False)

def points_inimage(camera_id,nvmodel):
    inimage = []
    for i in range(len(nvmodel[0])):
        for j in range(len(nvmodel[0][i]['cameras'])):
            if int(nvmodel[0][i]['cameras'][j]['ID'])==int(camera_id):
                inimage.append([nvmodel[0][i],j])
    return inimage

def points_inimage_doubleref(photoanal,nvmodel):
    # check that the original 'points_inimage' could not be simply improved with the double reference
    cam_id = nto.photo_in_model(os.path.basename(photoanal),nvmodel)['ID']
    inimage = points_inimage(cam_id,nvmodel)
    inimage_pixel = []
    for i in range(len(inimage)):
        shouldpixel = inimage[i][0]['cameras'][inimage[i][1]]
        inimage_pixel.append([shouldpixel['xpix'],shouldpixel['ypix']])
    return inimage,inimage_pixel

def point_in_triangle(point, triangle):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    for i in range(len(triangle)):
        triangle[i]= [float(x) for x in triangle[i]]
    A,B,C = triangle
    d1 = sign(point, A, B)
    d2 = sign(point, B, C)
    d3 = sign(point, C, A)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def pixelto3d(pointcoords,pixeltree,inimage_pixel,cloud3d, reference_edge = 0):
    # neighbours = pixeltree.query(pointcoords,k=3)
    def close_containing(pixeltree,pointcoords,inimage_pixel, maxdist = 15):
        neighbours = pixeltree.query(pointcoords,k=maxdist)
        indexes = [0,1,2]
        while not point_in_triangle(pointcoords,[inimage_pixel[neighbours[1][indexes[0]]],inimage_pixel[neighbours[1][indexes[1]]],inimage_pixel[neighbours[1][indexes[2]]]]) and max(indexes)<maxdist-1:
            indexes[indexes.index(min(indexes))]+=3
        return [[neighbours[i][j] for j in indexes] for i in range(len(neighbours))]
    
    neighbours = close_containing(pixeltree,pointcoords,inimage_pixel)
    if type(cloud3d[0][0]) is dict:
        targetpoints = [[float(cloud3d[ne][0]['xyz'][dim]) for dim in range(2)] for ne in neighbours[1]]
    else:
        targetpoints = [[float(cloud3d[ne][dim]) for dim in range(2)] for ne in neighbours[1]]
    sourcepoints = [inimage_pixel[ne] for ne in neighbours[1]]
    x = 0
    y = 0
    for reference in range(3):
        dx,dy = change_coordinate_system(neighbours[0],sourcepoints,targetpoints,reference_edge=reference)
        x+=dx
        y+=dy
    return [x/3, y/3]

def makejoblist(modcoord,nvmodel,photofolder):
    coo2d = []
    joblist = []
    for i in range(len(nvmodel[0])):
        last = []
        for j in range(2):
            last.append(float(nvmodel[0][i]['xyz'][j]))
        coo2d.append(last)
    tree = KDTree(coo2d)
    ktree = 50
    neighbours = [tree.query(modcoord[0], k=ktree),tree.query(modcoord[1], k=ktree)]
    for onecamera in nvmodel[1]:
        inimage = points_inimage(onecamera['ID'],nvmodel)
        anal = 0
        threenear = [[],[]]
        threecoord = [[],[]]
        threedistance = [[],[]]
        threepixel = [[],[]]
        cropcoord = [[],[]]
        while any([len(threenear[ex])<3 for ex in range(len(threenear))]) and anal<len(inimage):
            for ex in range(len(threenear)):
                if int(inimage[anal][0]['ID']) in list(neighbours[ex][1]):
                    threenear[ex].append(inimage[anal][0])
            anal+=1
        
        if all([len(threenear[ex])>=3 for ex in range(len(threenear))]):
            for ln in range(len(threenear)):
                threenear[ln]= threenear[ln][:3]
                for point in threenear[ln]:
                    threecoord[ln].append(point['xyz'][:2])
                    threedistance[ln].append(pdist([[float(point['xyz'][i]) for i in range(2)],modcoord[ln]])[0])
                    for camerainfo in point['cameras']:
                        if int(camerainfo['ID'])==int(onecamera['ID']):
                            threepixel[ln].append([float(camerainfo['xpix']),float(camerainfo['ypix'])])
                cropcoord[ln] = [round(i) for i in change_coordinate_system(threedistance[ln],threecoord[ln],threepixel[ln])]
            conditions = []
            for coord in cropcoord:
                for dim in coord:
                    conditions.append(dim>0)
            if all(conditions):
                joblist.append((nvmodel,photofolder,cropcoord,onecamera['ID']))
    return joblist

def dtectmod(nvmodel,photofolder,cropcoord,imageid):

    def detect(nvmodel,photofolder, cropcoord, imageid):
        snippet = cv2.imread(os.path.join(photofolder,nvmodel[1][int(imageid)]['name']))[cropcoord[0][1]:cropcoord[1][1],cropcoord[0][0]:cropcoord[1][0]]
        gray=cv2.cvtColor(snippet,cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        dst = dtct.DoubleOtsu(cv2.equalizeHist(cv2.GaussianBlur(gray, (5, 5), 0)))
        surefg = dtct.FragAnalFg(dtct.DistanceBasedInside,cv2.threshold(dst,200,255,cv2.THRESH_BINARY)[1],snippet,gray,h,w)[0]
        surebg = cv2.threshold(dst,0,255,cv2.THRESH_BINARY)[1]
        unknown = cv2.subtract(surebg,surefg)
        markers = cv2.connectedComponents(surefg, ltype=cv2.CV_32S)[1]
        markers = markers+1
        markers[unknown==255] = 0
        markers = cv2.watershed(snippet,markers)
        gsd = dtct.exportGSD(dtct.empty_borders(markers).copy(),snippet, 'not in use', centroidcolor = [197,97,255], min_area = 16,export=False) 
        return gsd
    try:
        print("detecting",imageid)
        gsd = detect(nvmodel,photofolder, cropcoord, imageid)
    except:
        print('the snippet has wrong coordinates',cropcoord)
        gsd = dtct.exportGSD(np.zeros((10,10),dtype =np.uint8),np.zeros((10,10),dtype =np.uint8), 'not in use', centroidcolor = [197,97,255], min_area = 16,export=False)
    photocoord = []
    for i in range(len(gsd['axes'])):
        photocoord.append([])
        for j in range(len(gsd['axes'][i])):
            photocoord[i].append([])
            for k in range(len(gsd['axes'][i][j])):
                photocoord[i][j].append([gsd['axes'][i][j][k][a] + cropcoord[0][a] for a in range(len(cropcoord[0]))])

    photoanal = os.path.join(photofolder,nvmodel[1][int(imageid)]['name'])
    detect3D_list = []
    centroid3D_list = []

    inimage,inimage_pixel = points_inimage_doubleref(photoanal,nvmodel)
    tree = KDTree(inimage_pixel)
    for i in range(len(photocoord)):
        detection = []
        detcentroid = [0,0]
        for j in photocoord[i]:
            detection.append([])
            for k in j:
                detect3d = pixelto3d(k,tree,inimage_pixel,inimage)
                detection[-1].append(detect3d)
                for dim in range(len(detect3d)):
                    detcentroid[dim]+=detect3d[dim]/4
        detect3D_list.append(detection)
        centroid3D_list.append(detcentroid)
    print('completed detection of',imageid)
    return {'ID':imageid,'axes':detect3D_list,'centroids':centroid3D_list}

def multidetect(modcoord,photofolder,nvmodel,export = False, exportname = 'multidetect_try.json'):
    joblist = makejoblist(modcoord,nvmodel,photofolder)
    dellist = []
    for job in joblist:
        img = cv2.imread(os.path.join(photofolder,nvmodel[1][int(job[3])]['name']))
        if all([job[2][0][1]>0,job[2][0][0]>0,job[2][1][1]<img.shape[1],job[2][1][0]<img.shape[0],job[2][0][1]<job[2][1][1]-5,job[2][0][0]<job[2][1][0]-5]):
            snippet = img[job[2][0][1]:job[2][1][1],job[2][0][0]:job[2][1][0]]
            # try:
            #     cv2.imshow(str(job[3]),snippet)
            # except:
            #     dellist.append(joblist.index(job))
            #     print('error snippet',[job[2][0][1],job[2][1][1],job[2][0][0],job[2][1][0]])
        else:
            dellist.append(joblist.index(job))
    # cv2.waitKey(1000)
    for i in sorted(dellist,reverse=True):
        del joblist[i]
    cv2.destroyAllWindows()
    coreuse = round(multiprocessing.cpu_count()*(2/3))
    print('the process is using',coreuse,'processors')
    pool = multiprocessing.Pool(processes = coreuse)
    moddetect_list = pool.starmap(dtectmod,joblist)
    moddetect_list.insert(0,{'cropping coordinates': modcoord})
    pool.close()
    pool.join()
    if export:
        with open(exportname,'w') as jsonout:
            jsonout.write(json.dumps(moddetect_list))
    return moddetect_list

def confirm3d(rawdict,max_oblateness = 3.5,max_delta = 0.1,export = False,exportname = 'filtered.json'):
    centroids = []
    axes = []
    IDs = []
    vowels,consonnants = nmgen.makeletterlists()
    del rawdict[0]
    for photodata in rawdict:
        for detection in range(len(photodata['centroids'])):
            centroids.append(photodata['centroids'][detection])
            axes.append(photodata['axes'][detection])
            IDs.append(photodata['ID'])

    tree = KDTree(centroids)
    neighbourlen = 2*len(rawdict)
    pastdetections = []
    confirmedict = []

    for ic in range(len(centroids)):
        point_anal = centroids[ic]
        axes_anal_lens = []
        for singleaxe in axes[ic]:
            axes_anal_lens.append(pdist([singleaxe[0],singleaxe[1]])[0])
        filtered = []
        if all(axes_anal_lens[i]> 0 for i in range(len(axes_anal_lens))):
            neighbours = tree.query(point_anal,k=neighbourlen)
            for pc in range(1,len(neighbours)):
                index_confirmation = neighbours[1][pc]
                distance = neighbours[0][pc]
                two_axes = axes[index_confirmation]
                axelens = []
                for singleaxe in two_axes:
                    axelens.append(pdist([singleaxe[0],singleaxe[1]])[0])
                if all(axelens[i]> 0 for i in range(len(axelens))):
                    if max(axelens)/min(axelens)<= max_oblateness:
                        if distance < max(axes_anal_lens):
                            area_anal = min(axes_anal_lens)*max(axes_anal_lens)
                            area_candidate = min(axelens)*max(axelens)
                            if max([area_anal,area_candidate])/min([area_anal,area_candidate])-1 < max_delta:
                                filtered.append(index_confirmation)
        if len(filtered)>0:
            anal_detection = [IDs[ic],IDs[:ic+1].count(IDs[ic])]
            if anal_detection not in pastdetections:
                pastdetections.append(anal_detection)
                confirmeditem = {'name':nmgen.makeaname(6,vowels,consonnants),'color':nmgen.makeacolor(),'instances[ID,detection]':[anal_detection],'instances axes':[axes[ic]]}
                for index_confirmed in filtered:
                    amongdetections = [IDs[index_confirmed],IDs[:index_confirmed+1].count(IDs[index_confirmed])]
                    confirmeditem['instances[ID,detection]'].append(amongdetections)
                    pastdetections.append(amongdetections)
                    confirmeditem['instances axes'].append(axes[index_confirmed])
                confirmedict.append(confirmeditem)
            else:
                for confirmeditem in confirmedict:
                    if anal_detection in confirmeditem['instances[ID,detection]']:
                        for index_confirmed in filtered:
                            amongdetections = [IDs[index_confirmed],IDs[:index_confirmed+1].count(IDs[index_confirmed])]
                            if amongdetections not in pastdetections:
                                confirmeditem['instances[ID,detection]'].append(amongdetections)
                                pastdetections.append(amongdetections)
                                confirmeditem['instances axes'].append(axes[index_confirmed])
    if export:
        with open(exportname,'w') as jsonout:
            jsonout.write(json.dumps(confirmedict))
    return confirmedict

def cluster_1st(importname,exportname,export = True,megalith=0.8,axe_detail=0.4,core_fraction = 0.25,min_confirmations = 5):
    '''aaloa 0.3211218891790367 0.27684637920980454 4.223035087643648 6
vloci 0.4601285667025782 0.38193560475093596 3.582462859520657 5
ailoo 0.42970694617175687 0.4319624353525562 2.9836220967220086 4
mvefs 0.9477736182526602 0.4373537661163098 4.7617503841750555 4'''
    def listdetections(rawdict,megalith,max_oblateness = 3.5):
        rawdetections = [] # the rawdetections contain [centroid x, centroid y, len max axis,len min axis]
        listdict = []
        cropbound = find_bounds(rawdict[0]['cropping coordinates'])
        for photo in range(1,len(rawdict)):
            for detectnum in range(len(rawdict[photo]['centroids'])):
                centroid = rawdict[photo]['centroids'][detectnum]
                if all([centroid[0]>=cropbound[0],centroid[0]<=cropbound[1],centroid[1]>=cropbound[2],centroid[1]<=cropbound[3]]):
                    axes = rawdict[photo]['axes'][detectnum]
                    lens = []
                    for axe in axes:
                        lens.append(pdist([axe[0],axe[1]])[0])
                    if min(lens)>0:
                        if max(lens)< megalith*pdist([rawdict[0]['cropping coordinates'][0],rawdict[0]['cropping coordinates'][1]])[0]:
                            if max(lens)/min(lens)< max_oblateness:
                                rawdetections.append(rawdict[photo]['centroids'][detectnum])
                                rawdetections[-1].append(max(lens))
                                rawdetections[-1].append(min(lens))
                                listdict.append({'ID':rawdict[photo]['ID'],'centroids':rawdict[photo]['centroids'][detectnum],'axes':rawdict[photo]['axes'][detectnum]})
        return rawdetections,listdict
    listdict = []
    cube = []
    for singleimport in importname:
        print('reading file',importname.index(singleimport)+1,'of',len(importname))
        with open(singleimport, 'r') as json_file:
            json_data = json_file.read()
        rawdict = json.loads(json_data)
        partialcube,partiallist = listdetections(rawdict,megalith)
        listdict.extend(partiallist)
        cube.extend(partialcube)
    median_stone_len = sorted(cube.copy(), key=lambda x: x[2])[round(len(cube)/2)][2]
    thresholdaxe = median_stone_len*axe_detail
    core_fraction *=median_stone_len
    confirmed = []
    vowels,consonnants = nmgen.makeletterlists()
    tree = KDTree([x[:2] for x in cube])
    index = 0
    indexlist = [x for x in range(len(listdict))]
    while index < len(listdict):
        if (index+indexlist[-1]+1-len(listdict))%50 ==0:
            percentage = (index+indexlist[-1]+1-len(listdict))/(indexlist[-1]+1)
            print('clustering',percentage*100,'%')
        ingroups = [indexlist[index]]
        chosenone = cube[index]
        index_neighbours = tree.query_ball_point(chosenone[:2],float(core_fraction))
        for onenbr in index_neighbours:
            if onenbr in indexlist:
                otherchosen = cube[indexlist.index(onenbr)]
                if all([chosenone[2]-thresholdaxe<=otherchosen[2]<=chosenone[2]+thresholdaxe,chosenone[3]-thresholdaxe<=otherchosen[3]<=chosenone[3]+thresholdaxe,onenbr not in ingroups]):
                    ingroups.append(onenbr)
        if len(ingroups)>=min_confirmations:
            confirmed.append({'name':nmgen.makeaname(6,vowels,consonnants),'color':nmgen.makeacolor(),'centroids':[],'axes':[],'axes lenghts':[]})
            for onenbr in ingroups:
                if onenbr in indexlist:
                    ingroup = indexlist.index(onenbr)
                    confirmed[-1]['axes'].append(listdict[ingroup]['axes'])
                    confirmed[-1]['axes lenghts'].append(cube[ingroup][-2:])
                    confirmed[-1]['centroids'].append(cube[ingroup][:2])
                    del listdict[ingroup]
                    del cube[ingroup]
                    del indexlist[ingroup]
            # del listdict[index]
            # del cube[index]
            # del indexlist[index]
        else:
            index +=1
    if export:
        with open(exportname,'w') as jsonout:
            jsonout.write(json.dumps(confirmed))
    return confirmed

def cluster_2nd(clusteredict,model,acti_threshold = 0.8, export = True, exportname = 'cleaned.json'):
    def groupcentroids(clusteredict):
        allcentroids = []
        groups = []
        for i in range(len(clusteredict)):
            groups.append([])
            for centroid in clusteredict[i]['centroids']:
                groups[-1].append(len(allcentroids))
                allcentroids.append(centroid)
        return allcentroids,groups
    
    def find_superimposed(tree,clusteredict,i,groups):
        detection = clusteredict[i]
        superimposed = []
        in_proximity = []
        unique_inside = []
        avglen = 0
        boundcentroid = find_bounds(detection['centroids'])
        for centroid in range(len(detection['centroids'])):
            avglen+= detection['axes lenghts'][centroid][1]/len(detection['centroids'])
        for centroid in [[boundcentroid[0],boundcentroid[2]],[boundcentroid[1],boundcentroid[3]]]:
            neighbour = tree.query_ball_point(centroid,avglen)
            in_proximity.extend(neighbour)
        [unique_inside.append(item) for item in in_proximity if item not in unique_inside]
        for index in unique_inside:
            for dictindex in range(len(groups)):
                if index in groups[dictindex]:
                    entry = clusteredict[dictindex]
                    if entry not in superimposed:
                        superimposed.append(entry)
        return superimposed
    
    def gen_detection_features(superimposed):
        detection_features = {'name':[],'dispersion':[],'avg centroid':[],'avg axe len':[],'number detections':[]}
        for j in superimposed:
            detection_features['name'].append(j['name'])
            distlist = pdist(j['centroids'])
            detection_features['dispersion'].append(sum(distlist)/len(distlist))
            detection_features['number detections'].append(len(j['centroids']))
            avgcentroid =[0,0]
            avg_axelen = 0
            for centroid in range(len(j['centroids'])):
                avgcentroid[0]+= j['centroids'][centroid][0]/len(j['centroids'])
                avgcentroid[1]+= j['centroids'][centroid][1]/len(j['centroids'])
                avg_axelen+= ((j['axes lenghts'][centroid][0]/len(j['centroids']))+(j['axes lenghts'][centroid][1]/len(j['centroids'])))/2
            detection_features['avg centroid'].append(avgcentroid)
            detection_features['avg axe len'].append(avg_axelen)
        return detection_features
    
    def feature_to_input(detection_features,b):
        training_x = []
        training_x.append(detection_features['dispersion'][0]/max([detection_features['dispersion'][x] for x in [0,b]]))
        training_x.append(detection_features['dispersion'][b]/max([detection_features['dispersion'][x] for x in [0,b]]))
        training_x.append(detection_features['number detections'][0]/max([detection_features['number detections'][x] for x in [0,b]]))
        training_x.append(detection_features['number detections'][b]/max([detection_features['number detections'][x] for x in [0,b]]))
        rawdist = pdist([detection_features['avg centroid'][x] for x in [0,b]])[0]
        distances = [rawdist/detection_features['avg axe len'][x] for x in [0,b]]
        training_x.append(distances[0]/max(distances))
        training_x.append(distances[1]/max(distances))
        return training_x
    
    allcentroids,groups = groupcentroids(clusteredict)
    tree = KDTree(allcentroids)
    inputlist = []
    nameslist = []
    clusterindex = []
    clean_dict = []
    print('finding superimposed')
    for i in range(len(clusteredict)):
        if i%50 == 0:
            print('finding superimposed',(i/len(clusteredict))*100,'%')
        clusterindex.append(clusteredict[i]['name'])
        superimposed = find_superimposed(tree,clusteredict,i,groups)
        detection_features= gen_detection_features(superimposed)
        if len(detection_features['name'])>=2:
            for b in range(1, len(detection_features['name'])):
                inputlist.append(feature_to_input(detection_features,b))
                nameslist.append([detection_features['name'][0],detection_features['name'][b]])
        else:
            clean_dict.append(clusteredict[i])
    predictionlist = model.predict(inputlist)

    print('second clustering')
    for i in range(len(predictionlist)):
        listindexes =[clusterindex.index(nameslist[i][0]),clusterindex.index(nameslist[i][1])]
        for index in listindexes:
            if 'erase likelyhood' not in clusteredict[index].keys():
                clusteredict[index]['erase likelyhood'] = []
            if 'merge partners' not in clusteredict[index].keys():
                clusteredict[index]['merge partners'] = []
            clusteredict[index]['erase likelyhood'].append(0)
        if predictionlist[i]<=2:
            clusteredict[listindexes[predictionlist[i]-1]]['erase likelyhood'][-1] += 1
        if predictionlist[i]==3:
            clusteredict[listindexes[0]]['merge partners'].append(listindexes[1])
            clusteredict[listindexes[1]]['merge partners'].append(listindexes[0])

    merged = []
    for i in range(len(clusteredict)):
        if 'erase likelyhood' and 'merge partners' in clusteredict[i].keys():
            if all([i not in merged,sum(clusteredict[i]['erase likelyhood'])/len(clusteredict[i]['erase likelyhood'])<acti_threshold]):
                clean_dict.append(clusteredict[i])
                del clean_dict[-1]['erase likelyhood']
                if len(clusteredict[i]['merge partners'])!=0:
                        for partnerindex in clusteredict[i]['merge partners']:
                            if all([partnerindex!=i,partnerindex not in merged]):
                                merged.append(partnerindex)
                                for keyword in ['centroids','axes','axes lenghts']:
                                    clean_dict[-1][keyword].extend(clusteredict[partnerindex][keyword])
                del clean_dict[-1]['merge partners']
    if export:
        with open(exportname,'w') as jsonout:
            jsonout.write(json.dumps(clean_dict))
    return clean_dict

def gen2_cluster(multi_folder,exportname, min_detection = 5, max_oblatenes = 3.5, radius_median = 0.25, deltamax_median = 0.4):
    vowels,consonnants = nmgen.makeletterlists()
    print('collecting multidetect files')
    big_dic ={}
    for i in os.listdir(multi_folder):
        with open(os.path.join(multi_folder,i), 'r') as json_file:
            json_data = json_file.read()
        rawdict = json.loads(json_data)
        cropco = ai.find_bounds(rawdict[0]['cropping coordinates'])
        for j in range(1,len(rawdict)):
            lastid = rawdict[j]['ID']
            if lastid not in big_dic.keys():
                big_dic[lastid] = {'axes':[], 'centroids':[], 'axes lenghts':[]}
            for centroid in rawdict[j]['centroids']:
                if all([cropco[0]<centroid[0]<cropco[1],cropco[2]<centroid[1]<cropco[3]]):
                    cindex = rawdict[j]['centroids'].index(centroid)
                    twoaxe = rawdict[j]['axes'][cindex]
                    lens = [pdist(twoaxe[0])[0],pdist(twoaxe[1])[0]]
                    if 0 not in lens:
                        if min(lens)/max(lens)>(1/max_oblatenes):
                            big_dic[lastid]['centroids'].append(centroid)
                            big_dic[lastid]['axes'].append(rawdict[j]['axes'][cindex])
                            big_dic[lastid]['axes lenghts'].append(lens)

    centroidlist = []
    features = []
    allens = []
    burninglist = []

    for akey in big_dic.keys():
        for instance in range(len(big_dic[akey]['centroids'])):
            burninglist.append(len(centroidlist))
            centroidlist.append(big_dic[akey]['centroids'][instance])
            features.append([akey,instance])
            allens.append(min(big_dic[akey]['axes lenghts'][instance]))

    tree = KDTree(centroidlist)
    searchradius = list(sorted(allens.copy()))[round(len(allens)/2)]*radius_median
    deltalen = list(sorted(allens.copy()))[round(len(allens)/2)]*deltamax_median

    grouped = []
    while len(burninglist)>0:
        burn = math.ceil(rnd.random()*len(burninglist))-1
        toanal = burninglist[burn]
        potentials = tree.query_ball_point(centroidlist[toanal],r=searchradius)
        potentials = list(set(potentials))
        if len(potentials)> min_detection:
            grouped.append(sorted([i for i in potentials]))
        del burninglist[burn]
        print('grouping potential detetions',round((1-(len(burninglist)/len(centroidlist)))*100),'%')

    for g in reversed(range(len(grouped))):
        if grouped.count(grouped[g])>1:
            del grouped[g] 
        else:
            print('filtering by centroid condition',round(100-((grouped.index(grouped[g])/len(grouped))*100)),'%')
            for instance in grouped[g]:
                groupindexes = []
                grouplens = []
                for againgroup in grouped:
                    if instance in againgroup:
                        instanceindex = grouped.index(againgroup)
                        if instanceindex not in groupindexes:
                            groupindexes.append(instanceindex)
                            grouplens.append(len(againgroup))
                topreserve = grouplens.index(max(grouplens))
                for i in range(len(groupindexes)):
                    if i != topreserve:
                        del grouped[groupindexes[i]][grouped[groupindexes[i]].index(instance)]

    for g in reversed(range(len(grouped))):
        if len(grouped[g])<min_detection:
            del grouped[g]
    cleandict=[]
    for group in grouped:
        print('filtering by characteristic lenght',round((grouped.index(group)/len(grouped))*100),'%')
        groupindexes = group.copy()
        grouplens = []
        for instance in group:
            grouplens.append(min(big_dic[features[instance][0]]['axes lenghts'][features[instance][1]]))
            # print('last grouplens is',grouplens[-1])
        while max(grouplens)-min(grouplens)>deltalen:
            nomin = grouplens.copy()
            nomax = grouplens.copy()
            del nomin[nomin.index(min(grouplens))]
            del nomax[nomax.index(max(grouplens))]
            bestimprovement = [max(nomin)-min(nomin),max(nomax)-min(nomax)].index(min([max(nomin)-min(nomin),max(nomax)-min(nomax)]))
            if bestimprovement == 0:
                del groupindexes[grouplens.index(min(grouplens))]
                del grouplens[grouplens.index(min(grouplens))]
            if bestimprovement == 1:
                del groupindexes[grouplens.index(max(grouplens))]
                del grouplens[grouplens.index(max(grouplens))]
        if len(groupindexes) >=min_detection:
            cleandict.append({'name':str(nmgen.makeaname(6,vowels=vowels,consonnants=consonnants)),'color':nmgen.makeacolor(),'centroids':[],'axes':[],'axes lenghts':[]})
            for instance in groupindexes:
                cleandict[-1]['centroids'].append(big_dic[features[instance][0]]['centroids'][features[instance][1]])
                cleandict[-1]['axes'].append(big_dic[features[instance][0]]['axes'][features[instance][1]])
                cleandict[-1]['axes lenghts'].append(big_dic[features[instance][0]]['axes lenghts'][features[instance][1]])
                
    with open(exportname,'w') as jsonout:
        jsonout.write(json.dumps(cleandict))

    return cleandict

def import_clustering_model():
    with open(os.path.join('gradeland_lib','training_dict.json'),'r') as jsonin:
        jsondata = jsonin.read()
    training = json.loads(jsondata)
    model = RandomForestClassifier(n_estimators=100)
    X_train = training['x']
    y_train = training['y']
    model.fit(X_train, y_train)
    print('fitting complete')
    return model

if __name__ == '__main__':
    photoanal = r'C:\Users\lovam\Documents\sklgp_lit\3d_drone\gully_2\rotated\DJI_20230715162906_0040_V.JPG'
    photofolder = os.path.dirname(photoanal)
    nvmpath = r'wenchuan_debris\wenchuan.nvm'
    nvmodel = nto.nvm_to_points(nvmpath,photofolder,' ')
    inimage,inimage_pixel = points_inimage_doubleref(photoanal,nvmodel)
    newarea = False
    mode = 'loop'

    if newarea:
        roicoord = roiselect(photoanal,maxdim =1000)
        # try to perform it efficiently if the 3D has been exported, recycle the goodpoint_index--------------
        meshwriter = nto.obj_exporter(photoanal,[apoint[0] for apoint in inimage])
        meshwriter.makelists()
        meshwriter.remove_abnormal_vertex()
        for i in reversed(range(len(inimage))):
            if i not in meshwriter.goodpoint_index:
                del inimage[i]
                del inimage_pixel[i]
        # try to perform it efficiently if the 3D has been exported-------------------------------------------
        tree = KDTree(inimage_pixel)
        if mode == 'simple':
            for rectangle in roicoord:
                modcoord = [pixelto3d(rectangle[0],tree,inimage_pixel,inimage),pixelto3d(rectangle[1],tree,inimage_pixel,inimage)]
                multidetect(modcoord,photofolder,nvmodel,export = True,exportname =os.path.join('multidetect_files',str(modcoord[0])+'_'+str(modcoord[1])+'.json'))
        if mode == 'loop':
            limitsize = [200,600]
            h,w = cv2.imread(photoanal).shape[:2]
            analyzed = []
            anal_mask = np.zeros([h,w],dtype =np.uint8)
            closer = max([h,w])
            for rectangle in roicoord:
                cv2.rectangle(anal_mask,[round(d) for d in rectangle[0]],[round(d) for d in rectangle[1]],color = 255,thickness=cv2.FILLED)
            while closer > limitsize[0]:
                if len(analyzed)==0:
                    centroid_anal_rect = [round(rnd.random()*(w-1)),round(rnd.random()*(h-1))]
                    while anal_mask[centroid_anal_rect[1],centroid_anal_rect[0]]==0:
                        centroid_anal_rect = [round(rnd.random()*w),round(rnd.random()*h)]
                else:
                    pointcloud = [[round(rnd.random()*(w-1)),round(rnd.random()*(h-1))] for _ in range(500)]
                    distancelist = []
                    for n in reversed(range(len(pointcloud))):
                        if anal_mask[pointcloud[n][1],pointcloud[n][0]]==0:
                            del pointcloud[n]
                    for p in range(len(pointcloud)):
                        exclusive =[pointcloud[p]]
                        exclusive.extend(analyzed)
                        excl_dist_list = pdist(exclusive)[:len(analyzed)]
                        distancelist.append(min(excl_dist_list))
                    closer = max(distancelist)
                    centroid_anal_rect= pointcloud[distancelist.index(closer)]
                cv2.circle(anal_mask, centroid_anal_rect, 2, 100, 2)
                analyzed.append(centroid_anal_rect)
            print(len(analyzed))
            cv2.imshow('bew',mkmk.scalemax(anal_mask,1000))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            for centroid_anal_rect in analyzed:
                # dx = round(rnd.random()*(limitsize[1]-limitsize[0]))+limitsize[0]
                # dy = round(rnd.random()*(limitsize[1]-limitsize[0]))+limitsize[0]
                dx = round(min([closer,limitsize[1]]))
                dy = round(dx* (rnd.random()*0.6+0.7))
                rectangle=[[max(0,centroid_anal_rect[0]-dx),max(0,centroid_anal_rect[1]-dy)],[min(w,centroid_anal_rect[0]+dx),min(h,centroid_anal_rect[1]+dy)]]
                modcoord = [pixelto3d(rectangle[0],tree,inimage_pixel,inimage),pixelto3d(rectangle[1],tree,inimage_pixel,inimage)]
                multidetect(modcoord,photofolder,nvmodel,export = True,exportname =os.path.join('multidetect_files',str(modcoord[0])+'_'+str(modcoord[1])+'.json'))

    files_list = os.listdir('multidetect_files')
    for i in range(len(files_list)):
        files_list[i]= os.path.join('multidetect_files',files_list[i])
    print('performing first clustering')
    filteredict = cluster_1st(files_list,'clustered.json')
    print('performing second clustering')
    with open('clustered.json', 'r') as json_file:
        json_data = json_file.read()
    filteredict = json.loads(json_data)
    clean_dict = cluster_2nd(filteredict,import_clustering_model())
    h,w = cv2.imread(photoanal).shape[:2]
    import visualize as viz
    stones = viz.paint_cleandict(clean_dict,max_height=h,bounds=find_bounds([[float(dim) for dim in x[0]['xyz'][:2]] for x in inimage]))
    cv2.imwrite('detection.jpg',stones)
    gsd_chart, detectionlist = viz.GSD_chart(clean_dict,900,1200)
    excel = {'detection number':[x for x in range(len(detectionlist))], 'characteristic radius':[x[1] for x in detectionlist],'smallest possible characteristic radius':[x[0] for x in detectionlist],'largest possible characteristic radius':[x[2] for x in detectionlist]}
    print([len(excel[x]) for x in excel.keys()])
    df = pd.DataFrame(excel)
    print(df)
    df.to_excel('GSD.xlsx',index=False)
