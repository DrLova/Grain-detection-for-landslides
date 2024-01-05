import json
import cv2
from sklearn.cluster import MeanShift
from scipy.spatial import KDTree
import numpy as np
import math

from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestClassifier
from gradeland_lib import name_generator as nmgen

def find_bounds(pointcloud):
    allx = []
    ally = []
    for i in pointcloud:
        allx.append(i[0])
        ally.append(i[1])
    return [min(allx),max(allx),min(ally),max(ally)]

def transpose_cube(pointlist):
    '''this function is designed to prepare the data for the mean shift fitting. therefore the array should be cubical to avoid biasing the group formation toward some specific dimensions'''
    cubical = []
    transposed = [[] for i in pointlist[0]]
    for point in pointlist:
        for dim in range(len(point)):
            transposed[dim].append(point[dim])
    bonded = [[min(transposed[x]),max(transposed[x])]for x in range(len(transposed))]
    for point in pointlist:
        cubicpoint = []
        for dim in range(len(point)):
            cubicpoint.append((point[dim]-bonded[dim][0])/(bonded[dim][1]-bonded[dim][0]))
        cubical.append(cubicpoint)
    return cubical


def paint_filteredict(filteredict,definition = 800,linealpha = 200):
    bounds = find_bounds_dict(filteredict)
    dimensions = [bounds[3]-bounds[2],bounds[1]-bounds[0]]
    shapelike = []
    for i in dimensions:
        if i == max(dimensions):
            shapelike.append(definition)
        else:
            shapelike.append(round((i/max(dimensions))*definition))
    shapelike.append(3)
    canvas = np.zeros(shapelike,np.uint8)

    for i in filteredict:
        detectlist = i['axes']
        for detection in detectlist:
            for axe in detection:
                digiaxe = []
                for point in axe:
                    digiaxe.append([round(((point[0]-bounds[0])/(bounds[1]-bounds[0]))*shapelike[1]),round(((point[1]-bounds[2])/(bounds[3]-bounds[2]))*shapelike[0])])
                if all(num > 0 for num in digiaxe[0]) and all(num > 0 for num in digiaxe[1]):
                    output_image = canvas.copy()
                    cv2.line(output_image, digiaxe[0],digiaxe[1],i['color'],5)
                    canvas = cv2.addWeighted(output_image, (1 - linealpha / 255), canvas, (linealpha / 255), 0)
    
    return cv2.flip(canvas,0)

def listdetections(rawdict,max_oblateness = 3.5, transpose = True):
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
                    lens.append(pdist([axe[0],axe[1]]))
                if min(lens)>0:
                    if max(lens)< 0.3*pdist([rawdict[0]['cropping coordinates'][0],rawdict[0]['cropping coordinates'][1]]):
                        if max(lens)/min(lens)< max_oblateness:
                            rawdetections.append(rawdict[photo]['centroids'][detectnum])
                            rawdetections[-1].append(max(lens))
                            rawdetections[-1].append(min(lens))
                            listdict.append({'ID':rawdict[photo]['ID'],'centroids':rawdict[photo]['centroids'][detectnum],'axes':rawdict[photo]['axes'][detectnum]})
    if transpose:
        return transpose_cube(rawdetections),listdict
    else:
        return rawdetections,listdict

def clusterpoints_meanshift(importname = 'multidetect_try.json',export = True,exportname = 'clustered.json'):
    with open(importname, 'r') as json_file:
        json_data = json_file.read()
    rawdict = json.loads(json_data)
    rawdetections,listdict = listdetections(rawdict)
    # sorted_list = sorted(rawdetections, key=lambda x: x[2])# sorted from smallest to biggest according to larger axes
    print('creating meanshift')
    ms = MeanShift(bandwidth=0.1)
    print('fitting points')
    ms.fit(rawdetections)
    print('generating labels')
    labels = ms.labels_
    print('generate cluster_centers')
    cluster_centers = ms.cluster_centers_
    confirmed = []
    vowels,consonnants = nmgen.makeletterlists()
    for i in cluster_centers:
        confirmed.append({'name':nmgen.makeaname(6,vowels,consonnants),'color':nmgen.makeacolor(),'instances[ID,detection]':[],'instances axes':[]})
    for i in range(len(rawdetections)):
        confirmed[labels[i]]['instances axes'].append(listdict[i]['axes'])
    if export:
        with open(exportname,'w') as jsonout:
            jsonout.write(json.dumps(confirmed))
    return confirmed

def find_bounds_dict(clustered_dictionary):
    detectionpoints = []
    for multipledetect in clustered_dictionary:
        for instance in multipledetect['axes']:
            for axe in instance:
                for point in axe:
                    detectionpoints.append(point)
    return find_bounds(detectionpoints)
 
# def cluster_1st(importname,exportname,export = True,megalith=0.8,axe_detail=0.4,core_fraction = 0.25,min_confirmations = 5):
#     '''aaloa 0.3211218891790367 0.27684637920980454 4.223035087643648 6
# vloci 0.4601285667025782 0.38193560475093596 3.582462859520657 5
# ailoo 0.42970694617175687 0.4319624353525562 2.9836220967220086 4
# mvefs 0.9477736182526602 0.4373537661163098 4.7617503841750555 4'''
#     def listdetections(rawdict,megalith,max_oblateness = 3.5):
#         rawdetections = [] # the rawdetections contain [centroid x, centroid y, len max axis,len min axis]
#         listdict = []
#         cropbound = find_bounds(rawdict[0]['cropping coordinates'])
#         for photo in range(1,len(rawdict)):
#             for detectnum in range(len(rawdict[photo]['centroids'])):
#                 centroid = rawdict[photo]['centroids'][detectnum]
#                 if all([centroid[0]>=cropbound[0],centroid[0]<=cropbound[1],centroid[1]>=cropbound[2],centroid[1]<=cropbound[3]]):
#                     axes = rawdict[photo]['axes'][detectnum]
#                     lens = []
#                     for axe in axes:
#                         lens.append(pdist([axe[0],axe[1]])[0])
#                     if min(lens)>0:
#                         if max(lens)< megalith*pdist([rawdict[0]['cropping coordinates'][0],rawdict[0]['cropping coordinates'][1]])[0]:
#                             if max(lens)/min(lens)< max_oblateness:
#                                 rawdetections.append(rawdict[photo]['centroids'][detectnum])
#                                 rawdetections[-1].append(max(lens))
#                                 rawdetections[-1].append(min(lens))
#                                 listdict.append({'ID':rawdict[photo]['ID'],'centroids':rawdict[photo]['centroids'][detectnum],'axes':rawdict[photo]['axes'][detectnum]})
#         return rawdetections,listdict
    
#     with open(importname, 'r') as json_file:
#         json_data = json_file.read()
#     rawdict = json.loads(json_data)
#     cube,listdict = listdetections(rawdict,megalith)
#     median_stone_len = sorted(cube.copy(), key=lambda x: x[2])[round(len(cube)/2)][2]
#     thresholdaxe = median_stone_len*axe_detail
#     core_fraction *=median_stone_len
#     confirmed = []
#     vowels,consonnants = nmgen.makeletterlists()
#     tree = KDTree([x[:2] for x in cube])
#     spent = []
#     for index in range(len(listdict)):
#         if index not in spent:
#             ingroups = [index]
#             chosenone = cube[index]
#             index_neighbours = tree.query_ball_point(chosenone[:2],float(core_fraction))
#             for onenbr in index_neighbours:
#                 otherchosen = cube[onenbr]
#                 if all([chosenone[2]-thresholdaxe<=otherchosen[2]<=chosenone[2]+thresholdaxe,chosenone[3]-thresholdaxe<=otherchosen[3]<=chosenone[3]+thresholdaxe,onenbr not in ingroups,onenbr not in spent]):
#                     ingroups.append(onenbr)
#             if len(ingroups)>=min_confirmations:
#                 spent.append(index)
#                 confirmed.append({'name':nmgen.makeaname(6,vowels,consonnants),'color':nmgen.makeacolor(),'centroids':[],'axes':[],'axes lenghts':[]})
#                 for ingroup in ingroups:
#                     spent.append(ingroup)
#                     confirmed[-1]['axes'].append(listdict[ingroup]['axes'])
#                     confirmed[-1]['axes lenghts'].append(cube[ingroup][-2:])
#                     confirmed[-1]['centroids'].append(cube[ingroup][:2])
#     if export:
#         with open(exportname,'w') as jsonout:
#             jsonout.write(json.dumps(confirmed))
#     return confirmed
# --------------------------------------------------------------------------------------------------------

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
    for centroid in range(len(detection['centroids'])):
        neighbour = tree.query_ball_point(detection['centroids'][centroid],detection['axes lenghts'][centroid][1]/2)
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

def show_superimposed(canvas,clusteredict,superimposed):
    localized = canvas.copy()
    fullbounds = find_bounds_dict(clusteredict)
    detailbounds = find_bounds_dict(superimposed)
    p1 = [round(((detailbounds[0]-fullbounds[0])/(fullbounds[1]-fullbounds[0]))*canvas.shape[1]),canvas.shape[0]-round(((detailbounds[2]-fullbounds[2])/(fullbounds[3]-fullbounds[2]))*canvas.shape[0])]
    p2 = [round(((detailbounds[1]-fullbounds[0])/(fullbounds[1]-fullbounds[0]))*canvas.shape[1]),canvas.shape[0]-round(((detailbounds[3]-fullbounds[2])/(fullbounds[3]-fullbounds[2]))*canvas.shape[0])]
    cv2.rectangle(localized,p1,p2,[255,255,255],2)
    cv2.imshow('colored',localized)
    cv2.imshow('superimposed',paint_filteredict(superimposed))
    for single in superimposed:
        cv2.imshow(single['name'],paint_filteredict([single],definition = 300))
    cv2.waitKey(10)
    return

def compile_training_set(clean_path ='clustered.json', training_dict_name='training_dict.json'):
    with open(clean_path, 'r') as json_file:
        json_data = json_file.read()
    clusteredict = json.loads(json_data)
    canvas = paint_filteredict(clusteredict)

    allcentroids,groups = groupcentroids(clusteredict)
    tree = KDTree(allcentroids)
    training_set = {'x':[],'y':[]}
    for i in range(len(clusteredict)):
        superimposed = find_superimposed(tree,clusteredict,i,groups)
        detection_features= gen_detection_features(superimposed)
        if len(detection_features['name'])>=2:
            show_superimposed(canvas,clusteredict,superimposed)
            if input('continue y/n? ')!='y':
                break
            else:
                for b in range(1, len(detection_features['name'])):
                    print(detection_features['name'][0],'vs',detection_features['name'][b])
                    y1 = input('1 erase '+detection_features['name'][0]+', 2 erase '+detection_features['name'][b]+', 3 merge them, 4 skip ')
                    training_y = int(y1)
                    training_x = feature_to_input(detection_features,b)
                    training_set['x'].append(training_x)
                    training_set['y'].append(training_y)
            cv2.destroyAllWindows()


    with open(training_dict_name,'r') as jsonin:
        jsondata = jsonin.read()
    training_old = json.loads(jsondata)

    for key in training_old.keys():
        training_old[key].extend(training_set[key])

    print('the training set has reached',len(training_old['y']),'examples')

    with open(training_dict_name,'w') as jsonout:
        jsonout.write(json.dumps(training_old))
    return

def cluster_2nd_cycle(clusteredict,model,acti_threshold = 0.8):
    allcentroids,groups = groupcentroids(clusteredict)
    tree = KDTree(allcentroids)
    inputlist = []
    nameslist = []
    clusterindex = []
    clean_dict = []
    for i in range(len(clusteredict)):
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
    return clean_dict

def paint_cleandict(clean_dict,max_height = 750,background_color = [28,26,23],flip = True, bounds ='auto'):
    def calculate_angle(point, centroid):
        dx = point[0] - centroid[0]
        dy = point[1] - centroid[1]
        return (math.atan2(dy, dx) + 2 * math.pi) % (2 * math.pi)

    def order_points(points):
        centroid = [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]
        points.sort(key=lambda p: calculate_angle(p, centroid))
        return points
    if bounds == 'auto':
        bounds = find_bounds_dict(clean_dict)
    factor= max_height/abs(bounds[3]-bounds[2])
    h = max_height
    w = round(abs(bounds[1]-bounds[0])*factor)
    canvas = np.full((h, w, 3), background_color, dtype=np.uint8)
    # ['name', 'color', 'centroids', 'axes', 'axes lenghts']
    for i in range(len(clean_dict)):
        linealpha = round(255/len(clean_dict[i]['axes']))
        for j in range(len(clean_dict[i]['axes'])):
            points = [clean_dict[i]['axes'][j][0][0],clean_dict[i]['axes'][j][0][1],clean_dict[i]['axes'][j][1][0],clean_dict[i]['axes'][j][1][1]]
            projected_points = []
            for point in points:
                projected= [round(((point[0]-bounds[0])/(bounds[1]-bounds[0]))*w),round(((point[1]-bounds[2])/(bounds[3]-bounds[2]))*h)]
                projected_points.append(projected)
            projected_points = order_points(projected_points)
            output_image = canvas.copy()
            quadripoints = np.array(projected_points,np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [quadripoints], clean_dict[i]['color'])
            canvas = cv2.addWeighted(output_image, (1 - linealpha / 255), canvas, (linealpha / 255), 0)
    if flip:
        return cv2.flip(canvas,0)
    else:
        return canvas
    
def import_clustering_model():
    with open('training_dict.json','r') as jsonin:
        jsondata = jsonin.read()
    training = json.loads(jsondata)
    model = RandomForestClassifier(n_estimators=100)
    X_train = training['x']
    y_train = training['y']
    model.fit(X_train, y_train)
    print('fitting complete')
    return model

# compile_training_set()
if __name__ == '__main__':
    with open('training_dict.json','r') as jsonin:
        jsondata = jsonin.read()
    training = json.loads(jsondata)

    with open('clustered.json', 'r') as json_file:
        json_data = json_file.read()
    clusteredict = json.loads(json_data)

    clean_dict = cluster_2nd_cycle(clusteredict,import_clustering_model())

    canvas = paint_cleandict(clean_dict)
