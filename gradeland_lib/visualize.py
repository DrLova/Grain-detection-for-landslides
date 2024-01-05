import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import json
from gradeland_lib import tredtect as dtc3
from gradeland_lib import AIfilter as ai

def calculate_angle(point, centroid):
    dx = point[0] - centroid[0]
    dy = point[1] - centroid[1]
    return (math.atan2(dy, dx) + 2 * math.pi) % (2 * math.pi)

def order_points(points):
    centroid = [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]
    points.sort(key=lambda p: calculate_angle(p, centroid))
    return points

def gen_dimensions(bounds, max_height):
    factor= max_height/abs(bounds[3]-bounds[2])
    h = max_height
    w = round(abs(bounds[1]-bounds[0])*factor)
    return h,w,bounds

def makecolorlist(lenght, palette= 'rainbow'):
    cmap = plt.get_cmap(palette)
    # Create a list of colors from the colormap
    col0 = [cmap(i / (lenght-1)) for i in range(lenght)]
    colors = []
    for color in col0:
        colors.append([round(x*255) for x in color[:3]])
    colors.reverse()
    return colors

def paint_filteredict(filteredict,definition = 800,linealpha = 200):
    bounds = dtc3.find_bounds_dict(filteredict)
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
    
def paint_cleandict(clean_dict,max_height = 750,background_color = [28,26,23],flip = True, bounds ='auto', color_code = True):
    print('elaborating output image')
    if bounds =='auto':
        h,w,bounds = gen_dimensions(dtc3.find_bounds_dict(clean_dict),max_height)
    else:
        h= max_height
        w = round(((bounds[1]-bounds[0])/(bounds[3]-bounds[2]))*h)
    canvas = np.full((h, w, 3), background_color, dtype=np.uint8)
    # ['name', 'color', 'centroids', 'axes', 'axes lenghts']
    if color_code:
        colorlist = makecolorlist(len(clean_dict))
        forsorting=[]
        for i in range(len(clean_dict)):
            mins = []
            for j in clean_dict[i]['axes lenghts']:
                mins.append(min(j))
            forsorting.append([i,sorted(mins)[round(len(mins)/2)]])
            # forsorting.append([i,sum(mins)/len(mins)])
        forsorting = sorted(forsorting, key=lambda x: x[1])
        increasing = [i[0] for i in forsorting]

    for i in range(len(clean_dict)):
        linealpha = round(255/len(clean_dict[i]['axes']))
        for j in range(len(clean_dict[i]['axes'])):
            points = [clean_dict[i]['axes'][j][0][0],clean_dict[i]['axes'][j][0][1],clean_dict[i]['axes'][j][1][0],clean_dict[i]['axes'][j][1][1]]
            projected_points = []
            for point in points:
                projected= [round(((point[0]-bounds[0])/(bounds[1]-bounds[0]))*w),round(((point[1]-bounds[2])/(bounds[3]-bounds[2]))*h)]
                projected_points.append(projected)
            projected_points = order_points(projected_points)
            smallbounds = dtc3.find_bounds(projected_points)
            projected_points = [[x[0]-smallbounds[0],x[1]-smallbounds[2]] for x in projected_points]
            quadripoints = np.array(projected_points,np.int32).reshape((-1, 1, 2))
            output_image = canvas[smallbounds[2]:smallbounds[3],smallbounds[0]:smallbounds[1]].copy()
            if color_code:
                cv2.fillPoly(canvas[smallbounds[2]:smallbounds[3],smallbounds[0]:smallbounds[1]], [quadripoints], colorlist[increasing.index(i)])
            else:
                cv2.fillPoly(canvas[smallbounds[2]:smallbounds[3],smallbounds[0]:smallbounds[1]], [quadripoints], clean_dict[i]['color'])
            canvas_rect = cv2.addWeighted(output_image, (1 - linealpha / 255), canvas[smallbounds[2]:smallbounds[3],smallbounds[0]:smallbounds[1]], (linealpha / 255), 0)
            if all([smallbounds[0]>+0,smallbounds[2]>=0,smallbounds[1]<=w,smallbounds[3]<=h,smallbounds[1]>smallbounds[0],smallbounds[3]>smallbounds[2]]):
                canvas[smallbounds[2]:smallbounds[3],smallbounds[0]:smallbounds[1]]= canvas_rect
    print('operation successful')
    if flip:
        return cv2.flip(canvas,0)
    else:
        return canvas
    
def retranspose_detections(photoanal,clast_dict, inimage,inimage_pixel):
    inimage_basic = []
    for i in inimage:
        inimage_basic.append(i[0]['xyz'][:2])

    tree = KDTree(inimage_basic)
    pixelcentroids = []
    pixelcolors = []
    pixelaxes = []
    for clast in clast_dict:
        for c in range(len(clast['centroids'])):
            pixelcolors.append(clast['color'])
            pixelcentroids.append(dtc3.pixelto3d(clast['centroids'][c],tree,inimage_basic,inimage_pixel))
            pixelaxes.append([[[],[]],[[],[]]])
            for axe in range(len(clast['axes'][c])):
                for tip in range(len(clast['axes'][c][axe])):
                    pixelaxes[-1][axe][tip]= [round(i) for i in dtc3.pixelto3d(clast['axes'][c][axe][tip],tree,inimage_basic,inimage_pixel)]

    bounds = ai.find_bounds(pixelcentroids)
    img = cv2.imread(photoanal)
    for c in range(len(pixelcentroids)):
        cv2.circle(img, [round(i) for i in pixelcentroids[c]], 0, pixelcolors[c], 2)
        for axe in range(2):
            cv2.line(img, pixelaxes[c][axe][0],pixelaxes[c][axe][1],pixelcolors[c],1)
    img = img[round(bounds[2]):round(bounds[3]),round(bounds[0]):round(bounds[1])]
    # cv2.imshow('snippet',img)
    # cv2.waitKey(0)
    return img
    
def GSD_chart_legacy(clean_dict,h,w,background_color = [28,26,23],curve_color = [0,80,252],marks_color = [235,217,190] ):

    def logscale(lenlist, marks):
        scaled_lenlist = []
        scaled_marks = []
        for i in range(len(lenlist)):
            scaled_lenlist.append([])
            for j in range(len(lenlist[i])):
                scaled_lenlist[-1].append(math.log10(lenlist[i][j]*factor))
        for i in range(len(marks)):
            scaled_marks.append(math.log10(marks[i]))
        return scaled_lenlist,scaled_marks

    def add_grid(chart,scaled_marks,min,max,thicks,marks_color,thintrans):
        for m in range(len(scaled_marks)):
            mark = scaled_marks[m]
            x = w-round(((mark-min)/(max-min))*w)
            if thicks[m]:
                cv2.line(chart, [x,0],[x,h],marks_color,1)
            else:
                semitrans = chart.copy()
                cv2.line(chart, [x,0],[x,h],marks_color,1)
                chart = cv2.addWeighted(semitrans, (1 - thintrans / 255), chart, (thintrans / 255), 0)
        for m in range(0,101,10):
            y = round((m/100)*h)
            if m == 50:
                cv2.line(chart, [0,y],[w,y],marks_color,1)
            else:
                semitrans = chart.copy()
                cv2.line(chart, [0,y],[w,y],marks_color,1)
                chart = cv2.addWeighted(semitrans, (1 - thintrans / 255), chart, (thintrans / 255), 0)
        return chart
    
    def logarithmic_marks(min,max,step = 1):
        factor = 1
        while min<1:
            factor *=10
            min*=10
            max *=10
        mark = step
        marks = []
        thicks = []
        divisor = 10
        while mark < max:
            if mark%divisor ==0:
                marks.append(mark)
                thicks.append(True)
                step *=10
                divisor*=10
                mark+=step
            else:
                marks.append(mark)
                thicks.append(False)
                mark+=step

        for i in range(1,len(marks)):
            if marks[i]>=min:
                return marks[i-1:],factor,thicks
    
    thintrans = 127
    chart = np.full((h, w, 3), background_color, dtype=np.uint8)
    
    lenlist = detection_to_gsd(clean_dict)
    min = lenlist[0][0]
    max = lenlist[-1][2]
    marks,factor,thicks = logarithmic_marks(min,max)
    lenlist.reverse() #move below alongside others
    marks.reverse()
    thicks.reverse()
    original_marks = marks.copy()
    scaled_lenlist,scaled_marks = logscale(lenlist,marks)
    marks = [x/factor for x in marks]

    max = scaled_lenlist[0][2]
    min = scaled_marks[-1]

    polyline = []
    uncertainty = []

    for l in range(len(scaled_lenlist)):
        measure = 1-((scaled_lenlist[l][1]-min)/(max-min))
        x = round(measure*w)
        y = h-round((1-(l/len(scaled_lenlist)))*h)
        if [x,y] not in polyline:
            polyline.append([x,y])
            uncertainty.append([])
        for ind in [0,2]:
            uncertainty[-1].append([round((1-((scaled_lenlist[l][ind]-min)/(max-min)))*w),y])
    
    uncertain_color = [255-curve_color[i] for i in range(3)]
    for superimposed_lines in uncertainty:
        linealpha = round(510/len(superimposed_lines))
        for pair in range(0,len(superimposed_lines),2):
            temp_out = chart.copy()
            cv2.line(chart, superimposed_lines[pair],superimposed_lines[pair+1],uncertain_color,1)
            chart = cv2.addWeighted(temp_out, (1 - linealpha / 255), chart, (linealpha / 255), 0)

    chart = add_grid(chart,scaled_marks,min,max,thicks,marks_color,thintrans)

    cv2.polylines(chart, [np.array(polyline)], isClosed=False, color=curve_color, thickness=2)
    frame = 10
    margins = round(h/20)
    chart=cv2.copyMakeBorder(chart,top=2,bottom=2,left=2,right=2,borderType=cv2.BORDER_CONSTANT,value=marks_color)
    chart=cv2.copyMakeBorder(chart,top=frame,bottom=frame,left=frame,right=frame,borderType=cv2.BORDER_CONSTANT,value=background_color)
    chart=cv2.copyMakeBorder(chart,top=0,bottom=margins,left=margins,right=0,borderType=cv2.BORDER_CONSTANT,value=background_color)

    for m in range(0,101,10):
        y = round((m/100)*h)
        chart = cv2.putText(chart, str(m), (2,(h-y)+round(margins*0.5)), cv2.FONT_HERSHEY_SIMPLEX, h/1000, marks_color, 2, cv2.LINE_AA)

    for m in range(len(scaled_marks)):
        mark = scaled_marks[m]
        if original_marks[m] in [5,50,500,5000,10,100,1000,10000]:
            x = w-round(((mark-min)/(max-min))*w)
            # cv2.circle(chart,(x+frame+margins,h+frame),1,marks_color,1)
            chart = cv2.putText(chart, str(int(marks[m])), (x+frame+margins,h+margins), cv2.FONT_HERSHEY_SIMPLEX, h/1000, marks_color, 2, cv2.LINE_AA)
    
    chart=cv2.copyMakeBorder(chart,top=0,bottom=margins,left=margins,right=0,borderType=cv2.BORDER_CONSTANT,value=background_color)
    chart = cv2.putText(chart, 'shorter diameter [cm]', (round(w/2)+frame+margins,h+(2*margins)), cv2.FONT_HERSHEY_SIMPLEX, h/1000, marks_color, 2, cv2.LINE_AA)
    
    chart = cv2.rotate(chart, cv2.ROTATE_90_CLOCKWISE)
    cv2.putText(chart, 'percentage below size [%]', [round(chart.shape[1]/2),30], cv2.FONT_HERSHEY_SIMPLEX, h/1000, marks_color, 2, cv2.LINE_AA)
    chart= cv2.rotate(chart, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return chart, lenlist

def GSD_chart(clean_dict,h,w,background_color = [28,26,23],curve_color = [0,80,252],marks_color = [235,217,190], second_curve = 'none' ):

    def logscale(lenlist, marks):
        scaled_lenlist = []
        scaled_marks = []
        for i in range(len(lenlist)):
            scaled_lenlist.append([])
            for j in range(len(lenlist[i])):
                scaled_lenlist[-1].append(math.log10(lenlist[i][j]*factor))
        for i in range(len(marks)):
            scaled_marks.append(math.log10(marks[i]))
        return scaled_lenlist,scaled_marks

    def add_grid(chart,scaled_marks,min,max,thicks,marks_color,thintrans):
        for m in range(len(scaled_marks)):
            mark = scaled_marks[m]
            x = w-round(((mark-min)/(max-min))*w)
            if thicks[m]:
                cv2.line(chart, [x,0],[x,h],marks_color,1)
            else:
                semitrans = chart.copy()
                cv2.line(chart, [x,0],[x,h],marks_color,1)
                chart = cv2.addWeighted(semitrans, (1 - thintrans / 255), chart, (thintrans / 255), 0)
        for m in range(0,101,10):
            y = round((m/100)*h)
            if m == 50:
                cv2.line(chart, [0,y],[w,y],marks_color,1)
            else:
                semitrans = chart.copy()
                cv2.line(chart, [0,y],[w,y],marks_color,1)
                chart = cv2.addWeighted(semitrans, (1 - thintrans / 255), chart, (thintrans / 255), 0)
        return chart
    
    def logarithmic_marks(min,max,step = 1):
        factor = 1
        while min<1:
            factor *=10
            min*=10
            max *=10
        mark = step
        marks = []
        thicks = []
        divisor = 10
        while mark < max:
            if mark%divisor ==0:
                marks.append(mark)
                thicks.append(True)
                step *=10
                divisor*=10
                mark+=step
            else:
                marks.append(mark)
                thicks.append(False)
                mark+=step

        for i in range(1,len(marks)):
            if marks[i]>=min:
                return marks[i-1:],factor,thicks
            
    def makepolypoints(scaled_lenlist, min, max):
        polyline = []
        uncertainty = []

        for l in range(len(scaled_lenlist)):
            measure = 1-((scaled_lenlist[l][1]-min)/(max-min))
            x = round(measure*w)
            y = h-round((1-(l/len(scaled_lenlist)))*h)
            if [x,y] not in polyline:
                polyline.append([x,y])
                uncertainty.append([])
            for ind in [0,2]:
                uncertainty[-1].append([round((1-((scaled_lenlist[l][ind]-min)/(max-min)))*w),y])
        return polyline,uncertainty
    
    thintrans = 127
    chart = np.full((h, w, 3), background_color, dtype=np.uint8)
    
    lenlist = detection_to_gsd(clean_dict)
    min = lenlist[0][0]
    max = lenlist[-1][2]
    marks,factor,thicks = logarithmic_marks(min,max)
    if second_curve != 'none':
        second = detection_to_gsd(second_curve)
        second.reverse()
        scaled_second,_=logscale(second,marks)

    lenlist.reverse() 
    marks.reverse()
    thicks.reverse()
    original_marks = marks.copy()
    scaled_lenlist,scaled_marks = logscale(lenlist,marks)
    marks = [x/factor for x in marks]
    max = scaled_lenlist[0][2]
    min = scaled_marks[-1]
    polyline, uncertainty = makepolypoints(scaled_lenlist,min,max)
    if second_curve != 'none':
        seconpoly, seconduncertain = makepolypoints(scaled_second,min,max)
        cv2.polylines(chart, [np.array(seconpoly)], isClosed=False, color=[0,0,0], thickness=2)

    uncertain_color = [255-curve_color[i] for i in range(3)]
    for superimposed_lines in uncertainty:
        linealpha = round(510/len(superimposed_lines))
        for pair in range(0,len(superimposed_lines),2):
            temp_out = chart.copy()
            cv2.line(chart, superimposed_lines[pair],superimposed_lines[pair+1],uncertain_color,1)
            chart = cv2.addWeighted(temp_out, (1 - linealpha / 255), chart, (linealpha / 255), 0)

    chart = add_grid(chart,scaled_marks,min,max,thicks,marks_color,thintrans)

    cv2.polylines(chart, [np.array(polyline)], isClosed=False, color=curve_color, thickness=2)
    frame = 10
    margins = round(h/20)
    chart=cv2.copyMakeBorder(chart,top=2,bottom=2,left=2,right=2,borderType=cv2.BORDER_CONSTANT,value=marks_color)
    chart=cv2.copyMakeBorder(chart,top=frame,bottom=frame,left=frame,right=frame,borderType=cv2.BORDER_CONSTANT,value=background_color)
    chart=cv2.copyMakeBorder(chart,top=0,bottom=margins,left=margins,right=0,borderType=cv2.BORDER_CONSTANT,value=background_color)

    for m in range(0,101,10):
        y = round((m/100)*h)
        chart = cv2.putText(chart, str(m), (2,(h-y)+round(margins*0.5)), cv2.FONT_HERSHEY_SIMPLEX, h/1000, marks_color, 2, cv2.LINE_AA)

    for m in range(len(scaled_marks)):
        mark = scaled_marks[m]
        if original_marks[m] in [5,50,500,5000,10,100,1000,10000]:
            x = w-round(((mark-min)/(max-min))*w)
            # cv2.circle(chart,(x+frame+margins,h+frame),1,marks_color,1)
            chart = cv2.putText(chart, str(int(marks[m])), (x+frame+margins,h+margins), cv2.FONT_HERSHEY_SIMPLEX, h/1000, marks_color, 2, cv2.LINE_AA)
    
    chart=cv2.copyMakeBorder(chart,top=0,bottom=margins,left=margins,right=0,borderType=cv2.BORDER_CONSTANT,value=background_color)
    chart = cv2.putText(chart, 'shorter diameter [cm]', (round(w/2)+frame+margins,h+(2*margins)), cv2.FONT_HERSHEY_SIMPLEX, h/1000, marks_color, 2, cv2.LINE_AA)
    
    chart = cv2.rotate(chart, cv2.ROTATE_90_CLOCKWISE)
    cv2.putText(chart, 'percentage below size [%]', [round(chart.shape[1]/2),30], cv2.FONT_HERSHEY_SIMPLEX, h/1000, marks_color, 2, cv2.LINE_AA)
    chart= cv2.rotate(chart, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return chart, lenlist

def detection_to_gsd(clean_dict):
    lenlist = []
    for detection in clean_dict:
        charactlen = []
        for singlestrike in detection['axes lenghts']:
            charactlen.append(min(singlestrike))
        lenlist.append([])
        for x in [0.25,0.5,0.75]:
            lenlist[-1].append(100*sorted(charactlen)[int(round(len(charactlen)*x))])
    lenlist = sorted(lenlist, key=lambda x: x[1])
    return lenlist


if __name__ == '__main__':
    with open(r'C:\Users\lovam\Documents\sklgp_lit\3d_drone\sfm_photogrammetry\wenchuan_debris\detection_maps\DJI_20230715162937_0060_V\clean.json','r') as jsonin:
        jsondata = jsonin.read()
    clean_dict = json.loads(jsondata)
    canvas = paint_cleandict(clean_dict)

    h,w = [900,1200]
    chart,_ = GSD_chart(clean_dict,h,w)

    cv2.imshow('prova',chart)
    cv2.imwrite(r'C:\Users\lovam\Documents\sklgp_lit\3d_drone\sfm_photogrammetry\wenchuan_debris\detection_maps\DJI_20230715162937_0060_V\cacca_nera_puzzolente.jpg', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
