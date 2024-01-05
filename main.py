import math
import multiprocessing as mtp
import os
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from scipy.spatial import KDTree
import numpy as np
import random as rnd
from scipy.spatial.distance import pdist
import pandas as pd


from gradeland_lib import detection as dtct

from gradeland_lib import nvm_to_obj as nto
from gradeland_lib import tredtect as dtc3
from gradeland_lib import visualize as viz
from gradeland_lib import maskmkr as mkmk

workspace = 'empty'
photofolder = 'empty'
photoanal = 'empty'
nvmpath = 'empty'
nvmodel = 'empty'
meshwriter = 'empty'

def button_click(button_text, maxdim = 1000):
    
    def createnvm(photofolder,workspace):
        if photofolder == 'empty':
            messagebox.showwarning("Warning", "select the photo folder first!!")
        if workspace == 'empty':
            messagebox.showwarning("Warning", "adopt a workspace folder first!!")
        else:
            messagebox.showwarning("Warning", "This operation can last hours!")
            print("select nvm file name")
            global nvmpath
            nvmpath = filedialog.asksaveasfilename(defaultextension=".nvm", filetypes=[("NVM Files", "*.nvm")],initialdir=workspace)
            if os.path.basename(photofolder) != 'rotated':
                print("adjusting images orientation...")
                nto.rotate_images(photofolder)
            photofolder = os.path.join(photofolder,'rotated')
            print('orientation adjusted!')
            lonestring = '.\colmap automatic_reconstructor --image_path @dd_photofolder --workspace_path @dd_workspace --quality extreme \n .\colmap model_converter --input_path @dd_workspace/sparse/0 --output_path @dd_nvmodel --output_type nvm  '
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'colmap_path.txt'), 'r') as bp:
                cmpath = bp.read()
            # numbercores = int(math.ceil(mtp.cpu_count()/2))
            lonestring = lonestring.replace('@dd_photofolder',photofolder)
            lonestring = lonestring.replace('@dd_nvmodel',nvmpath)
            lonestring = lonestring.replace('@dd_workspace',workspace)
            # lonestring = lonestring.replace('@dd_numthreads',str(numbercores))
            commandlist = lonestring.split('\n')
            os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),cmpath))
            for i in range(len(commandlist)):
                print(i,commandlist[i])
                print(subprocess.call(commandlist[i],shell = True))
            # with open('colmap_launcher.bat', 'w') as bp:
            #     bp.write(lonestring)

    def export3D(workspace,photoanal):
        if workspace == 'empty':
            messagebox.showwarning("Warning", "adopt a workspace folder first!!")
            return
        if photoanal == 'empty':
            messagebox.showwarning("Warning", "select photo first!!")
            return
        if nvmpath == 'empty':
            messagebox.showwarning("Warning", "nvm file not found")
        else:
            treedphotopath = os.path.join(workspace,'3D_photos')
            if not os.path.exists(treedphotopath):
                os.makedirs(treedphotopath)
            exportpath = os.path.join(treedphotopath,os.path.basename(photoanal))
            if not os.path.exists(exportpath):
                os.makedirs(exportpath)
            finalfilename = os.path.join(exportpath,os.path.basename(photoanal))
            if not os.path.exists(finalfilename):
                shutil.copy(photoanal,finalfilename)
            nto.export_singlephoto(photoanal, nto.nvm_to_points(nvmpath,photofolder,' '),exportname = finalfilename,textures=True)
            return meshwriter

    def filterveg(photoanal,workspace):
        if photoanal == 'empty':
            messagebox.showwarning("Warning", "select photo first!!")
            return
        if workspace == 'empty':
            messagebox.showwarning("Warning", "adopt a workspace folder first!!")
            return
        else:
            img = cv2.imread(photoanal)
            landslide_colors,mask_colors = mkmk.selectcolors(img,maxlen = maxdim)
            mask = mkmk.genmask(img,landslide_colors,mask_colors)
            filteredphotopath = os.path.join(workspace,'filtered_photos')
            if not os.path.exists(filteredphotopath):
                os.makedirs(filteredphotopath)
            mkmk.mergemask(img,mask,exportname = os.path.join(filteredphotopath,os.path.basename(photoanal)))
            photoanal =  os.path.join(filteredphotopath,os.path.basename(photoanal))
        return photoanal
    
    def detect(workspace, photoanal):
                if photoanal == 'empty':
                    messagebox.showwarning("Warning", "select photo first!!")
                    return
                if workspace == 'empty':
                    messagebox.showwarning("Warning", "adopt a workspace folder first!!")
                    return
                img = cv2.imread(photoanal)
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                h, w = gray.shape[:2]
                print('finding edges...')
                dst = dtct.DoubleOtsu(cv2.equalizeHist(cv2.GaussianBlur(gray, (5, 5), 0)))
                detectedphotopath = os.path.join(workspace,'detected_photos')
                print('finding foreground and background...')
                surefg = dtct.FragAnalFg(dtct.DistanceBasedInside,cv2.threshold(dst,200,255,cv2.THRESH_BINARY)[1],img,gray,h,w)[0]
                surebg =  cv2.threshold(dst,0,255,cv2.THRESH_BINARY)[1]
                unknown = cv2.subtract(surebg,surefg)
                print('final elaboration')
                markers = cv2.connectedComponents(surefg, ltype=cv2.CV_32S)[1]
                markers = markers+1
                markers[unknown==255] = 0
                markers = cv2.watershed(img,markers)
                if not os.path.exists(detectedphotopath):
                    os.makedirs(detectedphotopath)
                photoanal = os.path.join(detectedphotopath,os.path.basename(photoanal))
                cv2.imwrite(photoanal,dtct.colordetect(markers.copy(),img, rerange= True))
                # dtct.exportGSD(markers,img, photoanal, centroidcolor = [197,97,255], min_area = 16)
                print('detection completed...')
                return

    def tredtect(photofolder, photoanal, workspace, nvmpath):
        nvmodel = nto.nvm_to_points(nvmpath,photofolder,' ')
        if photofolder == 'empty':
            messagebox.showwarning("Warning", "select the photo folder first!!")
            return
        if photoanal == 'empty':
            messagebox.showwarning("Warning", "select photo first!!")
            return
        if workspace == 'empty':
            messagebox.showwarning("Warning", "adopt a workspace folder first!!")
            return
        if nvmpath == 'empty':
            messagebox.showwarning("Warning", "nvm file not found")
            return
        messagebox.showwarning("instruction", "select the area to analyze by clicking the left button and dragging a rectangle. If one rectangle is not sufficient, press space bar to add another rectangle. Press enter to start the analysis")
        roicoord = dtc3.roiselect(photoanal, maxdim = maxdim)
        inimage,inimage_pixel = dtc3.points_inimage_doubleref(photoanal,nvmodel)
        if meshwriter!= 'empty':
            for i in reversed(range(len(inimage))):
                if i not in meshwriter.goodpoint_index:
                    del inimage[i]
                    del inimage_pixel[i]
        tree = KDTree(inimage_pixel)
        limitsize = [600,300,150]
        h,w = cv2.imread(photoanal).shape[:2]
        detect_files_path = os.path.join(workspace,'multidetect_files',os.path.basename(photoanal)[:-3])
        if not os.path.exists(detect_files_path):
            os.makedirs(detect_files_path)
        analyzed = []
        dxlist = []
        anal_mask = np.zeros([h,w],dtype =np.uint8)
        closer = max([h,w])
        for rectangle in roicoord:
            cv2.rectangle(anal_mask,[round(d) for d in rectangle[0]],[round(d) for d in rectangle[1]],color = 255,thickness=cv2.FILLED)
        for limit in limitsize:
            pointcloud = []
            while closer > limit:
                if len(analyzed)==0:
                    centroid_anal_rect = [round(rnd.random()*(w-1)),round(rnd.random()*(h-1))]
                    while anal_mask[centroid_anal_rect[1],centroid_anal_rect[0]]==0:
                        centroid_anal_rect = [round(rnd.random()*(w-1)),round(rnd.random()*(h-1))]
                else:
                    pointcloud.extend([[round(rnd.random()*(w-1)),round(rnd.random()*(h-1))] for _ in range(500)])
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
                # cv2.circle(anal_mask, centroid_anal_rect, 2, 100, 2)
                analyzed.append(centroid_anal_rect)
                dxlist.append(round(min([closer,limit])))
        print('the number of sub-areas to analyze is',len(analyzed),' the following operation might take several hours')
        for centroid_anal_rect in analyzed:
            dx = dxlist[analyzed.index(centroid_anal_rect)]
            dy = dx#round(dx* (rnd.random()*0.6+0.7))
            rectangle=[[max(0,centroid_anal_rect[0]-dx),max(0,centroid_anal_rect[1]-dy)],[min(w,centroid_anal_rect[0]+dx),min(h,centroid_anal_rect[1]+dy)]]
            modcoord = [dtc3.pixelto3d(rectangle[0],tree,inimage_pixel,inimage),dtc3.pixelto3d(rectangle[1],tree,inimage_pixel,inimage)]
            dtc3.multidetect(modcoord,photofolder,nvmodel,export = True,exportname =os.path.join(detect_files_path,str(modcoord[0])+'_'+str(modcoord[1])+'.json'))
        files_list = os.listdir(detect_files_path)
        for i in range(len(files_list)):
            files_list[i]= os.path.join(detect_files_path,files_list[i])
        print('performing first clustering')
        detectpath = os.path.join(workspace,'detection_maps')
        if not os.path.exists(detectpath):
            os.makedirs(detectpath)
        exportpath = os.path.join(detectpath,os.path.basename(photoanal)[:-3])
        if not os.path.exists(exportpath):
            os.makedirs(exportpath)
        # filteredict = dtc3.cluster_1st(files_list,exportname=os.path.join(exportpath,'clustered.json'))
        clean_dict = dtc3.gen2_cluster(detect_files_path,exportname=os.path.join(exportpath,'clean.json'))
        # clean_dict = dtc3.cluster_2nd(filteredict,dtc3.import_clustering_model(),exportname=os.path.join(exportpath,'clean.json'))
        h,w = cv2.imread(photoanal).shape[:2]
        # stones = viz.paint_cleandict(clean_dict,max_height=h,bounds=dtc3.find_bounds([[float(dim) for dim in x[0]['xyz'][:2]] for x in inimage]))
        stones = viz.paint_cleandict(clean_dict,max_height=h,bounds='auto')
        retransposed = viz.retranspose_detections(photoanal,clean_dict,inimage,inimage_pixel)
        gsd_chart, detectionlist = viz.GSD_chart(clean_dict,900,1200)
        cv2.imwrite(os.path.join(exportpath,os.path.basename(photoanal)),stones)
        #cv2.imwrite(os.path.join(exportpath,os.path.basename(photoanal)[:-4]+'trn.jpg'),retransposed)
        cv2.imwrite(os.path.join(exportpath,os.path.basename(photoanal)[:-4]+'GSD_chart.jpg'),gsd_chart)
        excel = {'detection number':[x for x in range(len(detectionlist))], 'characteristic radius [cm]':[x[1] for x in detectionlist],'smallest possible characteristic radius [cm]':[x[0] for x in detectionlist],'largest possible characteristic radius [cm]':[x[2] for x in detectionlist]}
        df = pd.DataFrame(excel)
        print('exporting to excel file')
        df.to_excel(os.path.join(exportpath,'GSD.xlsx'),index=False)
        print('3D detection complete')
        return

    print("Button clicked:", button_text)
    if button_text=="create workspace":
        global workspace
        workspace = filedialog.askdirectory()
    if button_text=="select photo folder":
        global photofolder
        photofolder = filedialog.askdirectory()
        if os.path.exists(os.path.join(photofolder,'rotated')):
            photofolder = os.path.join(photofolder,'rotated')
    if button_text=='select photo':
        global photoanal
        photoanal = filedialog.askopenfilename(initialdir=photofolder)
    if button_text=="select nvm model":
        global nvmpath
        nvmpath = filedialog.askopenfilename(initialdir=workspace)
    if button_text=="create nvm model":
        createnvm(photofolder,workspace)
    if button_text=="export 3D photo":
        export3D(workspace,photoanal)
    if button_text=="filter vegetation":
        photoanal = filterveg(photoanal,workspace)
    # if button_text=="perform detection":
    #     detect(workspace, photoanal)
    if button_text=="perform 3D detection":
        global nvmodel
        tredtect(photofolder, photoanal, workspace, nvmpath)
        

if __name__ == '__main__':
    root = tk.Tk()
    button_labels = ["create workspace","select photo folder", "select nvm model", "create nvm model","select photo", "filter vegetation","export 3D photo", "perform 3D detection"]

    row = 0
    column = 0
    for label in button_labels:
        button = tk.Button(root, text=label, command=lambda label=label: button_click(label))
        button.grid(row=row, column=column, padx=10, pady=10)
        column += 1
        if column > 1:
            column = 0
            row += 1
    root.mainloop()