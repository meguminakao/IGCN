import numpy as np
import pickle
import threading
import queue
import os
import copy
from skimage import io,transform

import cv2
from render import *
from utils import *

class DataFetcher(threading.Thread):

    def __init__(self, file_train, organ):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.queue = queue.Queue(64)
        self.pkl_list = []

        if not file_train == "":
            with open(file_train, 'r') as f:
                while (True):
                    line = f.readline().strip()
                    if not line:
                        break
                    
                    path = line + organ + ".pickle"                    
                    if os.path.exists("../Data/" + path):
                        self.pkl_list.append(path)
                    
        self.index = 0
        self.number = len(self.pkl_list)
        np.random.shuffle(self.pkl_list)
        
        # Settings
        filename = os.path.basename(self.pkl_list[0])
        filename = filename.replace(".pickle", "")
        
    def load_data(self, idx):

        pkl_path = "../Data/" + self.pkl_list[idx]
        adj, features, labels, trans_vec, face_norm, face, rmax, projM = pickle.load(open(pkl_path, 'rb'))

        img_path = pkl_path.replace(pkl_path.split("/")[-1],"DRR_0000.bmp")
        img_path = img_path.replace("Pickle/","DRR/")
        img = io.imread(img_path)
        img = img.astype('float32')
        
        caseID = pkl_path.split('/')[3]
        caseID = caseID.replace("case", "")

        if "3D-CT" in pkl_path:
            phaseID = "00"
        else:
            phaseID = pkl_path.split('/')[4]

        img = img[20:480, 20:620]
        img = cv2.copyMakeBorder(img, 20, 160, 20, 20, cv2.BORDER_CONSTANT, (0,0,0))

        gray = img[:,:,0]

        features = (features + trans_vec) / rmax
        labels = (labels + trans_vec) / rmax

        center = np.mean(features, 0)
        deform = labels - features

        n = len(features)
        shapes = np.zeros((n, 3))
        for i in range(n):
            shapes[i] = features[i] - center

        ipos = np.zeros((n, 2))
        for i in range(n):
            coord = np.insert(np.transpose([features[:,0], features[:,2], features[:,1]])[i], 3, 1)
            P = projM * coord
            ipos[i] = [640-(P[2, 1]+1)*320, (P[0, 0]+1)*320]

        # render initial shape
        img_label = render_multi(features, labels, face, deform, True)
        img_label = np.array(img_label, dtype=np.float32)

        img_init = render(features, face, deform, False)
        img_init = img_init[:,:,0]
        img_init = np.array(img_init / img_init.max() * 255, dtype=np.float32)

        shift = 122.0
        constant = np.full((640, 640), shift)
        img = np.stack((img_init, constant, gray), axis = 2)

        return adj, features, labels, trans_vec, face_norm, face, rmax, projM, img, img_label, shapes, ipos, caseID, phaseID


    def load_deformation(self, num, filepath):

        with open(filepath, "r", encoding='utf-8') as f:
            #get GT
            coord = []
            line = f.readlines()
            for i, rows in enumerate(line): 
                if i in range(13, num+13): 
                    rows = rows.split() 
                    coord.append(rows[0:3])

            return np.array(coord, dtype='float')

    def run(self):
        while self.index < 90000000 and not self.stopped:
            
            self.queue.put(self.load_data(self.index % self.number))
            self.index += 1

            if self.index % self.number == 0: 
                np.random.shuffle(self.pkl_list)
                self.pkl_list = copy.deepcopy(self.pkl_list)


    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()
