import os
from common.util import alignment, face_ToTensor
from models.models import CreateModel
import torch
import cv2
import numpy as np
from src import get_loader
from arguments.test_args import get_args

import pymysql

import pandas as pd
import time


from scipy.spatial.distance import cdist
from numpy import linalg as la
import base64
from src import detect_faces
from PIL import Image
import csv

_lfw_landmarks = 'data/LFW.csv'
_lfw_images = 'data/peopleDevTest.txt'
_lfw_root = 'data/images/'
_lbpfaces_path = 'data/lbpfaces.npy'
meanface_path = 'data/meanImage.npy'
eigenVec_path = 'data/eigenVectors_new.npy'
weightVec_path = 'data/weightVectors_updated.npy'

MODEL_PATH = 'models/20_softmax.pth'

args = get_args()


def get_landmarks(image):
    # bb: [x0,y0,x1,y1]
    bounding_boxes, landmarks = detect_faces(image)
    if len(bounding_boxes) > 1:
        # pick the face closed to the center
        center = np.asarray(np.asarray(image).shape[:2]) / 2.0
        ys = bounding_boxes[:, :4][:, [1, 3]].mean(axis=1).reshape(-1, 1)
        xs = bounding_boxes[:, :4][:, [0, 2]].mean(axis=1).reshape(-1, 1)
        coord = np.hstack((ys, xs))
        dist = ((coord - center) ** 2).sum(axis=1)
        index = np.argmin(dist, axis=0)
        landmarks = landmarks[index]
    else:
        landmarks = landmarks[0]
    landmarks = landmarks.reshape(2, 5).T
    landmarks = landmarks.reshape(-1)

    return landmarks


def get_alignedface(image, landmarks):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    face = alignment(img, landmarks.reshape(-1, 2))
    return face

def get_feature_of_image(net, file_path):
    image = Image.open(file_path)
    landmarks = get_landmarks(image)
    aligned_face = get_alignedface(image, landmarks)
    feature =  net(face_ToTensor(aligned_face).to(args.device).view([1, 3, 112, 96]), is_feature = True)
    return feature.cpu().detach().numpy()





#
def generate_features(net):
    dataloader = get_loader(batch_size=128).dataloader
    features_total = torch.Tensor(np.zeros((args.num_faces, 512), dtype=np.float32)).to(args.device)
    names_total = []
    labels = torch.Tensor(np.zeros((args.num_faces, 1), dtype=np.float32)).to(args.device)
    with torch.no_grad():
        bs_total = 0
        for index, (img, targets, names) in enumerate(dataloader):
            bs = len(targets)
            img = img.to(args.device)
            features = net(img)
            features_total[bs_total:bs_total + bs] = features
            names_total[bs_total:bs_total + bs] = names
            labels[bs_total:bs_total + bs] = targets
            bs_total += bs
        # assert bs_total == args.num_faces, print('Database should have {} faces!'.format(args.num_faces))


    features_total = features_total.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return features_total, labels, names_total

if __name__ == '__main__':
    #init model
    netModel = CreateModel(args)
    netModel.backbone.load_state_dict(torch.load(MODEL_PATH))

    #init database
    db = pymysql.connect(host='localhost', user='root', password='123456', database='face', charset='utf8')
    cursor = db.cursor()

    #init table
    stmt = "SHOW TABLES LIKE 'faces'"
    cursor.execute(stmt)
    result = cursor.fetchone()
    if not result:
        sql = 'CREATE TABLE IF NOT EXISTS `faces`(' \
              '`face_id` INT UNSIGNED AUTO_INCREMENT,' \
              '`name` VARCHAR(100) NOT NULL,' \
              '`image_name` VARCHAR(100) NOT NULL,' \
              '`feature` BLOB, ' \
              'PRIMARY KEY ( `face_id` )' \
              ')ENGINE=InnoDB DEFAULT CHARSET=utf8;'
        try:
            cursor.execute(sql)
            cursor.connection.commit()
        except Exception as e:
            print(e)
            exit()


    # check existing db status and save features to database
    res = cursor.execute('select count(*) from faces')
    value = cursor.fetchall()
    if value[0][0] == 0:

        features, labels, names = generate_features(netModel.backbone)
        for i in range(features.shape[0]):
            feature = features[i]
            name = names[i]
            identity, image_name = name.split('/')

            byte_feature = feature.tostring()
            try:
                res = cursor.execute('INSERT INTO faces(name, image_name, feature) VALUES(%s, %s, %s);',([identity, image_name, byte_feature]))
            except pymysql.ProgrammingError as e:
                print(e[1])
                continue
            cursor.connection.commit()

    # retrevial features from database
    start = time.process_time()

    try:
        res = cursor.execute('select feature from faces')
        values = cursor.fetchall()
        features = []
        for item in values:
            features.append(np.frombuffer(item[0], dtype=np.float32))
        features = np.array(features)
    except pymysql.ProgrammingError as e:
        print(e[1])
    tp = (time.process_time() - start)
    print('get test feature from db time: ', tp)




    # get evaluation list
    evaluation_list = pd.read_csv('data/evaluationImages.csv')
    print(evaluation_list.head())

    evaluation_features = []
    start = time.process_time()
    for index, row in evaluation_list.iterrows():
        test_path = 'data/images/' + evaluation_list.iloc[0].IMAGE_NAME
        feature = get_feature_of_image(netModel, test_path)
        evaluation_features.append(feature)
        break
    tp = (time.process_time() - start)
    print('extract feature time: ', tp)

    start = time.process_time()
    for feature in evaluation_features:
       score =  1 - cdist(feature, features, 'cosine')
    tp = (time.process_time() - start)
    print('calculate distance time: ', tp)

