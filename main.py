import os
from common.util import alignment, face_ToTensor
from models.models import CreateModel
import torch
import cv2
import numpy as np
from src import get_loader, get_celeba_loader
from arguments.test_args import get_args

import pymysql

import pandas as pd
import time
import pickle

from scipy.spatial.distance import cdist
from numpy import linalg as la
import base64
from src import detect_faces
from PIL import Image
import csv

from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from ENVS import *



_lfw_landmarks = 'data/LFW.csv'
_lfw_images = 'data/peopleDevTest.txt'
_lfw_root = 'data/images/'
_lbpfaces_path = 'data/lbpfaces.npy'
meanface_path = 'data/meanImage.npy'
eigenVec_path = 'data/eigenVectors_new.npy'
weightVec_path = 'data/weightVectors_updated.npy'

celeba_root = 'data/img_align_celeba_png/'




args = get_args()



def init_database_tables(cursor):
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

    stmt = "SHOW TABLES LIKE 'faces_test'"
    cursor.execute(stmt)
    result = cursor.fetchone()
    if not result:
        sql = 'CREATE TABLE IF NOT EXISTS `faces_test`(' \
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

    # do Kmeans for features saved in DB
    # create db for kmeans centers
    sql_kmeans_centers = 'CREATE TABLE IF NOT EXISTS `kmeans_centers`(' \
                         '`kmeans_index` INT UNSIGNED,' \
                         '`feature` BLOB, ' \
                         'PRIMARY KEY ( `kmeans_index` )' \
                         ')ENGINE=InnoDB DEFAULT CHARSET=utf8;'

    sql_face_kmeans = 'CREATE TABLE IF NOT EXISTS `faces_kmeans`(' \
                      '`face_id` INT UNSIGNED AUTO_INCREMENT,' \
                      '`name` VARCHAR(100) NOT NULL,' \
                      '`image_name` VARCHAR(100) NOT NULL,' \
                      '`feature` BLOB, ' \
                      '`kmeans_index` INT UNSIGNED,' \
                      'PRIMARY KEY ( `face_id` ), ' \
                      'FOREIGN KEY(`kmeans_index`) REFERENCES kmeans_centers(`kmeans_index`)' \
                      ')ENGINE=InnoDB DEFAULT CHARSET=utf8;'

    sql_face_dbscan = 'CREATE TABLE IF NOT EXISTS `faces_dbscan`(' \
                      '`face_id` INT UNSIGNED AUTO_INCREMENT,' \
                      '`name` VARCHAR(100) NOT NULL,' \
                      '`image_name` VARCHAR(100) NOT NULL,' \
                      '`feature` BLOB, ' \
                      '`dbscan_index` INT UNSIGNED,' \
                      'PRIMARY KEY ( `face_id` )' \
                      ')ENGINE=InnoDB DEFAULT CHARSET=utf8;'

    try:
        cursor.execute(sql_kmeans_centers)
        cursor.execute(sql_face_kmeans)
        cursor.execute(sql_face_dbscan)
        cursor.connection.commit()

    except Exception as e:
        print(e)
        exit()



def save_features_to_baseline_db(cursor):
    features, labels, names = generate_celeba_features(netModel.backbone)

    label_dic = {}
    for i in range(features.shape[0]):
        feature = features[i]
        name = names[i]
        identity = labels[i]
        byte_feature = feature.tostring()

        if identity in label_dic:
            label_dic[identity] += 1
        else:
            label_dic[identity] = 1

        is_test = True if label_dic[identity] == 2 else False

        try:
            if is_test:
                res = cursor.execute('INSERT INTO faces_test (name, image_name, feature) VALUES(%s, %s, %s);',
                                 ([identity,  name, byte_feature]))
            else:
                res = cursor.execute('INSERT INTO faces (name, image_name, feature) VALUES(%s, %s, %s);',
                                     ([identity,  name, byte_feature]))
        except pymysql.ProgrammingError as e:
            print(e)
            continue
        cursor.connection.commit()

    print(features.shape)


def get_train_data_from_baseline_db(cursor):
    # retrevial features from baseline database
    start = time.process_time()
    features_saved = []
    identities = []
    image_names = []


    try:
        res = cursor.execute('select name, image_name, feature from faces')
        values = cursor.fetchall()
        for item in values:
            features_saved.append(np.frombuffer(item[2], dtype=np.float32))
            identities.append(item[0])
            image_names.append(item[1])

        features_saved = np.array(features_saved)
    except pymysql.ProgrammingError as e:
        print(e)
    tp = (time.process_time() - start)
    print('get train feature from db time: ', tp)

    import sys
    size_of_feature = sys.getsizeof(features_saved) / 1024 / 1024
    print('feature memory size in mb: ', size_of_feature)
    return identities, image_names, features_saved


def get_train_data_from_kmeans_db(cursor):
    # retrevial features from baseline database
    start = time.process_time()
    features_saved = []
    identities = []
    image_names = []
    kmeans_index = []

    try:
        res = cursor.execute('select name, image_name, feature, kmeans_index from faces_kmeans')
        values = cursor.fetchall()
        for item in values:
            features_saved.append(np.frombuffer(item[2], dtype=np.float64))
            identities.append(item[0])
            image_names.append(item[1])
            kmeans_index.append(int(item[3]))

        features_saved = np.array(features_saved).astype(np.float32)
    except pymysql.ProgrammingError as e:
        print(e)
    tp = (time.process_time() - start)
    print('get all train feature from kmeans db time: ', tp)

    import sys
    size_of_feature = sys.getsizeof(features_saved) / 1024 / 1024
    print('feature memory size in mb: ', size_of_feature)
    return identities, image_names, features_saved, kmeans_index




def get_test_data_from_test_db(cursor):
    start = time.process_time()
    features_saved = []
    identities = []
    image_names = []


    try:
        res = cursor.execute('select name, image_name, feature from faces_test')
        values = cursor.fetchall()
        for item in values:
            features_saved.append(np.frombuffer(item[2], dtype=np.float32))
            identities.append(item[0])
            image_names.append(item[1])

        features_saved = np.array(features_saved)
    except pymysql.ProgrammingError as e:
        print(e)
    tp = (time.process_time() - start)
    print('get test feature from db time: ', tp)


    return identities, image_names, features_saved

def get_kmeans_centers(cursor):
    centers = []

    try:
        res = cursor.execute('select kmeans_index, feature from kmeans_centers')
        values = cursor.fetchall()
        for item in values:
            centers.append(np.frombuffer(item[1], dtype=np.float32))

    except Exception as e:
        print(e)

    return np.array(centers).astype(np.float32)



def Kmeans_cluster_on_feature(features, names, identities):



    if os.path.exists('kmeans.pickle'):
        # Load model
        with open('kmeans.pickle', 'rb') as f:
            kmeans = pickle.load(f)
    else:

        start = time.process_time()

        kmeans = MiniBatchKMeans(n_clusters=100, random_state=0, batch_size=40000).fit(features)

        tp = (time.process_time() - start)

        print('kmeans time: ', tp)

        print("Kmeans cluster complete")

        # Save model
        with open('kmeans.pickle', 'wb') as f:
            pickle.dump(kmeans, f)

    kmeans_labels = kmeans.labels_
    kmeans_centers = kmeans.cluster_centers_

    feature_with_label = np.append(features, kmeans_labels.reshape(-1, 1), axis=1)
    new_sort_index = np.argsort(feature_with_label[:, -1])

    feature_with_clueter_label = feature_with_label[new_sort_index]
    names = np.array(names)[new_sort_index]
    identities = np.array(identities)[new_sort_index]

    # save kmeans_result to faces_kmeans db
    for i in range(100):
        center =  np.array(kmeans_centers[i])
        byte_center = center.tostring()

        try:
            res = cursor.execute('INSERT INTO kmeans_centers(kmeans_index, feature) VALUES(%s, %s);',
                                 ([int(i), byte_center]))

        except pymysql.ProgrammingError as e:
            print(e)
            continue
        cursor.connection.commit()

    for i in range(feature_with_clueter_label.shape[0]):
        feature = feature_with_clueter_label[i][:-1]
        name = names[i]
        identity = identities[i]
        cluster_label = feature_with_clueter_label[i][-1].astype(int)

        byte_feature = feature.tostring()

        try:
            res = cursor.execute('INSERT INTO faces_kmeans(name, image_name, feature, kmeans_index) VALUES(%s, %s, %s, %s);',
                                 ([identity, name, byte_feature,  int(cluster_label)]))
        except pymysql.ProgrammingError as e:
            print(e)
            continue
        cursor.connection.commit()


# Not in use since bad performance
def DBSCAN_cluster_on_feature(features, names, identities):



    if 0:
        # Load model
        with open('dbscan.pickle', 'rb') as f:
            dbscan = pickle.load(f)
    else:

        start = time.process_time()

        dbscan = SpectralClustering(n_clusters= 20).fit(features[:50000])

        tp = (time.process_time() - start)

        print('dbscan time: ', tp)

        print("dbscan cluster complete")

        # Save model
        with open('dbscan.pickle', 'wb') as f:
            pickle.dump(dbscan, f)

    dbscan_labels = dbscan.labels_
    feature_with_label = np.append(features, dbscan_labels.reshape(-1, 1), axis=1)
    new_sort_index = np.argsort(feature_with_label[:, -1])

    feature_with_clueter_label = feature_with_label[new_sort_index]
    names = np.array(names)[new_sort_index]
    identities = np.array(identities)[new_sort_index]

    # save dbscan result to faces_dbscan db
    for i in range(feature_with_clueter_label.shape[0]):
        feature = feature_with_clueter_label[i][:-1]
        name = names[i]
        identity = identities[i]
        cluster_label = feature_with_clueter_label[i][-1].astype(int)

        byte_feature = feature.tostring()

        try:
            res = cursor.execute('INSERT INTO faces_dbscan(name, image_name, feature, dbscan_index) VALUES(%s, %s, %s, %s);',
                                 ([identity, name, byte_feature,  int(cluster_label)]))
        except pymysql.ProgrammingError as e:
            print(e)
            continue
        cursor.connection.commit()









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


def generate_celeba_features(net):
    dataloader = get_celeba_loader(batch_size=256).dataloader
    features_total = torch.Tensor(np.zeros((202599, 512), dtype=np.float32)).to(args.device)
    names_total = []
    labels = []
    with torch.no_grad():
        bs_total = 0
        for index, (img, targets, names) in enumerate(dataloader):
            bs = len(targets)
            img = img.to(args.device)
            features = net(img)
            features_total[bs_total:bs_total + bs] = features
            names_total[bs_total:bs_total + bs] = names
            labels += targets
            bs_total += bs

        # assert bs_total == args.num_faces, print('Database should have {} faces!'.format(args.num_faces))

    features_total = features_total.cpu().detach().numpy()
    return features_total, labels, names_total


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

    #init database connecting cursor
    db = pymysql.connect(host='localhost', user=DB_USER_NAME, password=DB_PASSWORD, database=DB_DATABASE, charset='utf8')
    cursor = db.cursor()

    #init table
    init_database_tables(cursor)


    # check existing db status and save features to database
    res = cursor.execute('select count(*) from faces')
    value = cursor.fetchall()



    # Celeba dataset saved to dataset
    # save data to database if its empty

    if value[0][0] < 10000:
        save_features_to_baseline_db(cursor)

# ==============================================================================
    identities, image_names, features_saved = get_train_data_from_baseline_db(cursor)
    # do kmeans and saved ordered feature to new db table faces_kmeans

    #DBSCAN_cluster_on_feature(features_saved, image_names, identities)
    Kmeans_cluster_on_feature(features_saved, image_names, identities)




# ======================================================================================================================

    # get evaluation list


    test_identities, test_image_names, test_features  = get_test_data_from_test_db(cursor)
    sample_size = 1000
    top_k = 5


    num_kmeans_index_retrivial = 5


    print('==================================================')
    print('Baseline')
    #baseline
    start = time.process_time()

    good_predict = 0
    bad_predict = 0


    score = 1 - cdist(test_features[0:sample_size], features_saved, 'cosine')

    top_k_index =  np.fliplr(np.argsort(score, axis = 1))[:,0:top_k]
    idn = np.array(identities)
    baseline_predict_identity = idn[top_k_index]
    for i in range(sample_size):
       if test_identities[i] in list(baseline_predict_identity[i]):
           good_predict += 1
       else:
           bad_predict += 1

    print('accuracy: ', good_predict / (good_predict + bad_predict))


    tp = (time.process_time() - start)
    print('calculate distance time: ', tp)

    print('==================================================')
    print('Kmeans')
    #K-means

    index_category = []
    identities_kmeans, image_name_kmeans, features_kmeans, kmeans_index = get_train_data_from_kmeans_db(cursor)

    kmeans_index = np.array(kmeans_index)
    for i in range(100):
        idx = np.argwhere(kmeans_index == i)
        start = np.min(idx) + 1
        end = np.max(idx) + 1
        index_category.append({
            'kmeans_index': i,
            'start': start,
            'end': end
        })


    with open('kmeans_index_category.csv', 'w') as f:
        keys = index_category[0].keys()
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(index_category)

    kmeans_centers = get_kmeans_centers(cursor)

    start = time.process_time()

    def find_indexs_with_center(features_test, centers, num_of_returned_value):
        dist = cdist(features_test, centers)
        idn = np.argsort(dist, axis=1)[:, 0:num_of_returned_value]
        return idn


    sample_indexes = find_indexs_with_center(test_features[:sample_size], kmeans_centers, num_kmeans_index_retrivial)

    good_predict = 0
    bad_predict = 0

    kmeans_total_prediction_identity = []
    for i in range(sample_size):
        partial_index = np.argwhere(kmeans_index == sample_indexes[i][0]).reshape(-1)
        for k in range(1, num_kmeans_index_retrivial):
            partial_index = np.append(partial_index, np.argwhere(kmeans_index == sample_indexes[i][k]).reshape(-1))


        feature_to_retrivial_from =  features_kmeans[partial_index]
        identities_partial = np.array(identities_kmeans)[partial_index]

        score = 1 - cdist(test_features[i].reshape((1, -1)), feature_to_retrivial_from, 'cosine')

        top_k_index = np.fliplr(np.argsort(score, axis = 1))[:,0:top_k].reshape(-1)

        predict_identity = identities_partial[top_k_index]
        kmeans_total_prediction_identity.append(predict_identity)

        if test_identities[i] in list(predict_identity):
            good_predict += 1
        else:
            bad_predict += 1

    print('accuracy: ', good_predict / (good_predict + bad_predict))
    tp = (time.process_time() - start)
    print('Kmeans calculate distance time: ', tp)
