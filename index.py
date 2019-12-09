import numpy as np

import pymysql

import time

from scipy.spatial.distance import cdist
import csv
from flask import Flask, jsonify, request,  send_file
from flask_cors import CORS
from io import BytesIO
from ENVS import *


app = Flask(__name__)
CORS(app)


def get_kmeans_categories():
    kmeans_category = {}
    with open('kmeans_index_category.csv') as f:
        dict_reader = csv.DictReader(f)
        for row in dict_reader:
            kmeans_category[int(row['kmeans_index'])] = {
                'start': row['start'],
                'end': row['end']
            }
    return kmeans_category




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
    # print('get train feature from db time: ', tp)

    import sys
    size_of_feature = sys.getsizeof(features_saved) / 1024 / 1024
    # print('feature memory size in mb: ', size_of_feature)
    return identities, image_names, features_saved, tp, size_of_feature


def get_train_data_from_kmeans_db(cursor, cluster_indexes):
    # retrevial features from baseline database
    start = time.process_time()
    features_saved = []
    identities = []
    image_names = []
    cluster_indexes = cluster_indexes.reshape(-1)
    for cluster_index in cluster_indexes:
        cluster_index = int(cluster_index)
        l =  kmeans_category[cluster_index]['start']
        r = kmeans_category[cluster_index]['end']
        try:
            res = cursor.execute('select name, image_name, feature from faces_kmeans where face_id between %s and %s',([l, r]))
            values = cursor.fetchall()
            for item in values:
                features_saved.append(np.frombuffer(item[2], dtype=np.float64))
                identities.append(item[0])
                image_names.append(item[1])


        except pymysql.ProgrammingError as e:
            print(e)
    features_saved = np.array(features_saved).astype(np.float32)

    tp = (time.process_time() - start)
    print('get all train feature from kmeans db time: ', tp)

    import sys
    size_of_feature = sys.getsizeof(features_saved) / 1024 / 1024
    print('feature memory size in mb: ', size_of_feature)
    return identities, image_names, features_saved, tp, size_of_feature


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


def reconize_feature_with_baseline(feature, label, num_of_result, cursor):
    identities, image_names, features_saved, tp, size_of_feature = get_train_data_from_baseline_db(cursor)

    start = time.process_time()
    feature = feature.reshape((1, -1))
    dist = cdist(feature, features_saved, 'cosine').reshape(-1)
    top_k_idx = np.argsort(dist)[0:num_of_result]
    scores = 1 - dist[top_k_idx]
    identities = np.array(identities)[top_k_idx]
    result = []
    for i in range(num_of_result):
        result.append({
            'identity': identities[i],
            'correct': label == identities[i],
            'score': scores[i],
            'image': image_names[top_k_idx[i]].replace(celeba_root, '')
        })
    time_recog = time.process_time() - start

    return {
        'result': result,
        'time_db': tp,
        'time_recog': time_recog,
        'mem': size_of_feature
    }


def find_indexs_with_center(features_test, centers, num_of_returned_value):
    if features_test.ndim == 1:
        features_test = features_test.resape((1, -1))

    dist = cdist(features_test, centers)
    idn = np.argsort(dist, axis=1)[:, 0:num_of_returned_value]
    return idn

def reconize_feature_with_kmeans(feature, label, num_of_result, num_of_kmeans_cluster, cursor, centers):
    cluster_indexes = find_indexs_with_center(feature, centers, num_of_kmeans_cluster)
    identities, image_names, features_saved, tp, size_of_feature = get_train_data_from_kmeans_db(cursor, cluster_indexes)


    start = time.process_time()

    feature = feature.reshape((1, -1))
    dist = cdist(feature, features_saved, 'cosine').reshape(-1)
    top_k_idx = np.argsort(dist)[0:num_of_result]
    scores = 1 - dist[top_k_idx]
    identities = np.array(identities)[top_k_idx]
    result = []
    for i in range(num_of_result):
        result.append({
            'identity': identities[i],
            'correct': label == identities[i],
            'score': scores[i],
            'image': image_names[top_k_idx[i]].replace(celeba_root, '')
        })
    time_recog = time.process_time() - start

    return {
        'result': result,
        'time_db': tp,
        'time_recog': time_recog,
        'mem': size_of_feature
    }

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






celeba_root = 'data/img_align_celeba_png/'
# init database connecting cursor
db = pymysql.connect(host='localhost', user=DB_USER_NAME, password=DB_PASSWORD, database=DB_DATABASE, charset='utf8')
cursor = db.cursor()

# ==============================================================================
# prepare test data
identities_test, image_names_test, features_test = get_test_data_from_test_db(cursor)
pure_name_list = [x.replace(celeba_root, '') for x in image_names_test]
np_pure = np.array(pure_name_list)

kmeans_category = get_kmeans_categories()
kmeans_centers = get_kmeans_centers(cursor)



# iden, image_name, features, t, mem = get_train_data_from_baseline_db(cursor)
# iden_k, image_name_k, features_k, kmeans_index, t_k, mem_k = get_train_data_from_kmeans_db(cursor, {'kmeans_index': 0, 'start': 1, 'end': 1903})



@app.route('/')
def hello_world():
    return jsonify({
        'success': True
    })

@app.route('/image/<name>', methods=['GET'])
def get_image_stream(name):
    name = celeba_root + name
    with open(name, 'rb') as f:
        return send_file(BytesIO(f.read()), attachment_filename=name, mimetype='image/png')

@app.route('/list', methods=['GET'])
def get_test_image_list():
    return jsonify({
        'success':True,
        'image_list': pure_name_list
    })

@app.route('/recognize/<name>', methods=['POST'])
def recognize_image(name):
    idx = np.where(np_pure == name)[0]
    feature = features_test[idx]
    label = identities_test[int(idx)]



    res_kmeans = reconize_feature_with_kmeans(feature, label, cursor=cursor, num_of_result= 10 , num_of_kmeans_cluster= 3, centers = kmeans_centers)
    res_base = reconize_feature_with_baseline(feature, label, cursor=cursor, num_of_result= 10)

    if idx.shape[0] == 1:
        return jsonify({
            'success': True,
            'result_baseline': res_base,
            'result_kmeans': res_kmeans
        })
    else:
        return jsonify({
            'success':False,
            'error': 'No Such File'
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=SERVER_PORT)

