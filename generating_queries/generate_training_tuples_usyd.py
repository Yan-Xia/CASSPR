# Modified PointNetVLAD code: https://github.com/mikacuy/pointnetvlad
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "/usr/wiss/xiya/storage/user/xia/data/USyd/"

runs_folder = "weeks/"
pointcloud_fols = "/pointclouds_with_locations_5m/"

folders = []
filenames = []
training_weeks = [1, 2, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25,
                  26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 46, 45, 47, 49, 52]

folders = [f'output_week{week}' for week in training_weeks]
filenames = [f'pointcloud_locations_5m_week_{str(week).zfill(2)}.csv' for week in training_weeks]
print("Number of runs: " + str(len(folders)))
print(folders)
print(filenames)

#####For training and test data split#####
x_width = 100
y_width = 100
buffer = 10
# points in easting, northing (x, y) format
p1_bl_corner = [332_530, -3_750_950]
p2_bl_corner = [332_250, -3_751_240]
p3_bl_corner = [332_630, -3_751_450]
p4_bl_corner = [332_555, -3_751_125]
p = [p1_bl_corner, p2_bl_corner, p3_bl_corner, p4_bl_corner]


# modified, since regions are defined by bottom left corner + width and buffer is added
def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    in_buffer_set = False

    for point in points:
        # points in easting, northing (x, y) format
        if (point[0] - buffer) < easting < (point[0] + x_width + buffer) and (point[1] - buffer) < northing < (point[1] + y_width + buffer):
            # in buffer range - test or reject:
            if (point[0]) < easting < (point[0] + x_width) and (point[1]) < northing < (point[1] + y_width):
                # in test range
                in_test_set = True
            else:
                in_buffer_set = True
            break
    return in_test_set, in_buffer_set


##########################################


def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['northing', 'easting']])
    # CURRENT DISTANCES: POS<10, NEG>25
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=10)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=25)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query": query,
                      "positives": positives,
                      "negatives": negatives}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


####Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

for folder, filename in zip(folders, filenames):
    print(os.path.join(base_path, runs_folder, folder, filename))
    df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
    df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    for index, row in df_locations.iterrows():
        in_test, in_buffer = check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)
        if in_test and not in_buffer:
            df_test = df_test._append(row, ignore_index=True)
        elif not in_buffer:
            df_train = df_train._append(row, ignore_index=True)

print("Number of train clouds: " + str(len(df_train['file'])))
print("Number of test clouds: " + str(len(df_test['file'])))

construct_query_dict(df_train, "usyd_training_queries.pickle")
construct_query_dict(df_test, "usyd_test_queries.pickle")
