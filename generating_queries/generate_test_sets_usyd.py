# Modified PointNetVLAD code: https://github.com/mikacuy/pointnetvlad
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
import argparse

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

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filenames, p, output_name):
    database_trees = []
    test_trees = []
    for folder, filename in zip(folders, filenames):
        print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        # df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        # df_locations=df_locations.rename(columns={'timestamp':'file'})
        for index, row in df_locations.iterrows():
            in_test, in_buffer = check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)
            if in_test:
                df_test = df_test._append(row, ignore_index=True)
            df_database = df_database._append(row, ignore_index=True)

        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []
    for folder, filename in zip(folders, filenames):
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(
            str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            in_test, in_buffer = check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)
            if in_test:
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting']}
        database_sets.append(database)
        test_sets.append(test)

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if (i == j):
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                # DISTANCES: CORRECT<10
                index = tree.query_radius(coor, r=10)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, output_name + '_evaluation_database.pickle')
    output_to_file(test_sets, output_name + '_evaluation_query.pickle')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    args = parser.parse_args()

    base_path = args.data_dir

    ###Building database and query files for evaluation

    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # base_path = "/usr/wiss/xiya/storage/user/xia/data/USyd/"

    # For USyd
    runs_folder = "weeks/"
    pointcloud_fols = "/pointclouds_with_locations_5m/"

    folders = []
    filenames = []
    validation_weeks = [1, 2, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25,
                        26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 46, 45, 47, 49, 52]

    folders = [f'output_week{week}' for week in validation_weeks]
    filenames = [f'pointcloud_locations_5m_week_{str(week).zfill(2)}.csv' for week in validation_weeks]
    print(folders)
    print(filenames)
    construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filenames, p, "usyd")
