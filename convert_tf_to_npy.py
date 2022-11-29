import os
import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2 

from tqdm import tqdm 

import os
import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2 

from tqdm import tqdm 

train_feature_description = {
  "BLUE": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "GREEN": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "RED": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "NIR": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "SWIR1": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "SWIR2": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "TEMP1": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "V001": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "V000": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
  "LATNUM": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "LONGNUM": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "V445": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "V006": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
  "V007": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
}

def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, train_feature_description)
 
def convert_filename_to_dhsid(file):
  code = file[0:6]
  num = file.split("_")[-1].split(".")[0]
  num_just = num.rjust(8, "0")
  return code + num_just

def check_valid_file(file, dhsid_dict):
  return convert_filename_to_dhsid(file) in dhsid_dict

def convert_tfrec_to_npy(in_files_paths, out_folder, label_dict):
    train_image_dataset = tf.data.TFRecordDataset(in_files_paths)
    train_image_dataset = train_image_dataset.map(_parse_image_function)
    print("Loaded images.")

    red = []
    green = []
    blue = []
    for image_features in tqdm(train_image_dataset):
        red.append(image_features['RED'].numpy())
        green.append(image_features['GREEN'].numpy())
        blue.append(image_features['BLUE'].numpy())

    print("\nExtracted colors.")

    def transform_arr(mat, mask = None):
        if mask: mat = np.array([mat[i] for i in range(len(mat)) if mask[i]])
        else: mat = np.array(mat)
        return np.reshape(mat, (mat.shape[0], 511, 511))

    def check_arr_shape(mat):
        mat = np.array(mat)
        if mat.shape == (511**2, ): return True
        return False

    arr_shape_ok = [check_arr_shape(i) for i in red]
    in_files_paths_ok = np.array(in_files_paths)[arr_shape_ok]

    red = transform_arr(red, arr_shape_ok)
    green = transform_arr(green, arr_shape_ok)
    blue = transform_arr(blue, arr_shape_ok)

    for i in tqdm(range(red.shape[0])):

        rgb = np.dstack((red[i], green[i], blue[i]))
        
        # nnormalize values
        norm_image = cv2.normalize(rgb, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        norm_image = norm_image.astype("uint8")

        # get dhsid
        dhsid = convert_filename_to_dhsid(in_files_paths_ok[i].split("/")[-1])

        # class label 
        label = label_dict[dhsid]

        np.save(out_folder + "class_" + str(label) + "/" + dhsid + ".npy", norm_image)


if __name__ == "__main__":

    ## NOTE: FILL THESE IN: 
    country_codes = []
    # main project directory 
    PROJECT_FOLDER = "FILL THIS IN"
    # where data folders are stored (probably full_dataset_tfrecord)
    in_folder = PROJECT_FOLDER + "/" + "IN_FOLDER"
    # where dat folders should be  written 
    out_folder = PROJECT_FOLDER + "/" + "OUT_FOLDER/" 

    # make sure path is correct 
    df = pd.read_csv("hi.csv")

    # filter to countries of interest
    df = df.loc[df["DHSID"].str[0:2].isin(country_codes)]

    # pick the year with most images for each country
    df_cntry_yr = df.groupby(["country", "year"])["DHSID"].count() \
    .reset_index().sort_values("DHSID", ascending = False) \
    .groupby("country").first().reset_index()
    df = df.merge(df_cntry_yr[["country", "year"]], on = ["country", "year"], how = "inner")

    # class encoding 
    # NOTE: change to use different discretization 
    df["Mean_BMI_bin"] = np.select(
        [df["Mean_BMI"] <= 19.9, df["Mean_BMI"] <= 24.9, True], 
        [0, 1, 2]
    )
    N_CLASSES = df["Mean_BMI_bin"].drop_duplicates().shape[0]
    dhsid_label_dict = dict(zip(df["DHSID"], df["Mean_BMI_bin"]))

    # get country years 
    country_years = df[["country", "year"]].drop_duplicates().reset_index(drop=True).values
    countries = country_years[:, 0]
    years = country_years[:, 1]
    country_year_pairs = [(countries[i], years[i]) for i in range(countries.shape[0])]

    # make folders (TODO: check if they exist)
    os.mkdir(out_folder)
    for i in range(N_CLASSES):
      os.mkdir(out_folder + "class_" + str(i))

    # loop over each pair
    for c, y in country_year_pairs: 
        print(f"Country {c} in year {y}")
        in_folder = in_folder + c + str(y) + "_" + str(y) + "/"
        in_files = os.listdir(in_folder)
        in_files = [f for f in in_files if check_valid_file(f, dhsid_label_dict)]
        in_files_paths = [in_folder + f for f in in_files]
        
        # convert 
        convert_tfrec_to_npy(in_files_paths, out_folder, dhsid_label_dict)
