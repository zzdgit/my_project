from random import shuffle
import glob
import cv2
import tensorflow as tf
import numpy as np
import sys
import os
import csv

celeba_training_data = '/nfsdata/share-read-only/datasets/celeba_face/training/*.jpg'

addrs_train = glob.glob(celeba_training_data)

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_lable():
    label_list = []
    label_file_path = '/nfsdata/share-read-only/datasets/celeba_face/celeba_attr.csv'
    file_path = '/nfsdata/share-read-only/datasets/celeba_face/training/'
    i_start = 1
    i_end = 180001
    index = 32
    with open(label_file_path) as f:
        reader = csv.reader(f, delimiter=' ')
        i = 0
        for row in reader:
            if i > i_start and i <= i_end :
                # generate gender labels and file path
                filename = file_path + row[0]
                if os.path.exists(filename):
                    if "1" == row[index]:
                        label = 0
                    elif "-1" == row[index]:
                        label = 1
                    label_list.append(label)
            i = i + 1
    return label_list


def generate_train_data(label_list):
    for i in range(len(addrs_train)/10000):
        train_filename = '/home/zhouzd2/celeba_face/celeba_train_{}.tfrecords'.format(i)  # address to save the TFRecords file

        # open the TFRecords file
        # options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        # writer = tf.python_io.TFRecordWriter(train_filename, options=options)
        writer = tf.python_io.TFRecordWriter(train_filename)

        for j in range(10000):
            # print how many images are saved every 1000 images
            if not j % 1000:
                print 'Train data: {}/{}'.format(j*(i+1), len(addrs_train))
                sys.stdout.flush()

            # Load the image
            img = load_image(addrs_train[i])

            label = label_list[i]

            # Create a feature
            feature = {'train/label': _int64_feature(label),
                       'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

def read_celeba_data():
    data_path = '/home/zhouzd2/celeba_face'
    file_name_list = os.listdir(data_path)
    data_list = [data_path + '/' + d for d in file_name_list if 'celeba_train' in d]

    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(data_list, num_epochs=1)
    # Define a reader and read the next record
    #options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    #options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    #reader = tf.TFRecordReader(options=options)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [224, 224, 3])
    print ('11111111111111111111111111111')
    print label
    print image
    return image, label


if __name__ == '__main__':
    #read_celeba_data()
    label_list = read_lable()
    print ("##############", len(label_list))
    generate_train_data(label_list)