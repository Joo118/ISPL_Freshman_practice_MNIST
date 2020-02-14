import tensorflow as tf
import numpy as np
import glob
import os
# use following commands when 'Segmentation fault' error occurs
# import matplotlib
# matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from PIL import Image
from random import shuffle



def _bytes_feature(value): # input : 'string/bytes'
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) # output : 'bytes_list (protocol message)'


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes(imagefile): # input type : 'string (path)'
    image = np.array(Image.open(imagefile)) # numpy array로 이미지 불러오기
    image_raw = image.tostring()
    return image_raw # output type : 'bytes(string)'

def make_example(img, lab): # input type : 'bytes', 'float'
    """ TODO: Return serialized Example from img, lab """
    feature = {'encoded' : _bytes_feature(img), # 'bytes' to 'bytes_list (protocol)'
               'label' : _float_feature(lab)} # float' to 'float_list (protocol)'
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString() # output type : 'bytes (string)'

def write_tfrecord(imagedir, datadir, val_datadir): # write시 shuffle을 해놓자
    """ TODO: write a tfrecord file containing img-lab pairs
            imagedir: directory of input images
            datadir: directory of output a tfrecord file (or multiple tfrecord files) """
    filenames = []
    for (path, dir, files) in os.walk(imagedir) :
        if files :
            for filename in files :
                filenames.append(os.path.join(path,filename))
    shuffle(filenames) # Shuffling a list whose elements are abs_paths of input images(.png)

    if val_datadir is None : # test data를 write할 때의 경우.
        print("Start converting test data...\n")
        writer_ts = tf.python_io.TFRecordWriter(datadir + '/test.tfrecord')
        for filename in filenames :
            image = _image_as_bytes(filename)
            label = float(filename.split(os.path.sep)[-2])
            example_ts = make_example(img=image, lab=label)
            writer_ts.write(example_ts)
        writer_ts.close()
        print("...Done...\n")

    else : # 나머지는 train data write할 때의 경우.
        print("Start converting training data...\n")
        tr = filenames[len(filenames)//4:]
        va = filenames[:len(filenames)//4]
        writer_tr = tf.python_io.TFRecordWriter(datadir + '/train.tfrecord')
        writer_va = tf.python_io.TFRecordWriter(val_datadir + '/val.tfrecord')
        for filename in tr :
            image = _image_as_bytes(filename)
            label = float(filename.split(os.path.sep)[-2])
            example_tr = make_example(img=image, lab=label)
            writer_tr.write(example_tr)
        for filename in va :
            image = _image_as_bytes(filename)
            label = float(filename.split(os.path.sep)[-2])
            example_va = make_example(img=image, lab=label)
            writer_va.write(example_va)
        writer_tr.close()
        writer_va.close()
        print("...Done...\n")

def read_tfrecord(folder, batch=100, epoch=1):
    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
        img: float, 0.0~1.0 normalized
        lab: dim 10 one-hot vectors
        folder: directory where tfrecord files are stored in
        epoch: maximum epochs to train, default: 1 """
    # filename queue
    filenames = glob.glob(folder + '/*.tfrecord')
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch) # shuffle=True 옵션은 filename들을 shuffle함...

    # read serialized examples
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    # parse examples into feaures, each
    key_to_feature = {'encoded' : tf.FixedLenFeature([], tf.string, default_value=''),
                      'label' : tf.FixedLenFeature([], tf.float32, default_value=0.)}
    features = tf.parse_single_example(serialized_example, features=key_to_feature)

    # decode data
    img = tf.decode_raw(features['encoded'], tf.uint8)
    img /= tf.math.reduce_max(img)      # normalize img (dtype:float)
    img = tf.reshape(img, [28,28,1])
    #dim10 one-hot vector
    lab = tf.cast(features['label'], tf.uint8)
    lab = tf.one_hot(indices=lab, depth=10)

    # mini-batch examples queue
    batch_size = batch
    min_after_dequee = 10000 # Buffer의 최소 크기 : 버퍼에서 무작위로 샘플을 뽑아 배치 만든다.

    img, lab = tf.train.shuffle_batch([img, lab], batch_size=batch_size,
                                      capacity=min_after_dequee+3*batch_size, num_threads=1,
                                      min_after_dequeue=min_after_dequee)
    return img, lab
