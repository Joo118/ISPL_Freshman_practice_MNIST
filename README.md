# ISPL_Freshman_practice_MNIST
 
1. How to Write TFRecord.

python main.py --process=write --imagedir=./mnist/train --test_imagedir=./mnist/test --datadir=./mnist/tfrecord/train_tfrecord --val_datadir=./mnist/tfrecord/val_tfrecord --test_datadir=./mnist/tfrecord/test_tfrecord


2. How to Train.

python main.py --process=train --datadir=./mnist/tfrecord/train_tfrecord --val_datadir=./mnist/tfrecord/val_tfrecord --epoch=3 --lr=1e-3 --ckptdir=./ckpt --batch=100 --restore=False


3. How to Test.

python main.py --process=test --test_datadir=./mnist/tfrecord/test_tfrecord --ckptdir=./ckpt --batch=100


