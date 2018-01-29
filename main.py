# Copyright Jun Jo (mvpcamus). All Right Reserved.
# camus@kaist.ac.kr / camus7@gmail.com

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn import ConvLayer
from cnn import BatchNorm
from cnn import FCLayer

def get_image(data, index):
  org1 = np.array(data.band_1[index])
  org2 = np.array(data.band_2[index])
  org1.shape = (75,75)
  org2.shape = (75,75)
  max_all = max(np.max(org1), np.max(org2))
  min_all = min(np.min(org1), np.min(org2))

  img1 = org1 - min_all
  img1 /= (max_all - min_all)
  img1 -= np.min(img1)

  img2 = org2 - min_all
  img2 /= (max_all - min_all)
  img2 -= np.min(img2)

  blue = np.abs(img1*img2)

  condition = (org1 > np.mean(org1)+2*np.sqrt(np.var(org1))).astype(int)
  red = img1 * condition

  condition = (org2 > np.mean(org2)+2*np.sqrt(np.var(org2))).astype(int)
  green = img2 * condition

  picture = np.zeros([75, 75, 3])
  picture[:,:,0] = red
  picture[:,:,1] = green
  picture[:,:,2] = blue

  return picture

def model(X, Y_=None, p_keep=None):
  K1 = 50
  K2 = 100
  F1 = 500
  F2 = 250
  with tf.variable_scope('Conv1'):
    y1 = ConvLayer(X, 3, K1, 3, 1, activation='BN').out()
    with tf.variable_scope('BN'):
      y1 = BatchNorm(y1, K1, True, activation='ReLU').out()
  with tf.variable_scope('Conv2'):
    y2 = ConvLayer(y1, K1, K2, 5, 3, activation='ReLU').out()
  y2_rs = tf.reshape(y2, shape=[-1, 25*25*K2])
  with tf.variable_scope('Full1'):
    y3 = FCLayer(y2_rs, 25*25*K2, F1, activation='ReLU').out()
  with tf.variable_scope('Full2'):
    y4 = FCLayer(y3, F1, F2, activation='ReLU').out()
  with tf.variable_scope('Output'):
    Ylogits = FCLayer(y4, F2, 2).out()
  Y = tf.nn.softmax(Ylogits, name='Y')

  if Y_ is not None:
    with tf.variable_scope('cross_entropy'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
      cross_entropy = tf.reduce_mean(cross_entropy)
    with tf.variable_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      incorrects = tf.squeeze(tf.where(tf.logical_not(correct_prediction)), [1])
    return Y, cross_entropy, accuracy, incorrects
  else:
    return Y

def do_train():
  data = pd.read_json('./train.json')
  N_DATA = len(data.id)
  MAX_STEP = 1000
  BATCH_SIZE = 100
  L_RATE = 0.001

  X = tf.placeholder(tf.float32, [None, 75, 75, 3], name='X')
  Y_ = tf.placeholder(tf.float32, [None, 2], name='Y_')
  global_step = tf.Variable(0, name='global_step', trainable=False)

  Y, cross_entropy, accuracy, _ = model(X, Y_)
  train_op = tf.train.AdamOptimizer(L_RATE).minimize(cross_entropy, global_step=global_step)

  with tf.variable_scope('Metrics'):
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cross_entropy', cross_entropy)
  
  merged = tf.summary.merge_all()
  saver = tf.train.Saver(max_to_keep=100)

  # training session
  print('----- training start -----')
  with tf.Session() as sess:
    sum_writer = tf.summary.FileWriter('./log', sess.graph)
    tf.global_variables_initializer().run()
    step = 1
    while step <= MAX_STEP:
      for batch in range(int(N_DATA/BATCH_SIZE)+1):
        s = batch*BATCH_SIZE
        e = (batch+1)*BATCH_SIZE if (batch+1)*BATCH_SIZE < N_DATA else N_DATA
        if e <= s: break
        data_X = []
        data_Y_ = []
        for i in range(s,e):
          image = get_image(data, i)
          for j in range(4):
            data_X.append(np.rot90(image,j))
            data_Y_.append([data.is_iceberg[i], 1-data.is_iceberg[i]])
        _, summary, acc, ent = sess.run([train_op, merged, accuracy, cross_entropy],
                          {X:data_X, Y_:data_Y_})
        sum_writer.add_summary(summary, step)
        print('step:%3d, size:%3d, accuracy:%f, cross entropy:%f'%(step, e-s, acc, ent))
        if (MAX_STEP-step)<5 or step%100==0:
          saver.save(sess, './model/ckpt', global_step=step)
        step += 1
        if step > MAX_STEP: break
  print('-----  training end  -----')

def do_test():
  data = pd.read_json('./test.json')
  N_DATA = len(data.id)
  BATCH_SIZE = 100

  data_X = []
  for index in range(N_DATA):
    data_X.append(get_image(data,index))

  X = tf.placeholder(tf.float32, [None, 75, 75, 3], name='X')
  Y = model(X)
  saver = tf.train.Saver()

  csv = open('./output.csv', 'w')
  csv.write('id,is_iceberg\n')

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, './model/ckpt-1000')
    for batch in range(int(N_DATA/BATCH_SIZE)+1):
      s = batch*BATCH_SIZE
      e = (batch+1)*BATCH_SIZE if (batch+1)*BATCH_SIZE < N_DATA else N_DATA
      if e <= s: break
      y = sess.run([Y], {X:data_X[s:e]})
      for index in range(len(y[0])):
        text = '%s,%f'%(data.id[s+index], y[0][index][0])
        print(text)
        csv.write(text+'\n')
  csv.close() 

if __name__ == '__main__':
  if len(sys.argv) > 1:
    opt = sys.argv[1]
    if opt == 'train':
      do_train()
    elif opt == 'test':
      do_test()
    else:
      print('Invalid option: ',sys.argv[1])
  else:
    print('Help: main.py [train / test]')
