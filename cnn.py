# Copyright Jun Jo (mvpcamus). All Right Reserved.
# camus@kaist.ac.kr / camus7@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as imlib
import numpy as np
import tensorflow as tf

print('Tensorflow version ' + tf.__version__)

class Summary(object):
  @staticmethod
  def _summary(name, var):
    _mean = tf.reduce_mean(var)
    _variance = tf.reduce_mean(tf.square(var - _mean))
    tf.summary.scalar(name+'_mean', _mean)
    tf.summary.scalar(name+'_variance', _variance)
    tf.summary.histogram(name, var)


class ConvLayer(Summary):
  '''
  Construct a convolutional 2D layer and its summaries
  args:
    image  = input image array
    ch_in  = input image channel size (e.g. rgb = 3)
    ch_out = output channel size (number of kernels)
    size   = size of kernel (patch)
    stride = kernel (patch) stride
    activation = activation function (cf. 'bn': for batch normalization)
  '''
  def __init__(self, image, ch_in, ch_out, size, stride, activation='none'):
    self.img = image
    self.strd = stride
    self.act = activation.lower()
    _W_shape = [size, size, ch_in, ch_out]
    self.W = tf.Variable(tf.truncated_normal(_W_shape, stddev=0.1), trainable=True, name='W')
    self._summary('W', self.W)
    if self.act != 'bn':
      self.B = tf.Variable(tf.constant(0.1, tf.float32, [ch_out]), trainable=True, name='B')
      self._summary('B', self.B)

  def out(self):
    WX = tf.nn.conv2d(self.img, self.W, strides=[1, self.strd, self.strd, 1], padding='SAME')
    if self.act == 'relu':
      return tf.nn.relu(WX + self.B)
    elif self.act == 'bn':
      return WX
    elif self.act == 'none':
      return WX + self.B
    else:
      raise ValueError('ERROR: unsupported activation option')


class FCLayer(Summary):
  '''
  Construct a fully connected layer and its summaries
  args:
    input_ = input array
    n_in   = input size
    n_out  = output size
    activiation = activation function
  '''
  def __init__(self, input_, n_in, n_out, activation='none'):
    self.input_ = input_
    self.act = activation.lower()
    self.W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.1), trainable=True, name='W')
    self._summary('W', self.W)
    self.B = tf.Variable(tf.constant(0.0, tf.float32, [n_out]), trainable=True, name='B')
    self._summary('B', self.B)

  def out(self):
    if self.act == 'relu':
      return tf.nn.relu(tf.matmul(self.input_, self.W) + self.B)
    elif self.act == 'none':
      return tf.matmul(self.input_, self.W) + self.B
    else:
      raise ValueError('ERROR: unsupported activation option')


class BatchNorm(Summary):
  '''
  Construct a batch normalization for input array
  args
    input_ = input array tensor
    n_out  = output size
    train  = True: train phase, False: test phase
    activiation = activation function
  '''
  def __init__(self, input_, n_out, train, activation='none'):
    self.input_ = input_
    self.act = activation.lower()
    self.beta = tf.Variable(tf.constant(0.0, shape=[n_out]), trainable=True, name='beta')
    self._summary('beta', self.beta)
    self.gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), trainable=True, name='gamma')
    self._summary('gamma', self.gamma)
    batch_mean, batch_var = tf.nn.moments(input_, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    if train:
      with tf.control_dependencies([ema_apply_op]):
        self.mean, self.var = tf.identity(batch_mean), tf.identity(batch_var)
    else:
      self.mean, self.var = ema.average(batch_mean), ema.average(batch_var)

  def out(self):
    norm = tf.nn.batch_normalization(self.input_, self.mean, self.var, self.beta, self.gamma, 1e-3)
    if self.act == 'relu':
      return tf.nn.relu(norm)
    elif self.act == 'none':
      return norm
    else:
      raise ValueError('ERROR: unsupported activation option')


def model(X, Y_=None, p_keep=None):
  '''
  Define a DNN inference model
  args:
    X      = input image array
    Y_     = labels of input (solutions of Y)
    train  = True: train phase, False: test phase
    p_keep = keep probability of drop out, if NOT defined TEST phase model will run
  returns:
    Y      = predicted output array (e.g. [1, 0, 0, 0])
    cross_entropy
    accuracy
    incorrects = indices of incorrect inference
    f_maps = dictionary of convolution layer feature maps

  X     : [1000 x 1000 x 3] HWC image volume
  Conv1 : [100 x 100 x K1] output volume after [100 x 100 x 3] kernel with stride 10
  Conv2 : [50 x 50 x K2] output volume after [10 x 10 x K1] kernel with stride 2
  Conv3 : [25 x 25 x K3] output volume after [5 x 5 x K2] kernel with stride 2
  Conv4 : [25 x 25 x K4] output volume after [3 x 3 x K3] kernel with stride 1
  Full1 : [F1] output nodes from [25 * 25 * K4] input nodes
  Full2 : [F2] output nodes from [F1] input nodes
  Output: [4] ouput nodes from [F2] input nodes
  '''
  K1 = 10    # Conv1 layer feature map depth
  K2 = 20    # Conv2 layer feature map depth
  K3 = 40    # Conv3 layer feature map depth
  K4 = 20    # Conv4 layer feature map depth
  F1 = 500   # Full1 layer node size
  F2 = 50    # Full2 layer node size

  train_phase = False if p_keep is None else True

  with tf.variable_scope('Conv1'):
    y1 = ConvLayer(X, 3, K1, 100, 10, activation='BN').out()
    with tf.variable_scope('BN'):
      y1 = BatchNorm(y1, K1, train_phase, activation='ReLU').out()
  with tf.variable_scope('Conv2'):
    y2 = ConvLayer(y1, K1, K2, 10, 2, activation='ReLU').out()
  with tf.variable_scope('Conv3'):
    y3 = ConvLayer(y2, K2, K3, 5, 2, activation='ReLU').out()
  with tf.variable_scope('Conv4'):
    y4 = ConvLayer(y3, K3, K4, 3, 1, activation='ReLU').out()
  y4_rs = tf.reshape(y4, shape=[-1, 25*25*K4])

  with tf.variable_scope('Full1'):
    y5 = FCLayer(y4_rs, 25*25*K4, F1, activation='ReLU').out()
    if train_phase: y5 = tf.nn.dropout(y5, p_keep)
  with tf.variable_scope('Full2'):
    y6 = FCLayer(y5, F1, F2, activation='ReLU').out()
    if train_phase: y6 = tf.nn.dropout(y6, p_keep)
  with tf.variable_scope('Output'):
    Ylogits = FCLayer(y6, F2, 4).out()
  Y = tf.nn.softmax(Ylogits, name='Y')

  f_maps = {'Conv1':y1, 'Conv2':y2, 'Conv3':y3, 'Conv4':y4}

  if Y_ is not None:
    with tf.variable_scope('cross_entropy'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
      cross_entropy = tf.reduce_mean(cross_entropy)
    with tf.variable_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      incorrects = tf.squeeze(tf.where(tf.logical_not(correct_prediction)), [1])

    return Y, cross_entropy, accuracy, incorrects, f_maps
  else:
    return Y, f_maps

def gen_data(file_path, batch_size=1, one_hot=True, shuffle=True):
  '''
  Generate input data batches from png images
  args
    file_path  = input image(png) file path
    batch_size = batch size for each training step
    one_hot    = True: output labels (Y_) as one_hot arrays
    shuffle    = True: shuffle data queue sequence
  return
    data = {'X':[X1, ..., Xn], 'Y_':[Y_1, ..., Y_n], 'png':[png_path1, ..., png_pathn]}
  '''
  data = {}
  files = [os.path.join(file_path, s) for s in os.listdir(file_path)]
  pngs = []
  for f in files:
    if (not os.path.isdir(f)) and (os.path.splitext(f)[-1]=='.png'): pngs.append(f)
  queue = tf.train.string_input_producer(pngs, shuffle=shuffle)
  reader = tf.WholeFileReader()
  file_path, contents = reader.read(queue)
  img = tf.image.decode_png(contents, channels=3)
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    raw = [sess.run([file_path, img]) for _ in pngs]
    data['png'] = [f_i[0].decode() for f_i in raw]
    data['X'] = [f_i[1] for f_i in raw]
    data['Y_'] = [int(p.strip('.png').split('-')[-1]) for p in data['png']]
    if one_hot: data['Y_'] = sess.run(tf.one_hot(data['Y_'],4))
    coord.request_stop()
    coord.join(threads)
  return data


def do_train(MAX_STEP, BATCH_SIZE, INPUT_PATH, MODEL_PATH, LOG_DIR):
    startTime = time.time()
    # input generation
    with tf.Graph().as_default() as input_g:
      data = gen_data(INPUT_PATH, BATCH_SIZE)
      n_data = len(data['png'])
      print('[%6.2f] successfully generated train data: %d samples'%(time.time()-startTime, n_data))

    # training phase
    with tf.Graph().as_default() as train_g:
      # input X: 1000 x 1000 rgb color image
      X = tf.placeholder(tf.float32, [None, 1000, 1000, 3], name='X')
      # target values Y_: 0=normal, 1=stator fault, 2=rotor fault, 3=bearing fault
      Y_ = tf.placeholder(tf.float32, [None, 4], name='Y_')
      with tf.variable_scope('Config'):
        # dropout keep probability
        p_keep = tf.placeholder(tf.float32, name='p_keep')
        # learning rate
        global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(0.001, global_step, int(MAX_STEP/5), 0.5, staircase=True, name='lr')

      # load inference model
      Y, cross_entropy, accuracy, _, _ = model(X, Y_, p_keep)
      train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)

      with tf.variable_scope('Metrics'):
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('learning_rate', lr)

      # set tensorboard summary and saver
      merged = tf.summary.merge_all()
      saver = tf.train.Saver(max_to_keep=100)

      # training session
      print('----- training start -----')
      with tf.Session() as sess:
        sum_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        tf.global_variables_initializer().run()
        step = 1
        while step <= MAX_STEP:
          for batch in range(int(n_data/BATCH_SIZE)+1):
            s = batch*BATCH_SIZE
            e = (batch+1)*BATCH_SIZE if (batch+1)*BATCH_SIZE < n_data else n_data
            if e <= s: break
            _, summary, acc, ent = sess.run([train_op, merged, accuracy, cross_entropy],
                                            {X:data['X'][s:e], Y_:data['Y_'][s:e], p_keep:0.75})
            sum_writer.add_summary(summary, step)
            print('[%6.2f] step:%3d, size:%3d, lr:%f, accuracy:%f, cross entropy:%f'
                  %(time.time()-startTime, step, e-s, lr.eval(), acc, ent))
            if (MAX_STEP-step)<10 or step%100==0:
              saver.save(sess, MODEL_PATH, global_step=step)
            step += 1
            if step > MAX_STEP: break
      print('-----  training end  -----')


def do_test(BATCH_SIZE, INPUT_PATH, MODEL_PATH, LOG_DIR, F_MAP=None):
  startTime = time.time()
  # input generation
  with tf.Graph().as_default() as input_g:
    data = gen_data(INPUT_PATH, shuffle=False)
    n_data = len(data['png'])
    print('[%6.2f] successfully generated test data: %d samples'%(time.time()-startTime, n_data))

  # test phase
  with tf.Graph().as_default() as test_g:
    # input X: 1000 x 1000 rgb color image
    X = tf.placeholder(tf.float32, [None, 1000, 1000, 3], name='X')
    # target values Y_: 0=normal, 1=stator fault, 2=rotor fault, 3=bearing fault
    Y_ = tf.placeholder(tf.float32, [None, 4], name='Y_')

    # load inference model
    Y, cross_entropy, accuracy, incorrects, f_maps = model(X, Y_)

    with tf.variable_scope('Metrics'):
      tf.summary.scalar('accuracy', accuracy)
      tf.summary.scalar('cross_entropy', cross_entropy)

    # set tensorboard summary and saver
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    # test session
    print('----- test start -----')
    with tf.Session() as sess:
      sum_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
      tf.global_variables_initializer().run()
      saver.restore(sess, MODEL_PATH)
      avg_accuracy = 0
      for step in range(int(n_data/BATCH_SIZE)+1):
        s = step*BATCH_SIZE
        e = (step+1)*BATCH_SIZE if (step+1)*BATCH_SIZE < n_data else n_data
        if e <= s: break
        summary, acc, ent, incor, y_, y = sess.run([merged, accuracy, cross_entropy, incorrects, Y_, Y],
                                                    {X:data['X'][s:e], Y_:data['Y_'][s:e]})
        sum_writer.add_summary(summary, step+1)
        avg_accuracy += acc * (e-s)
        print('[%6.2f] step:%d, size:%d, accuracy:%f, cross entropy:%f'
                %(time.time()-startTime, step+1, e-s, acc, ent))
        if len(incor) > 0: print('   incorrects list:')
        for i in incor:
          print('   [%3d] Answer:Infer = %d:%d  at %s'
                  %(s+i, tf.argmax(y_[i],0).eval(),tf.argmax(y[i],0).eval(),data['png'][s+i]))
      print('-----  test end  -----')
      print('[%6.2f] total average accuracy: %f'%(time.time()-startTime, avg_accuracy/n_data))

      # feature map extraction for the first input data
      if F_MAP is not None:
        feature_maps = sess.run(f_maps, {X:[data['X'][0]], Y_:[data['Y_'][0]]})
        layers = sorted(list(feature_maps.keys()))
        for l in layers:
          n_kernel = len(feature_maps[l][0][0][0])
          if F_MAP.lower() == 'show': plt.figure().suptitle(l)
          for k in range(n_kernel):
            img = feature_maps[l][0][:,:,k]
            max_ = img.max()
            img = np.uint8(img / max_ * 255) if max_!=0 else np.uint8(img)
            if F_MAP.lower() == 'save':
              imlib.imsave('%s%s-%02d.png'%(LOG_DIR,l,k+1), img, cmap='gist_gray')
            else:
              plt.subplot(5, n_kernel/5, k+1)
              plt.imshow(img, cmap='gist_gray')
        if F_MAP.lower() == 'show': plt.show()


def do_infer(INPUT_FILE, MODEL_PATH, FMAP_DIR):
  # input generation
  with tf.Graph().as_default() as input_g:
    data = {}
    file_path, contents = tf.WholeFileReader().read(tf.train.string_input_producer([INPUT_FILE]))
    img = tf.image.decode_png(contents, channels=3)
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      data['png'], data['X'] = sess.run([file_path, img])
      coord.request_stop()
      coord.join(threads)

  # inference phase
  with tf.Graph().as_default() as infer_g:
    # input X: 1000 x 1000 rgb color image
    X = tf.placeholder(tf.float32, [None, 1000, 1000, 3], name='X')
    # load inference model
    Y, f_maps = model(X)
    # set tensorboard and saver
    saver = tf.train.Saver()

    # inference session
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      saver.restore(sess, MODEL_PATH)
      y, feature_maps = sess.run([Y, f_maps], {X:[data['X']]})
      index = tf.argmax(y[0], 0).eval()
      layers = sorted(list(feature_maps.keys()))
      for l in layers:
        n_kernel = len(feature_maps[l][0][0][0])
        for k in range(n_kernel):
          img = feature_maps[l][0][:,:,k]
          max_ = img.max()
          img = np.uint8(img / max_ * 255) if max_!=0 else np.uint8(img)
          imlib.imsave('%s%s-%02d.png'%(FMAP_DIR,l,k+1), img, cmap='gist_gray')

  return index, y[0]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-train', const=True, action='store_const',
                        help='run in train mode instead of test mode')
  parser.add_argument('-maxstep', type=int, default=1000,
                        help='step limit of train phase / meaningless in test mode .... [ default: %(default)s ]')
  parser.add_argument('-bsize', type=int, default=100,
                        help='batch size for each step ................................ [ default: %(default)s ]')
  parser.add_argument('-input', type=str, default='./input_data/',
                        help='directory path of input image files ..................... [ default: %(default)s ]')
  parser.add_argument('-model', type=str, default='./tmp/model/cnn.ckpt',
                        help='file path for the trained model ......................... [ default: %(default)s ]')
  parser.add_argument('-log', type=str, default='./tmp/log/',
                        help='directory path for Tensorboard logs ..................... [ default: %(default)s ]')
  parser.add_argument('-fmap', type=str, default=None,
                        help='show or save feature maps of the first input data ....... [ \'show\' or \'save\' ]')
  FLAGS = parser.parse_args()

  MAX_STEP = FLAGS.maxstep
  BATCH_SIZE = FLAGS.bsize
  INPUT_PATH = FLAGS.input
  MODEL_PATH = FLAGS.model
  LOG_DIR = FLAGS.log
  F_MAP = FLAGS.fmap

  if FLAGS.train:
    do_train(MAX_STEP, BATCH_SIZE, INPUT_PATH, MODEL_PATH, LOG_DIR)

  else:
    do_test(BATCH_SIZE, INPUT_PATH, MODEL_PATH, LOG_DIR, F_MAP)
