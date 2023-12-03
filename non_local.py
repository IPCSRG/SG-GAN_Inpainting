# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import ops
from inpaint_ops import gen_conv

def conv1x1(input_, output_dim,
            init=tf.contrib.layers.xavier_initializer(), name='conv1x1'):
  k_h = 1
  k_w = 1
  d_h = 1
  d_w = 1
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_conv1x1(input_, output_dim, update_collection,
              init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1'):
  with tf.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    w_bar = ops.spectral_normed_weight(w, num_iters=1, update_collection=update_collection)

    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_non_local_block_sim(x, update_collection, name, init=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope(name):
    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h * w
    downsampled_num = location_num // 4

    # theta path
    theta = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_theta')
    theta = tf.reshape(
        theta, [batch_size, location_num, num_channels // 8])

    # phi path
    phi = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_phi')
    phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    phi = tf.reshape(
        phi, [batch_size, downsampled_num, num_channels // 8])

    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn)
    print(tf.reduce_sum(attn, axis=-1))

    # g path
    g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g')
    g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
    g = tf.reshape(
      g, [batch_size, downsampled_num, num_channels // 2])

    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn')
    return x + sigma * attn_g



def sn_non_local_block_sim32(x0, x, update_collection, name, init=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        batch_size, h, w, num_channels = x.get_shape().as_list()
        location_num = h * w
        downsampled_num = location_num // 4

        # theta path
        theta = sn_conv1x1(x0, num_channels // 48, update_collection, init, 'sn_conv_theta32')
        theta = gen_conv(theta, num_channels // 48, 32, 1, name='conv32_theta')
        theta = tf.reshape(
            theta, [batch_size, location_num, num_channels // 48])

        # phi path
        phi = sn_conv1x1(x, num_channels // 48, update_collection, init, 'sn_conv_phi32')
        phi = gen_conv(phi, num_channels // 48, 32, 1, name='conv32_phi')
        phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
        phi = tf.reshape(
            phi, [batch_size, downsampled_num, num_channels // 48])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        print(tf.reduce_sum(attn, axis=-1))

        # g path
        g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g32')
        g = gen_conv(g, num_channels // 2, 32, 1, name='conv32_g')
        g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
        g = tf.reshape(
            g, [batch_size, downsampled_num, num_channels // 2])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
        sigma = tf.get_variable(
            'sigma_ratio32', [], initializer=tf.constant_initializer(0.0))
        attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn32')
        return x + sigma * attn_g


def sn_non_local_block_sim64(x0, x, update_collection, name, init=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        batch_size, h, w, num_channels = x.get_shape().as_list()
        location_num = h * w
        downsampled_num = location_num // 4

        # theta path
        theta = sn_conv1x1(x0, num_channels // 48, update_collection, init, 'sn_conv_theta64')
        theta = gen_conv(theta, num_channels // 48, 64, 1, name='conv64_theta')
        theta = tf.reshape(
            theta, [batch_size, location_num, num_channels // 48])

        # phi path
        phi = sn_conv1x1(x, num_channels // 48, update_collection, init, 'sn_conv_phi64')
        phi = gen_conv(phi, num_channels // 48, 64, 1, name='conv64_phi')
        phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
        phi = tf.reshape(
            phi, [batch_size, downsampled_num, num_channels // 48])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        print(tf.reduce_sum(attn, axis=-1))

        # g path
        g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g64')
        g = gen_conv(g, num_channels // 2, 64, 1, name='conv64_g')
        g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
        g = tf.reshape(
            g, [batch_size, downsampled_num, num_channels // 2])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
        sigma = tf.get_variable(
            'sigma_ratio64', [], initializer=tf.constant_initializer(0.0))
        attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn64')

        # return x + Z1

        return x + sigma * attn_g


def sn_non_local_block_sim128(x0, x, update_collection, name, init=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        batch_size, h, w, num_channels = x.get_shape().as_list()
        location_num = h * w
        downsampled_num = location_num // 4

        # theta path
        theta = sn_conv1x1(x0, num_channels // 48, update_collection, init, 'sn_conv_theta128')
        theta = gen_conv(theta, num_channels // 48, 128, 1, name='conv128_theta')
        theta = tf.reshape(
            theta, [batch_size, location_num, num_channels // 48])

        # phi path
        phi = sn_conv1x1(x, num_channels // 48, update_collection, init, 'sn_conv_phi128')
        phi = gen_conv(phi, num_channels // 48, 128, 1, name='conv128_phi')
        phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
        phi = tf.reshape(
            phi, [batch_size, downsampled_num, num_channels // 48])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        print(tf.reduce_sum(attn, axis=-1))

        # g path
        g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g128')
        g = gen_conv(g, num_channels // 2, 128, 1, name='conv128_g')
        g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
        g = tf.reshape(
            g, [batch_size, downsampled_num, num_channels // 2])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
        sigma = tf.get_variable(
            'sigma_ratio128', [], initializer=tf.constant_initializer(0.0))
        attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn128')


        return x + sigma * attn_g