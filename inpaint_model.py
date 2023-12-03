import logging

import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty
from neuralgym.ops.gan_ops import random_interpolates

from inpaint_ops import gen_conv, gen_snconv, gen_deconv
from inpaint_ops import resize_mask_like, contextual_attention, vgg_loss, style_loss, free_form_mask_tf
import non_local

logger = logging.getLogger()

class InpaintGCModel(Model):
    def __init__(self):
        super().__init__('InpaintGCModel')
#   生成器
    def build_inpaint_net(self, x0, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        """
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)

        cnum = 48
        with tf.variable_scope(name, reuse=reuse), \
             arg_scope([gen_conv, gen_deconv],
                       training=training, padding=padding):
            # 256
            x = gen_conv(x, cnum, 5, 1, name='conv1')
            # 128
            x = gen_conv(x, 2 * cnum, 3, 2, name='conv1_downsample')
            x = gen_conv(x, 2 * cnum, 3, 1, name='conv2')
            # 64
            x = gen_conv(x, 4 * cnum, 3, 2, name='conv2_downsample')
            x = gen_conv(x, 4 * cnum, 3, 1, name='conv3')
            x64_e = x
            # 32
            x = gen_conv(x, 4 * cnum, 3, 2, name='conv3_downsample')
            x = gen_conv(x, 4 * cnum, 3, 1, name='conv4')
            x32_e = x
            # 16
            x = gen_conv(x, 4 * cnum, 3, 2, name='conv4_downsample')
            x = gen_conv(x, 4 * cnum, 3, 1, name='conv5')
            x16_e = x
            # 8
            x = gen_conv(x, 4 * cnum, 3, 2, name='conv5_downsample')
            x = gen_conv(x, 4 * cnum, 3, 1, name='conv6')
            x8_e = x

            x_8_8 = gen_conv(x8_e, cnum, 3, 1, name='conv7')  # 8

            # for training
            x0 = gen_conv(x0, 2 * cnum, 3, 2, name='conv000_downsample')
            x128 = gen_conv(x0, 2 * cnum, 3, 1, name='conv001')
            x64 = gen_conv(x128, 2 * cnum, 3, 2, name='conv001_downsample')
            x64 = gen_conv(x64, 2 * cnum, 3, 1, name='conv002')
            x32 = gen_conv(x64, 2 * cnum, 3, 2, name='conv002_downsample')
            x32 = gen_conv(x32, 2 * cnum, 3, 1, name='conv003')

            x8_d = tf.concat([x8_e, x_8_8], axis=3)  # concat8
            x8_up_16 = gen_deconv(x8_d, cnum, name='conv7_upsample')
            x16_d = tf.concat([x16_e, x8_up_16], axis=3)  # concat32
            x16_up_32 = gen_deconv(x16_d, cnum, name='conv8_upsample')
            x16_32 = gen_deconv(x8_up_16, cnum, name='conv81_upsample')  # 48  20230617schadd
            x32_d = tf.concat([x32_e, x16_up_32, x16_32], axis=3)  # concat32

            # # att32
            x32_d = gen_conv(x32_d, 2 * cnum, 3, 1, name='conv_self32_1')
            mask_s = resize_mask_like(mask, x32_d)
            x32att = gen_conv(x32_d, 2 * cnum, 3, 1, name='pmconv32', activation=tf.nn.relu)
            att32, offset_flow_32 = contextual_attention(x32att, x32att, mask_s, 3, 1, rate=2)
            att32 = gen_conv(att32, 2 * cnum, 3, 1, name='conv9')
            x32_d_64 = gen_deconv(att32, 2 * cnum, name='conv9_upsample')
            x32_d_64 = gen_conv(x32_d_64, 2 * cnum, 3, 1, name='conv10')
            x32_up_64 = gen_deconv(x32_d, cnum, name='conv10_upsample')
            x32_64 = gen_deconv(x16_up_32, cnum, name='conv101_upsample')
            x64_d = tf.concat([x64_e, x32_d_64, x32_up_64, x32_64], axis=3)

            # att64
            x64_d = gen_conv(x64_d, 2 * cnum, 3, 1, name='conv10_1')
            x0_64 = x64
            att64 = non_local.sn_non_local_block_sim64(x0_64, x64_d, None, name='g_non_local64')

            x64_att = tf.concat([x64_d, att64], axis=3)
            x64_att = gen_conv(x64_att, 2 * cnum, 3, 1, name='conv10_2')


            mask_s = resize_mask_like(mask, x64_d)
            x64att_d = gen_conv(x64_d, 2 * cnum, 3, 1, name='pmconv64_d', activation=tf.nn.relu)
            att64_d, offset_flow_64_d = contextual_attention(x64att_d, x64att_d, mask_s, 3, 1, rate=2)

            mask_s = resize_mask_like(mask, x64_att)
            x64att = gen_conv(x64_att, 2 * cnum, 3, 1, name='pmconv64', activation=tf.nn.relu)
            att64, offset_flow_64 = contextual_attention(x64att, x64att, mask_s, 3, 1, rate=2)

            att64_d = gen_conv(att64_d, 2 * cnum, 3, 1, name='conv11_d')
            att64 = gen_conv(att64, 2 * cnum, 3, 1, name='conv11')

            x64_d_128_d = gen_deconv(att64_d, 2 * cnum, name='conv11_upsample_d')
            x64_d_128 = gen_deconv(att64, 2 * cnum, name='conv11_upsample')
            x64_d_128 = gen_conv(x64_d_128, 2 * cnum, 3, 1, name='conv12')
            x64_up_128 = gen_deconv(x64_d, cnum, name='conv12_upsample')
            x64_128 = gen_deconv(x32_up_64, cnum, name='conv121_upsample')
            x128_d = tf.concat([x64_d_128, x64_up_128, x64_128,x64_d_128_d], axis=3)  # concat128

            # att128
            x128_d = gen_conv(x128_d, 2 * cnum, 3, 1, name='conv12_1')
            x0_128 = x128
            att128 = non_local.sn_non_local_block_sim128(x0_128, x128_d, None, name='g_non_local128')

            x128_att = tf.concat([x128_d, att128], axis=3)  # concat128
            x128_att = gen_conv(x128_att, 2 * cnum, 3, 1, name='conv12_2')

            mask_s = resize_mask_like(mask, x128_d)
            x128att_d = gen_conv(x128_d, 2 * cnum, 3, 1, name='pmconv128', activation=tf.nn.relu)
            att128_d, offset_flow_d = contextual_attention(x128att_d, x128att_d, mask_s, 3, 1, rate=2)

            mask_s = resize_mask_like(mask, x128_att)
            x128att = gen_conv(x128_att, 2 * cnum, 3, 1, name='pmconv128_d', activation=tf.nn.relu)
            att128, offset_flow = contextual_attention(x128att, x128att, mask_s, 3, 1, rate=2)

            att128_d = gen_conv(att128_d, 2 * cnum, 3, 1, name='conv13_d')
            att128 = gen_conv(att128, 2 * cnum, 3, 1, name='conv13')

            x128_d_256_d = gen_deconv(att128_d, 2 * cnum, name='conv13_upsample_d')  # 96
            x128_d_256 = gen_deconv(att128, 2 * cnum, name='conv13_upsample')  # 96
            x128_d_256 = gen_conv(x128_d_256, cnum, 3, 1, name='conv14')
            x128_up_256 = gen_deconv(x128_d, cnum, name='conv14_upsample')
            x128_256 = gen_deconv(x64_up_128, cnum, name='conv15_upsample')  # 48
            x256_d = tf.concat([x128_d_256, x128_256, x128_up_256,x128_d_256_d], axis=3)  # concat256

            x256_d = gen_conv(x256_d, cnum, 3, 1, name='conv15')  # 256
            x = gen_conv(x256_d, cnum // 2, 3, 1, name='conv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
            x = tf.nn.tanh(x)

        return x


    #   判别器 1 -- 6 层
    def build_sn_pgan_discriminator_5(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            cnum = 64
            x = gen_snconv(x, cnum, 5, 2, name='conv1', training=training)
            xd1 = gen_snconv(x, cnum*2, 5, 2, name='conv2', training=training)
            xd2 = gen_snconv(xd1, cnum*4, 5, 2, name='conv3', training=training)
            xd3 = gen_snconv(xd2, cnum*4, 5, 2, name='conv4', training=training)
            xd4 = gen_snconv(xd3, cnum * 4, 5, 2, name='conv5', training=training)
            xd5 = gen_snconv(xd4, cnum * 4, 5, 2, name='conv6', training=training)
            xd5 = tf.contrib.layers.flatten(xd5)
            return x,xd1,xd2,xd3,xd4,xd5



    def build_sn_pgan_discriminator_4(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            cnum = 64
            x = gen_snconv(x, cnum, 5, 2, name='conv1_4', training=training)
            x = gen_snconv(x, cnum * 2, 5, 2, name='conv2_4', training=training)
            x_atrous1 = gen_conv(x, 4 * cnum, 3, rate=2, name='conv_atrous1')
            x_atrous2 = gen_conv(x_atrous1, 4 * cnum, 3, rate=4, name='conv_atrous2')
            x_atrous3 = gen_conv(x_atrous2, 4 * cnum, 3, rate=8, name='conv_atrous3')
            x = gen_snconv(x, cnum * 4, 5, 2, name='conv3_4', training=training)
            x = gen_snconv(x, cnum * 4, 5, 2, name='conv4_4', training=training)
            x1 = gen_snconv(x, cnum * 4, 5, 2, name='conv5_4', training=training)
            return x_atrous1,x_atrous2,x_atrous3,x1


    def build_graph_with_losses(self, batch_data,  config, training=True,
                                summary=False, reuse=False):
        batch_pos = batch_data / 127.5 - 1.

        irregular_mask = free_form_mask_tf(parts=5, im_size=(config.IMG_SHAPES[0], config.IMG_SHAPES[1]),
                                           maxBrushWidth=30, maxLength=80, maxVertex=16)
        mask = irregular_mask

        batch_incomplete = batch_pos * (1. - mask)

        x1 = self.build_inpaint_net(
            batch_pos, batch_incomplete, mask, config, reuse=reuse, training=training, padding=config.PADDING)

        batch_predicted = x1

        losses = {}

        batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)

        losses['l1_loss'] = tf.reduce_mean(
            tf.abs(batch_pos - x1))

        losses['vgg_loss'] = vgg_loss(batch_pos, batch_predicted)
        # Style Loss
        losses['style_loss'] = style_loss(batch_pos, batch_predicted)

# visualization
        scalar_summary('losses/l1_loss', losses['l1_loss'])
        scalar_summary('losses/vgg_loss', losses['vgg_loss'])
        scalar_summary('losses/style_loss', losses['style_loss'])
        viz_img = [batch_incomplete, batch_complete, batch_pos]
        images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_predicted_complete', config.VIZ_MAX_OUT)

        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [config.BATCH_SIZE * 2, 1, 1, 1])],axis=3)

        if config.GAN == 'sn_pgan':

            x, xd1, xd2, xd3, xd4, pos_neg_5 = self.build_sn_pgan_discriminator_5(batch_pos_neg, training=training,reuse=reuse)
            pos_global_5, neg_global_5 = tf.split(pos_neg_5, 2)
            xaa1, xaa2 = tf.split(x, 2)
            xbb1, xbb2 = tf.split(xd1, 2)
            xcc1, xcc2 = tf.split(xd2, 2)
            xdd1, xdd2 = tf.split(xd3, 2)
            xee1, xee2 = tf.split(xd4, 2)
            losses['fe_loss'] = tf.reduce_mean(tf.abs(xaa1 - xaa2))
            losses['fe_loss'] = tf.reduce_mean(tf.abs(xbb1 - xbb2))
            losses['fe_loss'] += tf.reduce_mean(tf.abs(xcc1 - xcc2))
            losses['fe_loss'] += tf.reduce_mean(tf.abs(xdd1 - xdd2))
            losses['fe_loss'] += tf.reduce_mean(tf.abs(xee1 - xee2))
            scalar_summary('fe_loss/fe_loss', losses['fe_loss'])

            x_atrous1,x_atrous2,x_atrous3,x1 = self.build_sn_pgan_discriminator_4(batch_pos_neg, training=training,reuse=reuse)

            xa1, xa2 = tf.split(x_atrous1, 2)
            xb1, xb2 = tf.split(x_atrous2, 2)
            xc1, xc2 = tf.split(x_atrous3, 2)
            xd1, xd2 = tf.split(x1, 2)
            losses['fe_loss1'] = tf.reduce_mean(tf.abs(xa1 - xa2))
            losses['fe_loss1'] = tf.reduce_mean(tf.abs(xb1 - xb2))
            losses['fe_loss1'] += tf.reduce_mean(tf.abs(xc1 - xc2))
            losses['fe_loss1'] += tf.reduce_mean(tf.abs(xd1 - xd2))
            scalar_summary('fe_loss/fe_loss1', losses['fe_loss1'])
            #sch
            g_loss_global_5, d_loss_global_5 = gan_wgan_loss(pos_global_5, neg_global_5, name='gan/global_gan_5')

            losses['g_loss_5'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global_5
            losses['d_loss_5'] = d_loss_global_5

            scalar_summary('convergence/g_loss_5', losses['g_loss_5'])
            gradients_summary(g_loss_global_5, batch_predicted, name='g_loss_global_5')

            interpolates_global = random_interpolates(
                tf.concat([batch_pos, tf.tile(mask, [config.BATCH_SIZE, 1, 1, 1])], axis=3),
                tf.concat([batch_complete, tf.tile(mask, [config.BATCH_SIZE, 1, 1, 1])],axis=3))

            x, xd1, xd2, xd3, xd4, dout_global_5 = self.build_sn_pgan_discriminator_5(interpolates_global,reuse=True)

            penalty_global_5 = gradients_penalty(interpolates_global, dout_global_5, mask= mask)
            losses['gp_loss_5'] = config.WGAN_GP_LAMBDA * penalty_global_5
            losses['d_loss_5'] = losses['d_loss_5'] + losses['gp_loss_5']
            losses['d_loss'] = losses['d_loss_5']
            scalar_summary('convergence/d_loss', losses['d_loss'])
            scalar_summary('convergence/d_loss_5', losses['d_loss_5'])
            scalar_summary('convergence/global_d_loss_5', d_loss_global_5)
            scalar_summary('gan_wgan_loss/gp_loss_5', losses['gp_loss_5'])
            scalar_summary('gan_wgan_loss/gp_penalty_global_5', penalty_global_5)

        losses['g_loss'] = config.GAN_LOSS_ALPHA_5 * losses['g_loss_5']
        losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        losses['g_loss'] += 0.00001 * losses['fe_loss']
        losses['g_loss'] += 0.00001 * losses['fe_loss1']
        losses['g_loss'] += config.VGG_LOSS_ALPHA * losses['vgg_loss']
        losses['g_loss'] += 0.0005 * losses['style_loss']

        scalar_summary('convergence/g_loss', losses['g_loss'])
# log
        logger.info('Set L1_LOSS_ALPHA to %f' % config.L1_LOSS_ALPHA)
        logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA_4)
        logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA_5)
        logger.info('Set AE_LOSS_ALPHA to %f' % config.AE_LOSS_ALPHA)
        logger.info('Set EDGE_LOSS_ALPHA to %f' % config.EDGE_LOSS_ALPHA)

        g_vars = tf.get_collection(  # g_vars
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(  # d_vars
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_server_graph(self, batch_data, reuse=False, is_training=False):
        """
        """
        masks_raw, batch_incomplete, comp_ori, batch_raw = tf.split(batch_data, 4, axis=2)
        batch_mask = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)
        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - batch_mask)
        x= self.build_inpaint_net(batch_pos, batch_incomplete, batch_mask, reuse=reuse, training=is_training,
            config=None)
        batch_predict = x
        # apply mask and reconstruct
        batch_complete = batch_predict * batch_mask + batch_incomplete * (1 - batch_mask)
        return batch_incomplete, batch_complete, batch_pos

    def build_server_graph_(self, batch_data, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        config = ng.Config('inpaint.yml')
        irregular_mask = free_form_mask_tf(parts=5, im_size=(config.IMG_SHAPES[0], config.IMG_SHAPES[1]),
                                           maxBrushWidth=30, maxLength=80, maxVertex=16)
        masks = irregular_mask
        masks_raw_o = (masks*255)/127.5 - 1.
        masks_raw_o = tf.concat([ masks_raw_o, masks_raw_o, masks_raw_o],axis=-1)
        batch_pos = batch_data/127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        batch_incomplete_o = batch_data * (1. - masks)
        batch_incomplete_o = (batch_incomplete_o + masks*255 )/127.5 - 1.
        # inpaint
        x1 = self.build_inpaint_net(
            batch_pos, batch_incomplete, masks, reuse=reuse, training=is_training,
            config=None)
        batch_predict = x1
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        return masks_raw_o, batch_incomplete_o,batch_complete,batch_pos