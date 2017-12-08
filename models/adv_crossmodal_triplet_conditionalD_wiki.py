from __future__ import print_function
import os, time, cPickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from random import shuffle
from scipy.spatial.distance import *
import sklearn.preprocessing
from base_model import BaseModel, BaseModelParams, BaseDataIter
import utils
from flip_gradient import flip_gradient
from sklearn.metrics.pairwise import cosine_similarity

class DataIter(BaseDataIter):
    def __init__(self, batch_size):
        BaseDataIter.__init__(self, batch_size)
        self.num_train_batch = 0
        self.num_test_batch = 0

        with open('./data/wikipedia_dataset/train_img_feats.pkl', 'rb') as f:
            self.train_img_feats = cPickle.load(f)
        with open('./data/wikipedia_dataset/train_txt_vecs.pkl', 'rb') as f:
            self.train_txt_vecs = cPickle.load(f)
        with open('./data/wikipedia_dataset/train_labels.pkl', 'rb') as f:
            self.train_labels = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_img_feats.pkl', 'rb') as f:
            self.test_img_feats = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_txt_vecs.pkl', 'rb') as f:
            self.test_txt_vecs = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_labels.pkl', 'rb') as f:
            self.test_labels = cPickle.load(f)
        
   
        self.num_train_batch = len(self.train_img_feats) / self.batch_size
        self.num_test_batch = len(self.test_img_feats) / self.batch_size

    def train_data(self):
        for i in range(self.num_train_batch):
            batch_img_feats = self.train_img_feats[i*self.batch_size : (i+1)*self.batch_size]
            batch_txt_vecs = self.train_txt_vecs[i*self.batch_size : (i+1)*self.batch_size]
            batch_labels = self.train_labels[i*self.batch_size : (i+1)*self.batch_size]
            yield batch_img_feats, batch_txt_vecs, batch_labels, i

    def test_data(self):
        for i in range(self.num_test_batch):
            batch_img_feats = self.test_img_feats[i*self.batch_size : (i+1)*self.batch_size]
            batch_txt_vecs = self.test_txt_vecs[i*self.batch_size : (i+1)*self.batch_size]
            batch_labels = self.test_labels[i*self.batch_size : (i+1)*self.batch_size]
            yield batch_img_feats, batch_txt_vecs, batch_labels, i


class ModelParams(BaseModelParams):
    def __init__(self):
        BaseModelParams.__init__(self)
        self.n_save_epoch = 10
        self.n_max_save = 10
        self.r_domain = 1.0
        self.r_pair = 0.1

        self.epoch = 500
        self.margin = .1
        self.alpha = 5
        self.batch_size = 64
        self.visual_feat_dim = 4096
        #self.word_vec_dim = 300
        self.word_vec_dim = 5000
        self.lr_total = 0.0001
        self.lr_emb = 0.0001
        self.lr_domain = 0.0001
        self.lr_pair = 0.0001
        self.top_k = 50
        self.semantic_emb_dim = 200
        self.dataset_name = 'wikipedia_dataset'
        self.model_name = 'adv_semantic_zsl'
        self.model_dir = 'adv_semantic_zsl_conditional_reductFirst_%d_%d_%d' % (self.visual_feat_dim, self.word_vec_dim, self.semantic_emb_dim)

        self.checkpoint_dir = 'checkpoint'
        self.sample_dir = 'samples'
        self.dataset_dir = './data'
        self.log_dir = 'logs'
 
    def update(self):
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)


class AdvCrossModalSimple(BaseModel):
    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)
        self.data_iter = DataIter(self.model_params.batch_size)

        self.tar_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.tar_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.pos_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.neg_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.unpair_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.unpair_img = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.pos_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.neg_txt = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.y = tf.placeholder(tf.int32, [self.model_params.batch_size,10])
        self.y_single = tf.placeholder(tf.int32, [self.model_params.batch_size,1])
        self.l = tf.placeholder(tf.float32, [])
        self.emb_v = self.visual_feature_embed(self.tar_img)
        self.emb_w = self.label_embed(self.tar_txt)
        self.emb_v_pos = self.visual_feature_embed(self.pos_img,reuse=True)
        self.emb_v_neg = self.visual_feature_embed(self.neg_img,reuse=True)
        self.emb_w_pos = self.label_embed(self.pos_txt,reuse=True)
        self.emb_w_neg = self.label_embed(self.neg_txt,reuse=True)
        self.emb_v_unpair = self.visual_feature_embed(self.unpair_img, reuse=True)
        self.emb_w_unpair = self.label_embed(self.unpair_txt, reuse=True)

        # triplet loss
        margin = self.model_params.margin
        alpha = self.model_params.alpha
        v_loss_pos = tf.reduce_sum(tf.nn.l2_loss(self.emb_v-self.emb_w_pos))
        v_loss_neg = tf.reduce_sum(tf.nn.l2_loss(self.emb_v-self.emb_w_neg))
        w_loss_pos = tf.reduce_sum(tf.nn.l2_loss(self.emb_w-self.emb_v_pos))
        w_loss_neg = tf.reduce_sum(tf.nn.l2_loss(self.emb_w-self.emb_v_neg))
        self.triplet_loss = tf.maximum(0.,margin+alpha*v_loss_pos-v_loss_neg) + tf.maximum(0.,margin+alpha*w_loss_pos-w_loss_neg)

        logits_v = self.label_classifier(self.emb_v)
        logits_w = self.label_classifier(self.emb_w, reuse=True)
        self.label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_v) + \
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_w)
        self.label_loss = tf.reduce_mean(self.label_loss)
        self.label_img_pred = tf.argmax(logits_v, 1)
        self.label_img_acc = tf.reduce_mean(tf.cast(tf.equal(self.label_img_pred, tf.argmax(self.y, 1)), tf.float32))
        self.label_shape_pred = tf.argmax(logits_w, 1)
        self.label_shape_acc = tf.reduce_mean(
            tf.cast(tf.equal(self.label_shape_pred, tf.argmax(self.y, 1)), tf.float32))
        self.label_class_acc = tf.divide(tf.add(self.label_img_acc, self.label_shape_acc), 2.0)
        self.emb_loss = 100*self.label_loss + self.triplet_loss
        # self.emb_v_class = self.domain_classifier(self.emb_v, self.l)
        # self.emb_w_class = self.domain_classifier(self.emb_w, self.l, reuse=True)

        all_emb_v = tf.concat([tf.ones([self.model_params.batch_size, 1]),
                                   tf.zeros([self.model_params.batch_size, 1])], 1)
        all_emb_w = tf.concat([tf.zeros([self.model_params.batch_size, 1]),
                                   tf.ones([self.model_params.batch_size, 1])], 1)
        # self.domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class, labels=all_emb_w) + \
        #     tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_w_class, labels=all_emb_v)
        # self.domain_class_loss = tf.reduce_mean(self.domain_class_loss)
        # self.domain_img_class_acc = tf.equal(tf.greater(0.5, self.emb_v_class), tf.greater(0.5, all_emb_w))
        # self.domain_shape_class_acc = tf.equal(tf.greater(self.emb_w_class, 0.5), tf.greater(all_emb_v, 0.5))
        # self.domain_class_acc = tf.reduce_mean(
        #     tf.cast(tf.concat([self.domain_img_class_acc, self.domain_shape_class_acc], axis=0), tf.float32))

        # conditional D loss
        self.img_conditional_v_pred = self.img_conditional_classifier(self.tar_img, self.emb_v, self.l)
        self.img_conditional_w_pred = self.img_conditional_classifier(self.tar_img, self.emb_w, self.l, reuse=True)
        self.img_conditional_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.img_conditional_v_pred, labels=all_emb_w) + \
                                    tf.nn.softmax_cross_entropy_with_logits(logits=self.img_conditional_w_pred, labels=all_emb_v)
        self.img_conditional_loss = tf.reduce_mean(self.img_conditional_loss)
        self.img_conditional_acc = tf.divide(tf.add(self.acc_op(self.img_conditional_v_pred, all_emb_w), self.acc_op(self.img_conditional_w_pred, all_emb_v)), 2.0)

        self.label_conditional_v_pred = self.label_conditional_classifier(self.tar_txt, self.emb_v, self.l)
        self.label_conditional_w_pred = self.label_conditional_classifier(self.tar_txt, self.emb_w, self.l, reuse=True)
        self.label_conditional_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.label_conditional_v_pred, labels=all_emb_w) + \
            tf.nn.softmax_cross_entropy_with_logits(logits=self.label_conditional_w_pred, labels=all_emb_v)
        self.label_conditional_loss = tf.reduce_mean(self.label_conditional_loss)
        self.label_conditional_acc = tf.divide(tf.add(self.acc_op(self.label_conditional_v_pred, all_emb_w), self.acc_op(self.label_conditional_w_pred, all_emb_v)), 2.0)


        # Pair D loss
        # self.emb_pair_pred = self.pair_classifier(self.emb_v, self.emb_w, self.l)
        # self.emb_unpair_pred = self.pair_classifier(tf.concat([self.emb_v, self.emb_w_unpair], axis=0), tf.concat([self.emb_v_unpair, self.emb_w], axis=0), self.l, reuse=True)
        # pair_labels, unpair_labels = tf.ones([self.model_params.batch_size, 1]), tf.zeros([self.model_params.batch_size*2, 1])
        # self.pair_loss = tf.concat([tf.nn.sigmoid_cross_entropy_with_logits(logits=self.emb_pair_pred, labels=pair_labels), \
        #                  tf.nn.sigmoid_cross_entropy_with_logits(logits=self.emb_unpair_pred, labels=unpair_labels)], axis=0)
        # self.pair_loss = tf.reduce_mean(self.pair_loss)
        # self.pair_acc = tf.equal(tf.greater(pair_labels, 0.5), tf.greater(self.emb_pair_pred, 0.5))
        # self.unpair_acc = tf.equal(tf.greater(0.5, unpair_labels), tf.greater(0.5, self.emb_unpair_pred))
        # self.pair_all_acc = tf.reduce_mean(tf.cast(tf.concat([self.pair_acc, self.unpair_acc], axis=0), tf.float32))
        # self.pair_acc = tf.reduce_mean(tf.cast(self.pair_acc, tf.float32))
        # self.unpair_acc = tf.reduce_mean(tf.cast(self.unpair_acc, tf.float32))

        # TODO G loss as paper
        # maximize domain class loss and minimize pair loss
        self.G_loss = self.emb_loss + self.model_params.r_domain * (self.img_conditional_loss + self.label_conditional_loss) # + self.model_params.r_pair * self.pair_loss

        self.t_vars = tf.trainable_variables()
        self.vf_vars = [v for v in self.t_vars if 'vf_' in v.name]
        self.le_vars = [v for v in self.t_vars if 'le_' in v.name]
        self.dc_vars = [v for v in self.t_vars if 'dc_' in v.name]
        self.icc_vars = [v for v in self.t_vars if 'icc_' in v.name] # image conditional D
        self.lcc_vars = [v for v in self.t_vars if 'lcc_' in v.name] # label conditional D
        self.lc_vars = [v for v in self.t_vars if 'lc_' in v.name]
        self.pc_vars = [v for v in self.t_vars if 'pc_' in v.name] # pair

    def acc_op(self, pred, label, threshold=0.5):
        return tf.reduce_mean(tf.cast(tf.equal(tf.greater(pred, threshold), tf.greater(label, threshold)), tf.float32))

    def visual_feature_embed(self, X, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(X, 2000, scope='vf_fc_0'))
            # net = tf.nn.tanh(slim.fully_connected(net, 200, scope='vf_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='vf_fc_2'))
        return net

    def label_embed(self, L, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(L, 500, scope='le_fc_0'))
            # net = tf.nn.tanh(slim.fully_connected(net, 100, scope='le_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='le_fc_2'))
        return net 
    def label_classifier(self, X, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = slim.fully_connected(X, 10, scope='lc_fc_0')
        return net         
    def domain_classifier(self, E, l, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            net = slim.fully_connected(E, self.model_params.semantic_emb_dim/2, scope='dc_fc_0')
            net = slim.fully_connected(net, self.model_params.semantic_emb_dim/4, scope='dc_fc_1')
            net = slim.fully_connected(net, 2, scope='dc_fc_2')
        return net

    def pair_classifier(self, V, W, l, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            V, W = flip_gradient(V, l), flip_gradient(W, l)
            net = slim.fully_connected(tf.concat([V, W], axis=1), self.model_params.semantic_emb_dim / 2, scope='pc_fc_0')
            net = slim.fully_connected(net, self.model_params.semantic_emb_dim/4, scope='pc_fc_1')
            net = slim.fully_connected(net, 1, scope='pc_fc_2')
        return net
    def label_conditional_classifier(self, ori_W, E, l, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            net = slim.fully_connected(ori_W, 40, scope="lcc_fc_0")
            net = slim.fully_connected(tf.concat([E, net], axis=1), 10, scope='lcc_fc_1')
            # net = slim.fully_connected(tf.concat([E, ori_W], axis=1), 512, scope='lcc_fc_0')
            # net = slim.fully_connected(net, 100, scope='lcc_fc_1')
            net = slim.fully_connected(net, 2, scope='lcc_fc_2')
        return net

    def img_conditional_classifier(self, ori_V, E, l, is_training=False, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            # net = slim.fully_connected(tf.concat([E, ori_V], axis=1), 512, scope='icc_fc_0')
            # net = slim.fully_connected(net, 100, scope='icc_fc_1')
            net = slim.fully_connected(ori_V, 40, scope='icc_fc_0')
            net = slim.fully_connected(tf.concat([E, net], axis=1), 10, scope='icc_fc_1')
            net = slim.fully_connected(net, 2, scope='icc_fc_2')
        return net

    def find_neg_pair(self, fcs1, fcs2):
        """
        find negative pair for each value of fcs1 from fcs2
        :param fcs1:
        :param fcs2:
        :return:
        """
        fcs1_np, fcs2_np = np.array(fcs1), np.array(fcs2)
        assert fcs1_np.shape[0] == fcs2_np.shape[0]
        size = fcs1_np.shape[0]
        sims = cosine_similarity(fcs1_np, fcs1_np)
        result = []
        for i in range(size):
            sims[i][i] = -1.0
            neg_index = np.argmax(sims[i, :], axis=0).astype(int)
            result.append(neg_index)
        return fcs2_np[result]

    def train(self, sess):
        #self.check_dirs()
 
        # total_loss = self.emb_loss + self.domain_class_loss
        # total_train_op = tf.train.AdamOptimizer(
        #     learning_rate=self.model_params.lr_total,
        #     beta1=0.5).minimize(total_loss)
        emb_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_emb,
            beta1=0.5).minimize(self.G_loss, var_list=self.le_vars+self.vf_vars)
        img_conditionalD_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_domain,
            beta1=0.5).minimize(self.img_conditional_loss, var_list=self.icc_vars)
        label_conditionalD_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_domain,
            beta1=0.5).minimize(self.label_conditional_loss, var_list=self.lcc_vars)
        # pair_train_op = tf.train.AdamOptimizer(
        #     learning_rate=self.model_params.lr_pair,
        #     beta1=0.5).minimize(self.pair_loss, var_list=self.pc_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

        start_time = time.time()
        map_avg_ti = []
        map_avg_it = []
        adv_loss = []
        emb_loss = []
        for epoch in range(self.model_params.epoch):
            p = float(epoch) / self.model_params.epoch
            l = 2. / (1. + np.exp(-10. * p)) - 1
            for batch_feat, batch_vec, batch_labels, idx in self.data_iter.train_data():
                # create one-hot labels
                batch_labels_ = batch_labels - np.ones_like(batch_labels)
                label_binarizer = sklearn.preprocessing.LabelBinarizer()
                label_binarizer.fit(range(max(batch_labels_)+1))
                b = label_binarizer.transform(batch_labels_)
                adj_mat = np.dot(b,np.transpose(b))
                mask_mat = np.ones_like(adj_mat) - adj_mat
                img_sim_mat = mask_mat*cosine_similarity(batch_feat,batch_feat)
                txt_sim_mat = mask_mat*cosine_similarity(batch_vec,batch_vec)
                img_neg_txt_idx = np.argmax(img_sim_mat,axis=1).astype(int)
                txt_neg_img_idx = np.argmax(txt_sim_mat,axis=1).astype(int)
                #print('{0}'.format(img_neg_txt_idx.shape)
                batch_vec_ = np.array(batch_vec)
                batch_feat_ = np.array(batch_feat)                
                img_neg_txt = batch_vec_[img_neg_txt_idx,:]
                txt_neg_img = batch_feat_[txt_neg_img_idx,:]
                img_unpair_txt = self.find_neg_pair(batch_feat, batch_vec)
                txt_unpair_img = self.find_neg_pair(batch_vec, batch_feat)
                #_, label_loss_val, dissimilar_loss_val, similar_loss_val = sess.run([total_train_op, self.label_loss, self.dissimilar_loss, self.similar_loss], feed_dict={self.tar_img: batch_feat, self.tar_txt: batch_vec, self.y: b, self.y_single: np.transpose([batch_labels]),self.l: l})
                # TODO no domain classifier
                # sess.run([emb_train_op, domain_train_op],
                # Update
                for i in range(1):
                    sess.run([emb_train_op],
                              feed_dict={self.tar_img: batch_feat,
                              self.tar_txt: batch_vec,
                              self.pos_txt: batch_vec,
                              self.neg_txt: img_neg_txt,
                              self.pos_img: batch_feat,
                              self.neg_img: txt_neg_img,
                              self.unpair_img: txt_unpair_img,
                              self.unpair_txt: img_unpair_txt,
                              self.y: b,
                              self.y_single: np.transpose([batch_labels]),
                              self.l: l})
                sess.run([img_conditionalD_train_op],
                         feed_dict={self.tar_img: batch_feat,
                                    self.tar_txt: batch_vec,
                                    self.pos_txt: batch_vec,
                                    self.neg_txt: img_neg_txt,
                                    self.pos_img: batch_feat,
                                    self.neg_img: txt_neg_img,
                                    self.unpair_img: txt_unpair_img,
                                    self.unpair_txt: img_unpair_txt,
                                    self.y: b,
                                    self.y_single: np.transpose([batch_labels]),
                                    self.l: l})
                sess.run([label_conditionalD_train_op],
                         feed_dict={self.tar_img: batch_feat,
                                    self.tar_txt: batch_vec,
                                    self.pos_txt: batch_vec,
                                    self.neg_txt: img_neg_txt,
                                    self.pos_img: batch_feat,
                                    self.neg_img: txt_neg_img,
                                    self.unpair_img: txt_unpair_img,
                                    self.unpair_txt: img_unpair_txt,
                                    self.y: b,
                                    self.y_single: np.transpose([batch_labels]),
                                    self.l: l})
                label_loss_val, triplet_loss_val, emb_loss_val, img_conditional_loss_val, label_conditional_loss_val, g_loss_val, label_acc_val, img_conditional_acc_val, label_conditional_acc_val = \
                    sess.run([self.label_loss, self.triplet_loss, self.emb_loss, self.img_conditional_loss, self.label_conditional_loss, self.G_loss, self.label_class_acc, self.img_conditional_acc, self.label_conditional_acc],
                          feed_dict={self.tar_img: batch_feat,
                          self.tar_txt: batch_vec,
                          self.pos_txt: batch_vec,
                          self.neg_txt: img_neg_txt,
                          self.pos_img: batch_feat,
                          self.neg_img: txt_neg_img,
                          self.unpair_img: txt_unpair_img,
                          self.unpair_txt: img_unpair_txt,
                          self.y: b,
                          self.y_single: np.transpose([batch_labels]),
                          self.l: l})
                print('Epoch: [%2d][%4d/%4d] time: %4.4f, emb_loss: %.8f, img_conditional_loss: %.8f, label_conditional_loss: %.8f, label_loss: %.8f, triplet_loss: %.8f, g_loss: %.8f, label_acc:%.8f, img_conditional_acc:%.8f, label_conditional_acc:%.8f' %(
                    epoch, idx, self.data_iter.num_train_batch, time.time() - start_time, emb_loss_val, img_conditional_loss_val, label_conditional_loss_val, label_loss_val, triplet_loss_val, g_loss_val, label_acc_val, img_conditional_acc_val, label_conditional_acc_val
                ))
            if (epoch+1) % self.model_params.n_save_epoch == 0:
                self.save(epoch+1, sess)
            # if epoch == (self.model_params.epoch - 1):
            #    self.emb_v_eval, self.emb_w_eval = sess.run([self.emb_v, self.emb_w],
            #             feed_dict={
            #                 self.tar_img: batch_feat,
            #                 self.tar_txt: batch_vec,
            #                 self.y: b,
            #                 self.y_single: np.transpose([batch_labels]),
            #                 self.l: l})
            #    with open('./data/wikipedia_dataset/train_img_emb.pkl', 'wb') as f:
            #        cPickle.dump(self.emb_v_eval, f, cPickle.HIGHEST_PROTOCOL)
            #    with open('./data/wikipedia_dataset/train_txt_emb.pkl', 'wb') as f:
            #        cPickle.dump(self.emb_w_eval, f, cPickle.HIGHEST_PROTOCOL)
        self.save(epoch, sess)
    def eval_random_rank(self):
        start = time.time()
        #with open('./data/wikipedia_dataset/test_labels.pkl', 'rb') as fpkl:
        #    test_labels = cPickle.load(fpkl)
        with open('./data/wiki_shallow/L_te.pkl', 'rb') as fpkl:
            test_labels = cPickle.load(fpkl)
        k = self.model_params.top_k
        avg_precs = []
        for i in range(len(test_labels)):
            query_label = test_labels[i]

            # distances and sort by distances
            sorted_idx = range(len(test_labels))
            shuffle(sorted_idx)

            # for each k do top-k
            precs = []
            for topk in range(1, k + 1):
                hits = 0
                top_k = sorted_idx[0 : topk]
                if query_label != test_labels[top_k[-1]]:
                    continue
                for ii in top_k:
                    retrieved_label = test_labels[ii]
                    if query_label != retrieved_label:
                        hits += 1
                precs.append(float(hits) / float(topk))
            avg_precs.append(np.sum(precs) / float(k))
        mean_avg_prec = np.mean(avg_precs)
        print('[Eval - random] mAP: %f in %4.4fs' % (mean_avg_prec, (time.time() - start)))
        

    def eval(self, sess):
        start = time.time()
        self.saver = tf.train.Saver()
        self.load(sess)

        test_img_feats_trans = []
        test_txt_vecs_trans = []
        test_labels = []
        test_img_feats, test_txt_feats = np.array([]), np.array([])
        for feats, vecs, labels, i in self.data_iter.test_data():
            test_img_feats = np.concatenate([test_img_feats, feats], axis=0) if test_img_feats.shape[0] > 0 else np.array(feats)
            test_txt_feats = np.concatenate([test_txt_feats, vecs], axis=0) if test_txt_feats.shape[0] > 0 else np.array(vecs)
            feats_trans = sess.run(self.emb_v, feed_dict={self.tar_img: feats})
            vecs_trans = sess.run(self.emb_w, feed_dict={self.tar_txt: vecs})
            test_labels += labels
            for ii in range(len(feats)):
                test_img_feats_trans.append(feats_trans[ii])
                test_txt_vecs_trans.append(vecs_trans[ii])
        # img_txt_D = []
        # for feats in test_img_feats:
        #     pair_sim = sess.run(self.emb_pair_pred, feed_dict={self.tar_img: np.tile(feats, (test_txt_feats.shape[0], 1)), self.tar_txt: test_txt_feats})
        #     img_txt_D.append(pair_sim.tolist())
        # img_txt_D = np.reshape(img_txt_D, [len(img_txt_D), len(img_txt_D[0])])
        # pair_labels = np.zeros(img_txt_D.shape)
        # for i in range(pair_labels.shape[0]):
        #     pair_labels[i][i] = 1.0
        # test_pair_acc = np.mean(np.equal(pair_labels > 0.5, img_txt_D > 0.5).astype(float))
        # print("eval pair acc:", test_pair_acc)
        # dim2dis_func = np.vectorize(lambda s: 1.0  - s)
        # img_txt_D = dim2dis_func(img_txt_D)
        # np.save('img_txt_D_train', img_txt_D)
        # print("img2shape pair:", img2shape(img_txt_D,
        #           np.arange(0, test_img_feats.shape[0]), top_k=self.model_params.top_k,
        #           tag="acmr-triplet-img2shape", save_dir='./result'))
        # print("txt2img pair:", img2shape(np.transpose(img_txt_D),
        #                                  np.arange(0, test_txt_feats.shape[0]), top_k=self.model_params.top_k,
        #                                  tag="acmr-triplet-txt2img", save_dir='./result'))

        test_img_feats_trans = np.asarray(test_img_feats_trans)
        test_txt_vecs_trans = np.asarray(test_txt_vecs_trans)
        test_feats_trans = np.concatenate((test_img_feats_trans[0:1000], test_txt_vecs_trans[-1000:]))
        np.save('./result/feature/test_joint_img_feat', test_img_feats_trans)
        np.save('./result/feature/test_joint_txt_feat', test_img_feats_trans)
        np.save('./result/feature/test_labels', np.asarray(test_labels))
        #with open('./data/wikipedia_dataset/test_feats_transformed.pkl', 'wb') as f:
        #    cPickle.dump(test_feats_trans, f, cPickle.HIGHEST_PROTOCOL)        
        with open('./data/wiki_shallow/test_feats_transformed.pkl', 'wb') as f:
            cPickle.dump(test_feats_trans, f, cPickle.HIGHEST_PROTOCOL)                   
        print('[Eval] transformed test features in %4.4f' % (time.time() - start))
        top_k = self.model_params.top_k
        avg_precs = []
        all_precs = []
        for k in range(1, top_k+1):
            for i in range(len(test_txt_vecs_trans)):
                query_label = test_labels[i]

                # distances and sort by distances
                wv = test_txt_vecs_trans[i]
                diffs = test_img_feats_trans - wv
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                #for each k do top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0 : topk]
                    if np.sum(query_label) != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        if np.sum(retrieved_label) == query_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            all_precs.append(mean_avg_prec)
        print('[Eval - txt2img] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))
        print(all_precs)
        t2i = all_precs[0]
        #with open('./data/wikipedia_dataset/txt2img_all_precision.pkl', 'wb') as f:
        #    cPickle.dump(all_precs, f, cPickle.HIGHEST_PROTOCOL)     
        with open('./data/wiki_shallow/txt2img_all_precision.pkl', 'wb') as f:
            cPickle.dump(all_precs, f, cPickle.HIGHEST_PROTOCOL)                  

        avg_precs = []
        all_precs = []

        for k in range(1, self.model_params.top_k+1):
            for i in range(len(test_img_feats_trans)):
                query_img_feat = test_img_feats_trans[i]
                ground_truth_label = test_labels[i]

                # calculate distance and sort
                diffs = test_txt_vecs_trans - query_img_feat
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                # for each k in top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0 : topk]
                    if np.sum(ground_truth_label) != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        if np.sum(ground_truth_label) == retrieved_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            all_precs.append(mean_avg_prec)            
        print('[Eval - img2txt] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))
        print(all_precs)
        
        
        #with open('./data/wikipedia_dataset/text_words_map.pkl', 'wb') as f:
        #    cPickle.dump(all_precs, f, cPickle.HIGHEST_PROTOCOL)
        with open('./data/wiki_shallow/text_words_map.pkl', 'wb') as f:
            cPickle.dump(all_precs, f, cPickle.HIGHEST_PROTOCOL)        
        #Text query    

        #with open('./data/wikipedia_dataset/text_words_map.pkl', 'rb') as f:
        #    txt_words = cPickle.load(f)
        #with open('./data/wikipedia_dataset/test_img_words.pkl', 'rb') as f:
        #    img_words = cPickle.load(f)
        #with open('./data/wikipedia_dataset/test_txt_files.pkl', 'rb') as f:
        #    test_txt_names = cPickle.load(f)
        #with open('./data/wikipedia_dataset/test_img_files.pkl', 'rb') as f:
        #    test_img_names = cPickle.load(f)   
        with open('./data/wikipedia_dataset/text_words_map.pkl', 'rb') as f:
            txt_words = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_img_words.pkl', 'rb') as f:
            img_words = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_txt_files.pkl', 'rb') as f:
            test_txt_names = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_img_files.pkl', 'rb') as f:
            test_img_names = cPickle.load(f)

        img2shape_pair_acc = img2shape(cdist(test_img_feats_trans, test_txt_vecs_trans),
                                       np.arange(0, test_img_feats_trans.shape[0]), top_k=self.model_params.top_k,
                                       tag="acmr-triplet-img2shape", save_dir='./result')
        shape2img_pair_acc = img2shape(cdist(test_txt_vecs_trans, test_img_feats_trans), np.arange(0, test_img_feats_trans.shape[0]), top_k=self.model_params.top_k,
                                       tag="acmr-triplet-txt2img", save_dir='./result')
        print('[Test - img2shape pair(Edu):]', img2shape_pair_acc)
        print('[Test - txt2img pair(Edu):]', shape2img_pair_acc)
        print('[Eval] finished precision-scope in %4.4fs' % (time.time() - start))

def dis_img_shape(img_fcs, shape_fcs):
    return cdist(img_fcs, shape_fcs)
def dis_shape_img(shape_fcs, img_fcs):
    return cdist(shape_fcs, img_fcs)

def img2shape(D, pair_img_model, top_k=50, tag="all", save_dir=""):
    # D = cdist(img_fcs, shape_fcs)
    image_N = D.shape[0]
    image2shape_retrieval_ranking = []
    for k in range(image_N):
        distances = D[k, :]  # [float(distance) for distance in line.strip().split()]
        ranking = range(len(distances))
        ranking.sort(key=lambda rank: distances[rank])
        # print 'image %d \t retrieval: %d' % (k, ranking.index(pair_img_model[k]) + 1)
        image2shape_retrieval_ranking.append(ranking.index(pair_img_model[k]) + 1)
    image2shape_topK_accuracies = []
    for topK in range(top_k):
        n = sum([r <= topK + 1 for r in image2shape_retrieval_ranking])
        image2shape_topK_accuracies.append(n / float(image_N))
    if save_dir and len(save_dir) > 0:
        np.savetxt(os.path.join(save_dir, 'image2txt_top%d_accuracy_%s.txt'%(top_k, tag)), image2shape_topK_accuracies, fmt='%.4f')
    return image2shape_topK_accuracies