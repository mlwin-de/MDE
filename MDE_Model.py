from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch as torch
import numpy as np
import sys
import numpy.matlib as matlib
import os
sys.path.append('../')
import HandleKGDBs.ReadDataset as ReadDataset
import timeit , csv
from multiprocessing import Process,Queue


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-d", "--dataset", dest="dataset",
                    help="training over dataset. It can be 'WN18' 'FB15' 'WN18RR' 'FB15K237' ", metavar="dataset")
parser.add_argument("-t", "--task",
                    dest="task", default=True,
                    help="set to perform 'train' or 'test' ")

args = parser.parse_args()

class SampleGenerator(nn.Module):
    def __init__(self, dataset_setting):
        super(SampleGenerator, self).__init__()  # Calling Super Class's constructor
        self.all_samples = {}
        if dataset_setting.dataset == "FB15":
            self.get_freebase()
        elif dataset_setting.dataset == "FB15K237":
            self.get_FB15K237()
        elif dataset_setting.dataset == "WN18":
            self.get_wordnet()
        elif dataset_setting.dataset == "WN18RR":
            self.get_wordnetRR()

    def get_freebase(self):
        sample_class = ReadDataset.readDataset()
        self.dbPath = sample_class.input_directory
        sample_class.read_FB15K()
        self.training_sample = torch.tensor(sample_class.train_data_fb)
        self.validation_samples = torch.tensor(sample_class.validation_data_fb)
        self.test_samples = torch.tensor(sample_class.test_data_fb)
        self.entity2id = sample_class.entity2id_fb
        self.relation2id = sample_class.relation2id_fb
        self.get_negative_samples = np.vectorize(self.get_negative_sample, signature='(),(),()->(n)')
        # print self.training_sample.shape[0]

    def get_FB15K237(self):
        sample_class = ReadDataset.readDataset()
        self.dbPath = sample_class.input_directory
        sample_class.read_FB15K237()
        self.training_sample = torch.tensor(sample_class.train_data_fb)
        self.validation_samples = torch.tensor(sample_class.validation_data_fb)
        self.test_samples = torch.tensor(sample_class.test_data_fb)
        self.entity2id = sample_class.entity2id_fb
        self.relation2id = sample_class.relation2id_fb
        self.get_negative_samples = np.vectorize(self.get_negative_sample, signature='(),(),()->(n)')

    def get_wordnet(self):
        sample_class = ReadDataset.readDataset()
        self.dbPath = sample_class.input_directory
        sample_class.read_WN18()
        self.training_sample = torch.tensor(sample_class.train_data_wn)
        self.validation_samples = torch.tensor(sample_class.validation_data_wn)
        self.test_samples = torch.tensor(sample_class.test_data_wn)
        self.entity2id = sample_class.entity2id_wn
        self.relation2id = sample_class.relation2id_wn
        self.get_negative_samples = np.vectorize(self.get_negative_sample, signature='(),(),()->(n)')

    def get_wordnetRR(self):
        sample_class = ReadDataset.readDataset()
        self.dbPath = sample_class.input_directory
        sample_class.read_WN18RR()
        self.training_sample = torch.tensor(sample_class.train_data_wn)
        self.validation_samples = torch.tensor(sample_class.validation_data_wn)
        self.test_samples = torch.tensor(sample_class.test_data_wn)
        self.entity2id = sample_class.entity2id_wn

        self.relation2id = sample_class.relation2id_wn
        self.get_negative_samples = np.vectorize(self.get_negative_sample, signature='(),(),()->(n)')

    def get_random_training_samples(self, sample_size):
        training_index = np.random.randint(0, self.training_sample.shape[0], size=sample_size)
        training = self.training_sample[training_index]
        return training

    def make_all_samples_dic(self):
        self.all_samples = {}
        for i in self.training_sample:
            self.all_samples[i[0].item(), i[1].item(), i[2].item()] = 1
        for i in self.validation_samples:
            self.all_samples[i[0].item(), i[1].item(), i[2].item()] = 1
        for i in self.test_samples:
            self.all_samples[i[0].item(), i[1].item(), i[2].item()] = 1
        return

    # add lables and makes one negative sample per positive sample
    def get_negative_sample(self, h, t, r):
        pos = np.random.randint(10, size=1)[0]
        pos2 = np.random.randint(self.entity2id.shape[0], size=1)[0]
        curr_entity = self.entity2id[pos2]
        if (pos < 5):
            X2 = [h, curr_entity, r]
        else:
            X2 = [curr_entity, t, r]
        X2 = np.array(X2)
        return X2

    def reshuffle(self):
        training_index = torch.randperm(self.training_sample.shape[0])
        training = self.training_sample[training_index]
        self.training_sample = training

    # returns splitted training samples
    def get_splitted_set_training_batchs(self, sample_size):
        return torch.split(self.training_sample, sample_size)


class MDE_Model(nn.Module):
    def __init__(self, config, sampler , embeddings):
        super(MDE_Model, self).__init__()  # Calling Super Class's constructor
        self.config = config
        self.delta = 0.0
        self.delta_neg = 0.0
        self.embeddings = embeddings
        self.sampler = sampler
        self.iter_ = 0
        self.x_drawing = np.zeros(1000)
        self.out_draw = np.zeros(1000)
        self.out_draw_negative_ = np.zeros(1000)
        self.loss_p_per_triple = 10 # initialize with bigger value than threshold for the first epoch
        self.loss_n_per_triple = 10
        self.batch_counter = 0
        self.gamma_1 = 0
        self.gamma_2 = 0
        self.beta1 = 0
        self.beta2 = 0

    def make_splitted_batch(self):
        self.batch = self.sampler.get_splitted_set_training_batchs(self.config.x_train_bach_size)

    def get_splitted_variables(self):
        self.x_train = Variable(self.batch[self.batch_counter])
        self.x_train_negative = Variable(torch.tensor(
            self.sampler.get_negative_samples(self.x_train[:, 0], self.x_train[:, 1], self.x_train[:, 2])))

    def get_variables(self):
        self.x_train = Variable(self.sampler.get_random_training_samples(self.config.x_train_bach_size))
        self.x_train_negative = Variable(torch.tensor(
            self.sampler.get_negative_samples(self.x_train[:, 0], self.x_train[:, 1], self.x_train[:, 2])))
        

    # limit-based scoring loss:  Learning knowledge embeddings by combining
    # limit-based scoring loss
    # Margin ranking loss(x_pos, x_neg) = max(0, (x_pos - x_neg) + margin)
    # Limit-based loss(x_pos, x_neg) = max(0, (x1_pos - margin_1)) + miu * max(0, (x1_neg - margin_2)

    # the default setting was not converging well so I played with it
    # This edition(convex combination) loss(x_pos, x_neg) = miu_1 *  max(0, (x1_pos - margin_1)) + miu_2 * max(0, (x1_neg - margin_2)
    def loss_func(self, p_score, n_score):
        criterion = nn.MarginRankingLoss(self.config.margin, reduction= 'sum')
        y =  torch.Tensor([-1.0])
        lambda_pos =  torch.Tensor([self.gamma_1 - self.delta])
        lambda_neg =  torch.Tensor([self.gamma_2 - self.delta_neg])

        pos_loss = criterion(p_score, lambda_pos, y)
        neg_loss = criterion(n_score, lambda_neg, -y)
        loss = self.beta1 * pos_loss + self.beta2 * neg_loss
        return loss, pos_loss, neg_loss

    def update_limits(self):
        print ("in update_limits", self.loss_p_per_triple * self.beta1 , self.loss_n_per_triple * self.beta2)
        if self.loss_p_per_triple * self.beta1 < 0.1 and self.delta < self.gamma_1:  #0.5
            self.delta = self.delta + 0.1
            print ("reducing gamma")
            print ("new gamma and gamma-neg:", self.gamma_1 - self.delta)
            if self.loss_n_per_triple > 0.05 and self.delta_neg < self.gamma_2 - 0.1:
                self.delta_neg = self.delta_neg + 0.1
            print ("reducing gamma neg")
            print ("new gamma-neg:", self.gamma_2 - self.delta_neg)
        elif self.loss_n_per_triple * self.beta2 < 0.1:  #0.05 # and self.delta_neg > 0.1: #and self.delta_neg <= self.delta - 0.1
            self.delta_neg = self.delta_neg - 0.1
            print ("adding gamma negative")
            print ("new gamma-neg:", self.gamma_2 - self.delta_neg)

    def init_loss_parameters(self, gamma_1, gamma_2, beta1, beta2):
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self):
        if self.config.batch_type == "random_batch":
            self.get_variables()
        elif self.config.batch_type == "pre_splitted_batch":
            #print self.batch_number
            if self.batch_counter == 0:
                self.sampler.reshuffle()
                self.make_splitted_batch()
                #print self.x_train[:, 0]
            self.batch_counter = self.batch_counter + 1
            self.get_splitted_variables()

        h = self.embeddings.get_vectorised_values_entity(self.x_train[:, 0])
        t = self.embeddings.get_vectorised_values_entity(self.x_train[:, 1])
        r = self.embeddings.get_vectorised_values_relation(self.x_train[:, 2])

        h_negative = self.embeddings.get_vectorised_values_entity(self.x_train_negative[:, 0])
        t_negative = self.embeddings.get_vectorised_values_entity(self.x_train_negative[:, 1])
        r_negative = self.embeddings.get_vectorised_values_relation(self.x_train_negative[:, 2])

        score_pos = self.predict(h , t , r)#r + alpha
        score_neg = self.predict(h_negative ,t_negative, r_negative)# r_negative + alpha
        loss, loss_p, loss_n = self.loss_func(score_pos , score_neg)
        return loss , loss_p, loss_n

    def predict(self, h,t,r):
        psi = 1.2
        a = h + r - t
        b = h + t - r
        c = t + r - h
        d = h - r * t
        #score_a = torch.norm((a[0, :, :]), p=2, dim=1)
        #score_b = torch.norm((b[1, :, :]), p=2, dim=1)
        #score_c = torch.norm((c[2, :, :]), p=2, dim=1)
        #score_d = (torch.norm((d[3, :, :]), p=2, dim=1))
        score_a = (torch.norm((a[0, :, :]), p=2, dim=1) + torch.norm((a[4, :, :]), p=2, dim=1)) / 2.0
        score_b = (torch.norm((b[1, :, :]), p=2, dim=1) + torch.norm((b[5, :, :]), p=2, dim=1)) / 2.0
        score_c = (torch.norm((c[2, :, :]), p=2, dim=1) + torch.norm((c[6, :, :]), p=2, dim=1)) / 2.0
        score_d = (torch.norm((d[3, :, :]), p=2, dim=1) + torch.norm((d[7, :, :]), p=2, dim=1)) / 2.0
        return (1.5 * score_a + 3.0 * score_b + 1.5 * score_c + 3.0 * score_d) / 9.0 - psi



class Embeddings(nn.Module):
    def __init__(self, sampler, config):
        super(Embeddings, self).__init__()  # Calling Super Class's constructor
        self.config = config
        self.entity_embedding = nn.Embedding(sampler.entity2id.shape[0], config.x_feature_dimension)
        self.entity_embedding1 = nn.Embedding(sampler.entity2id.shape[0], config.x_feature_dimension)
        self.entity_embedding2 = nn.Embedding(sampler.entity2id.shape[0], config.x_feature_dimension)
        self.entity_embedding3 = nn.Embedding(sampler.entity2id.shape[0], config.x_feature_dimension)
        self.entity_embedding4 = nn.Embedding(sampler.entity2id.shape[0], config.x_feature_dimension)
        self.entity_embedding5 = nn.Embedding(sampler.entity2id.shape[0], config.x_feature_dimension)
        self.entity_embedding6 = nn.Embedding(sampler.entity2id.shape[0], config.x_feature_dimension)
        self.entity_embedding7 = nn.Embedding(sampler.entity2id.shape[0], config.x_feature_dimension)

        self.relation_embedding = nn.Embedding(sampler.relation2id.shape[0], config.r_feature_dimension)
        self.relation_embedding1 = nn.Embedding(sampler.relation2id.shape[0], config.r_feature_dimension)
        self.relation_embedding2 = nn.Embedding(sampler.relation2id.shape[0], config.r_feature_dimension)
        self.relation_embedding3 = nn.Embedding(sampler.relation2id.shape[0], config.r_feature_dimension)
        self.relation_embedding4 = nn.Embedding(sampler.relation2id.shape[0], config.r_feature_dimension)
        self.relation_embedding5 = nn.Embedding(sampler.relation2id.shape[0], config.r_feature_dimension)
        self.relation_embedding6 = nn.Embedding(sampler.relation2id.shape[0], config.r_feature_dimension)
        self.relation_embedding7 = nn.Embedding(sampler.relation2id.shape[0], config.r_feature_dimension)

        # for vector of elements

    def get_vectorised_values_entity(self, x):
        a0 = self.entity_embedding(torch.LongTensor(x))
        a1 = self.entity_embedding1(torch.LongTensor(x))
        a2 = self.entity_embedding2(torch.LongTensor(x))
        a3 = self.entity_embedding3(torch.LongTensor(x))
        a4 = self.entity_embedding4(torch.LongTensor(x))
        a5 = self.entity_embedding5(torch.LongTensor(x))
        a6 = self.entity_embedding6(torch.LongTensor(x))
        a7 = self.entity_embedding7(torch.LongTensor(x))

        a = torch.stack((a0, a1, a2, a3, a4, a5, a6, a7), dim=0)#
        return a

    def get_vectorised_values_relation(self, x):
        a0 = self.relation_embedding(torch.LongTensor(x))
        a1 = self.relation_embedding1(torch.LongTensor(x))
        a2 = self.relation_embedding2(torch.LongTensor(x))
        a3 = self.relation_embedding3(torch.LongTensor(x))
        a4 = self.relation_embedding4(torch.LongTensor(x))
        a5 = self.relation_embedding5(torch.LongTensor(x))
        a6 = self.relation_embedding6(torch.LongTensor(x))
        a7 = self.relation_embedding7(torch.LongTensor(x))
        a = torch.stack((a0, a1, a2, a3, a4, a5, a6, a7), dim=0) #
        return a

    def get_vectorised_value_relation(self, x):
        a0 = self.relation_embedding(x)
        a1 = self.relation_embedding1(x)
        a2 = self.relation_embedding2(x)
        a3 = self.relation_embedding3(x)
        a4 = self.relation_embedding4(x)
        a5 = self.relation_embedding5(x)
        a6 = self.relation_embedding6(x)
        a7 = self.relation_embedding7(x)
        a = torch.stack((a0, a1, a2, a3, a4, a5, a6, a7), dim=0)#, a4, a5, a6, a7
        return a

    def get_vectorised_value_entity(self, x):
        a0 = self.entity_embedding(x)
        a1 = self.entity_embedding1(x)
        a2 = self.entity_embedding2(x)
        a3 = self.entity_embedding3(x)
        a4 = self.entity_embedding4(x)
        a5 = self.entity_embedding5(x)
        a6 = self.entity_embedding6(x)
        a7 = self.entity_embedding7(x)
        a = torch.stack((a0, a1, a2, a3, a4, a5, a6, a7), dim=0)#, a4, a5, a6, a7
        return a


class Experiment(object):
    def __init__(self, dataset_setting):
        self.dataset_setting = dataset_setting
        self.sampler = SampleGenerator(self.dataset_setting)
        self.config = HyperParameters(self.dataset_setting, self.sampler)
        self.embeddings = Embeddings(self.sampler, self.config)
        self.model = MDE_Model(self.config,self.sampler, self.embeddings)  # .double()
        self.update_gamma_for_loss_function = False
        self.name = "MDE"
        self.sampler.make_all_samples_dic()
        self.mean_rank = 0
        self.hit_ten_tail = 0
        self.hit_one_tail = 0
        self.hit_three_tail = 0
        self.hit_ten_head = 0
        self.hit_one_head = 0
        self.hit_three_head = 0
        self.hit_hundred_tail = 0
        self.hit_hundred_head = 0
        print (self.name)

    def train(self):

        self.model.init_loss_parameters(self.config.gamma_1, self.config.gamma_2, self.config.beta1, self.config.beta2)
        for epoch in range(0, self.config.epochs):
            sum_loss = 0
            sum_loss_p = 0
            sum_loss_n = 0
            if self.config.batch_type == "pre_splitted_batch":
                #self.sampler.reshuffle()
                self.model.batch_counter = 0

            for batch_counter in range(0, self.config.number_of_batch):
                optimizer = torch.optim.Adadelta(self.model.parameters(), lr= self.config.learning_rate, weight_decay=1e-6)

                optimizer.zero_grad()
                loss, loss_p, loss_n = self.model()
                loss.backward()  # back props
                optimizer.step()  # update the parameters
                sum_loss = sum_loss + loss.item()
                sum_loss_p = sum_loss_p + loss_p.item()
                sum_loss_n = sum_loss_n + loss_n.item()

            loss_per_triple = sum_loss / (self.sampler.training_sample.shape[0])
            self.model.loss_p_per_triple = sum_loss_p / (self.sampler.training_sample.shape[0])
            self.model.loss_n_per_triple = sum_loss_n / (self.sampler.training_sample.shape[0])

            if self.update_gamma_for_loss_function:
                self.model.update_limits()
            print('epoch {}, loss_per_triple {}, loss_p_per_triple {} , loss_n_per_triple {}'.format(epoch, loss_per_triple, self.model.loss_p_per_triple ,self.model.loss_n_per_triple ))
            #self.test()
            if epoch > 250 and epoch % 50 == 0:
                #self.save(epoch)
                self.test()
                #self.save_state(epoch)

    def save(self, epoch):
        torch.save(self.model.embeddings, self.config.result_dir +"/MDE" + str(epoch)+ self.dataset_setting.dataset + self.name)
        return

    def load(self,epoch):
        self.model.embeddings = torch.load(self.config.result_dir +"/MDE" + str(epoch)+ self.dataset_setting.dataset + self.name)

    def sample_exists(self, triple_1,triple_2,triple_3):
        return np.array([self.sampler.all_samples.get((triple_1,triple_2,triple_3), False)])

    def reset_test_values_per_epoch(self):
        self.mean_rank = 0
        self.hit_ten_tail = 0
        self.hit_one_tail = 0
        self.hit_three_tail = 0
        self.hit_ten_head = 0
        self.hit_one_head = 0
        self.hit_three_head = 0
        self.mean_rank_filtered = 0
        self.hit_ten_tail_filtered = 0
        self.hit_one_tail_filtered  = 0
        self.hit_three_tail_filtered  = 0
        self.hit_ten_head_filtered  = 0
        self.hit_one_head_filtered  = 0
        self.hit_three_head_filtered  = 0

    def test(self):
        test_batch_size = int(self.sampler.test_samples.shape[0] / 8)
        self.sum_hit_1 = 0
        self.sum_hit_1_filtered = 0
        self.sum_hit_3 = 0
        self.sum_hit_3_filtered = 0
        self.sum_hit_10 = 0
        self.sum_hit_10_filtered = 0
        self.mean_rank_sum = 0
        self.mrr_rank_sum = 0
        self.mean_rank_filtered_sum = 0
        self.mrr_rank_filtered_sum = 0
        self.hit_ten_filtered =0
        self.hit_ten = 0
        self.hit_one = 0
        self.hit_one_filtered = 0
        self.hit_three = 0
        self.hit_three_filtered = 0
        self.mean_rank_filtered = 0
        a = 0
        b = a + test_batch_size
        c = b + test_batch_size
        d = c + test_batch_size
        e = d + test_batch_size
        f = e + test_batch_size
        g = f + test_batch_size
        h = g + test_batch_size
        i = h + test_batch_size
        #print a,b,c,d,e,f

        q = Queue()
        processes = []
        p = Process(target=self.test_one_batch, args=(a,b, q))
        p.start()
        processes.append(p)
        p = Process(target=self.test_one_batch, args=(b,c,q))
        p.start()
        processes.append(p)
        p = Process(target=self.test_one_batch, args=(c,d,q))
        p.start()
        processes.append(p)
        p = Process(target=self.test_one_batch, args=(d,e,q))
        p.start()
        processes.append(p)
        p = Process(target=self.test_one_batch, args=(e,f,q))
        p.start()
        processes.append(p)
        p = Process(target=self.test_one_batch, args=(f, g, q))
        p.start()
        processes.append(p)
        p = Process(target=self.test_one_batch, args=(g, h, q))
        p.start()
        processes.append(p)
        p = Process(target=self.test_one_batch, args=(h, i, q))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()

        output_array1 = q.get()
        output_array2 = q.get()
        output_array3 = q.get()
        output_array4 = q.get()
        output_array5 = q.get()
        output_array6 = q.get()
        output_array7 = q.get()
        output_array8 = q.get()

        self.sum_hit_10 = output_array1[0] + output_array2[0]+  output_array3[0] + output_array4[0] + output_array5[0]+ output_array6[0] + output_array7[0] + output_array8[0]
        self.mean_rank_sum = output_array1[1] + output_array2[1]+  output_array3[1] + output_array4[1] + output_array5[1]+  output_array6[1] + output_array7[1] + output_array8[1]
        self.sum_hit_10_filtered = output_array1[2] + output_array2[2]+  output_array3[2] + output_array4[2] + output_array5[2]+  output_array6[2] + output_array7[2] + output_array8[2]
        self.mean_rank_filtered_sum = output_array1[3] + output_array2[3]+  output_array3[3] + output_array4[3] + output_array5[3]+  output_array6[3] + output_array7[3] + output_array8[3]
        self.mrr_rank_sum = output_array1[4] + output_array2[4]+  output_array3[4] + output_array4[4] + output_array5[4]+  output_array6[4] + output_array7[4] + output_array8[4]
        self.mrr_rank_filtered_sum = output_array1[5] + output_array2[5]+  output_array3[5] + output_array4[5] + output_array5[5]+  output_array6[5] + output_array7[5] + output_array8[5]
        self.sum_hit_1 = output_array1[6] + output_array2[6]+  output_array3[6] + output_array4[6] + output_array5[6]+  output_array6[6] + output_array7[6] + output_array8[6]
        self.sum_hit_1_filtered = output_array1[7] + output_array2[7]+  output_array3[7] + output_array4[7] + output_array5[7]+  output_array6[7] + output_array7[7] + output_array8[7]
        self.sum_hit_3 = output_array1[8] + output_array2[8]+  output_array3[8] + output_array4[8] + output_array5[8]+  output_array6[8] + output_array7[8] + output_array8[8]
        self.sum_hit_3_filtered = output_array1[9] + output_array2[9]+  output_array3[9] + output_array4[9] + output_array5[9]+  output_array6[9] + output_array7[9] + output_array8[9]

        self.hit_one = self.sum_hit_1 / (self.sampler.test_samples.shape[0] * 2)
        self.hit_three = self.sum_hit_3 / (self.sampler.test_samples.shape[0] * 2)
        self.hit_ten = self.sum_hit_10 / (self.sampler.test_samples.shape[0] * 2)
        self.mean_rank = self.mean_rank_sum / (self.sampler.test_samples.shape[0]* 2)
        self.mrr_rank = self.mrr_rank_sum / (self.sampler.test_samples.shape[0]* 2)

        self.hit_one_filtered = self.sum_hit_1_filtered / (self.sampler.test_samples.shape[0] * 2)
        self.hit_three_filtered = self.sum_hit_3_filtered / (self.sampler.test_samples.shape[0] * 2)
        self.hit_10_filtered = self.sum_hit_10_filtered / (self.sampler.test_samples.shape[0] * 2)
        self.mean_rank_filtered = self.mean_rank_filtered_sum / (self.sampler.test_samples.shape[0]* 2)
        self.mrr_rank_filtered = self.mrr_rank_filtered_sum / (self.sampler.test_samples.shape[0]* 2)
        print ("hit at 1,3, 10, and mean rank and mrr:")
        print (self.hit_one)
        print (self.hit_three)
        print (self.hit_ten)
        print (self.mean_rank)
        print (self.mrr_rank)

        print ("hit at 1,3, 10, and mean rank and mrr: _filtered:")
        print (self.hit_one_filtered)
        print (self.hit_three_filtered)
        print (self.hit_10_filtered)
        print (self.mean_rank_filtered)
        print (self.mrr_rank_filtered)

    def test_one_batch(self, start_index, end_index, q):
        self.mean_rank = 0
        self.hit_ten_tail = 0
        self.hit_one_tail = 0
        self.hit_three_tail = 0
        self.hit_ten_head = 0
        self.hit_one_head = 0
        self.hit_three_head = 0
        self.mean_rank_filtered = 0
        self.hit_ten_tail_filtered = 0
        self.hit_one_tail_filtered  = 0
        self.hit_three_tail_filtered  = 0
        self.hit_ten_head_filtered  = 0
        self.hit_one_head_filtered  = 0
        self.hit_three_head_filtered  = 0

        test_triple_exists = np.vectorize(self.sample_exists, signature='(),(),()->(n)')

        for triple in self.sampler.test_samples[start_index:end_index,:]:  # testing with validation set .test_samples:
            # print triple
            R = self.model.embeddings.get_vectorised_values_relation([triple[2]]).unsqueeze(0)[0]

            score_test = self.model.predict(self.model.embeddings.get_vectorised_values_entity([triple[0]]),
                                                       self.model.embeddings.get_vectorised_values_entity([triple[
                                                                                                             1]]),
                                                       R).detach().numpy()
            score_test = score_test[0]
            reproduce_head = matlib.repmat(triple, self.model.sampler.entity2id.shape[0], 1)
            reproduce_head[:, 0] = self.model.sampler.entity2id  # [:, 1]
            reproduce_tail = matlib.repmat(triple, self.model.sampler.entity2id.shape[0], 1)
            reproduce_tail[:, 1] = self.model.sampler.entity2id  # [:, 1]

            score_test_head = self.model.predict(
                self.model.embeddings.get_vectorised_values_entity(reproduce_head[:, 0]),
                self.model.embeddings.get_vectorised_values_entity(reproduce_head[:, 1]),
                self.model.embeddings.get_vectorised_values_relation(
                    reproduce_head[:, 2])).detach().numpy()
            score_test_tail = self.model.predict(
                self.model.embeddings.get_vectorised_values_entity(reproduce_tail[:, 0]),
                self.model.embeddings.get_vectorised_values_entity(reproduce_tail[:, 1]),
                self.model.embeddings.get_vectorised_values_relation(
                    reproduce_tail[:, 2])).detach().numpy()
            scored_reproduce_head = np.hstack((score_test_head[:, None], reproduce_head))
            head_triple_exists = test_triple_exists(reproduce_head[:, 0], reproduce_head[:, 1], reproduce_head[:, 2])
            scored_reproduce_head = np.hstack((scored_reproduce_head, head_triple_exists))
            scored_reproduce_head_filtered = scored_reproduce_head[scored_reproduce_head[:, 4] == 0]
            scored_reproduce_head_filtered = np.row_stack(
                (scored_reproduce_head_filtered, [score_test, triple[0], triple[1], triple[2], 1]))
            scored_reproduce_head = scored_reproduce_head[np.argsort(scored_reproduce_head[:, 0])]
            scored_reproduce_head_filtered = scored_reproduce_head_filtered[
                np.argsort(scored_reproduce_head_filtered[:, 0])]

            scored_reproduce_tail = np.hstack((score_test_tail[:, None], reproduce_tail))
            tail_triple_exists = test_triple_exists(reproduce_tail[:, 0], reproduce_tail[:, 1], reproduce_tail[:, 2])
            scored_reproduce_tail = np.hstack((scored_reproduce_tail, tail_triple_exists))
            scored_reproduce_tail_filtered = scored_reproduce_tail[scored_reproduce_tail[:, 4] == 0]
            scored_reproduce_tail_filtered = np.row_stack(
                (scored_reproduce_tail_filtered, [score_test, triple[0], triple[1], triple[2], 1]))

            scored_reproduce_tail_filtered = scored_reproduce_tail_filtered[
                np.argsort(scored_reproduce_tail_filtered[:, 0])]
            scored_reproduce_tail = scored_reproduce_tail[np.argsort(scored_reproduce_tail[:, 0])]
            try:
                hit_head_filtered = np.amin(np.where(scored_reproduce_head_filtered[:, 0] == score_test)[0])+1
                hit_tail_filtered = np.amin(np.where(scored_reproduce_tail_filtered[:, 0] == score_test)[0])+1
                hit_head = np.amin(np.where(scored_reproduce_head[:, 0] == score_test)[0])+1
                hit_tail = np.amin(np.where(scored_reproduce_tail[:, 0] == score_test)[0])+1
                # print hit_head
                # print hit_tail
                if hit_tail < 11:
                    self.hit_ten_tail = self.hit_ten_tail + 1
                if hit_head < 11:
                    self.hit_ten_head = self.hit_ten_head + 1
                if hit_tail < 2:
                    self.hit_one_tail = self.hit_one_tail + 1
                if hit_head < 2:
                    self.hit_one_head = self.hit_one_head + 1
                if hit_tail < 4:
                    self.hit_three_tail = self.hit_three_tail + 1
                if hit_head < 4:
                    self.hit_three_head = self.hit_three_head + 1
                self.mean_rank_sum = self.mean_rank_sum + (hit_head + hit_tail)
                if hit_head != 0:
                    self.mrr_rank_sum = self.mrr_rank_sum + (1.0/hit_head )
                if hit_tail != 0:
                    self.mrr_rank_sum = self.mrr_rank_sum + (1.0/hit_tail)

                if hit_tail_filtered < 11:
                    self.hit_ten_tail_filtered = self.hit_ten_tail_filtered + 1
                if hit_head_filtered < 11:
                    self.hit_ten_head_filtered = self.hit_ten_head_filtered + 1
                if hit_tail_filtered < 2:
                    self.hit_one_tail_filtered = self.hit_one_tail_filtered + 1
                if hit_head_filtered < 2:
                    self.hit_one_head_filtered = self.hit_one_head_filtered + 1
                if hit_tail_filtered < 4:
                    self.hit_three_tail_filtered = self.hit_three_tail_filtered + 1
                if hit_head_filtered < 4:
                    self.hit_three_head_filtered = self.hit_three_head_filtered + 1
                self.mean_rank_filtered_sum = self.mean_rank_filtered_sum + (hit_head_filtered + hit_tail_filtered)
                if hit_head_filtered != 0:
                    self.mrr_rank_filtered_sum = self.mrr_rank_filtered_sum + (1.0/hit_head_filtered)
                if hit_tail_filtered != 0:
                    self.mrr_rank_filtered_sum = self.mrr_rank_filtered_sum + (1.0 /hit_tail_filtered)

            except ValueError:  # raised if `score_test_head` is empty.
                print ("there was error in test")
                pass
        self.sum_hit_1 = self.sum_hit_1 + self.hit_one_tail + self.hit_one_head
        self.sum_hit_1_filtered = self.sum_hit_1_filtered + self.hit_one_head_filtered + self.hit_one_tail_filtered
        self.sum_hit_3 = self.sum_hit_3 + self.hit_three_tail + self.hit_three_head
        self.sum_hit_3_filtered = self.sum_hit_3_filtered + self.hit_three_tail_filtered + self.hit_three_head_filtered
        self.sum_hit_10 = self.sum_hit_10 + self.hit_ten_tail + self.hit_ten_head
        self.sum_hit_10_filtered = self.sum_hit_10_filtered + self.hit_ten_head_filtered +   self.hit_ten_tail_filtered

        q.put([self.sum_hit_10,self.mean_rank_sum,self.sum_hit_10_filtered, self.mean_rank_filtered_sum,self.mrr_rank_sum,self.mrr_rank_filtered_sum, self.sum_hit_1,self.sum_hit_1_filtered,self.sum_hit_3,self.sum_hit_3_filtered])
        return

    def save_state(self, epoch):
        out_file = self.sampler.dbPath + self.dataset_setting.dataset + "_T2_result.csv"
        out_array = [["epoch", str(epoch)]]
        #out_array.append(["hit_1", str((self.hit_one_tail + self.hit_one_head) / (self.sampler.test_samples.shape[0] * 2))])
        out_array.append(["hit_10", str((self.hit_ten) / (self.sampler.test_samples.shape[0] * 2))])
        out_array.append(["mean_rank", str((self.mean_rank) / (self.sampler.test_samples.shape[0] * 2))])
        #out_array.append(["hit_1_filtered", str((self.hit_one_tail_filtered + self.hit_one_head_filtered) / (self.sampler.test_samples.shape[0] * 2))])
        out_array.append(["hit_10_filtered", str((self.hit_ten_filtered) / (self.sampler.test_samples.shape[0] * 2))])
        out_array.append(["mean_rank_filtered", str((self.mean_rank_filtered) / (self.sampler.test_samples.shape[0] * 2))])
        out_array_np = np.asarray(out_array)
        with open(out_file, 'a') as f:
            csv.writer(f).writerows(out_array_np)
        return


class DatasetSetting(object):
    def __init__(self):
        self.dataset = ""

    def set_dataset(self,dataset_name):
        self.dataset = dataset_name


class HyperParameters(object):
    def __init__(self, dataset_setting, sampler):
        self.dataset_setting = dataset_setting

        self.entity = 0
        self.relation = 0
        self.epochs = 3600
        self.learning_rate = 10.0 #0.01
        if self.dataset_setting.dataset == "FB15":
            self.x_feature_dimension = 200#50#10
            self.margin = 1.0
            self.L1_Norm = False #L2
            self.number_of_batch = 280#500#460
            self.gamma_1 = 10
            self.gamma_2 = 13
            self.beta1 = 1
            self.beta2 = 1

        elif self.dataset_setting.dataset == "FB15K237":
            self.x_feature_dimension = 100#50#10
            self.margin = 1.0
            self.L1_Norm = False #L2
            self.number_of_batch = 1000#230#200#230  #with 1000 batch sizes becomes 272
            self.gamma_1 = 9
            self.gamma_2 = 9
            self.beta1 = 1
            self.beta2 = 1

        elif  self.dataset_setting.dataset == "WN18":
            self.x_feature_dimension = 50
            self.margin = 1.0
            self.gamma_1 = 1.9
            self.gamma_2 = 1.9
            self.beta1 = 1
            self.beta2 = 2
            self.L1_Norm = False
            self.number_of_batch = 100

        elif  self.dataset_setting.dataset == "WN18RR":
            self.x_feature_dimension = 50#20
            self.margin = 1.0
            self.L1_Norm = False
            self.number_of_batch = 50 #batch size becomes: 1736
            self.gamma_1 = 2#15#1  # 15#2#9#8#7#4#1.4#2
            self.gamma_2 = 2#15#1  # 15#2#9#8#7#4#1.4#2
            self.beta1 = 1
            self.beta2 = 5  # 1.5#5
        self.r_feature_dimension = self.x_feature_dimension
        self.x_train_bach_size = int(sampler.training_sample.shape[0] / self.number_of_batch)
        self.result_dir = "tmp"
        self.batch_type = "pre_splitted_batch"  #pre_splitted_batch  random_batch
        print (self.dataset_setting.dataset, self.x_train_bach_size,self.gamma_1, self.gamma_2, self.beta1, self.beta2, self.x_feature_dimension, self.learning_rate)

        # relation_number = 8  # for wn


def find_gamma_experiment(dataset_name):
    dataset_setting = DatasetSetting()
    dataset_setting.set_dataset(dataset_name)
    experiment = Experiment(dataset_setting)
    experiment.update_gamma_for_loss_function = True
    experiment.train()


def train_experiment(dataset_name):
    dataset_setting = DatasetSetting()
    dataset_setting.set_dataset(dataset_name)
    experiment = Experiment(dataset_setting)
    experiment.train()
    experiment.save(experiment.config.epochs)# so that to store the model last epoch as well.
    experiment.test()


def test_experiment(dataset_name):
    dataset_setting = DatasetSetting()
    dataset_setting.set_dataset(dataset_name)
    experiment = Experiment(dataset_setting)
    experiment.load(epoch = 2500)
    experiment.test()

#find_gamma_experiment("WN18RR")
#train_experiment("WN18RR")  #MDE_Model_8v.py"WN18RR"# "FB15K237"#"WN18RR"# "FB15"#"FB15"  # "WN18"
#test_experiment("WN18RR")


if args.task == "train":
    train_experiment(args.dataset)

elif args.task == "find_g":
    find_gamma_experiment(args.dataset)

elif args.task == "test":
    train_experiment(args.dataset)

else:
    print ("arguments are -t for task that can be 'train' or 'test' or 'find_g' and -d with dataset name which can be WN18RR FB15K237 FB15 WN18")

#testing batch size 272 for fb15k237