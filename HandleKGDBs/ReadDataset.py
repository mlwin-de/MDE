import numpy as np
import os
class readDataset():

    def __init__(self):
        home_dir = '.' #os.getenv("HOME")
        self.input_directory = home_dir + "/EmbedDBs/"
        return

    def read_WN18(self, dataset_dir_ = ""):
        if dataset_dir_ == "":
            self.dataset_dir = self.input_directory + "WN18"  # "WN18"   I reprlace the folder and filenames from WN18 folder to WN18-TransX
        else:
            self.dataset_dir = dataset_dir_

        entity2id_file = self.dataset_dir + "/entity2id.txt"
        relation2id_file = self.dataset_dir + "/relation2id.txt"
        # train_data_file = self.dataset_dir + "/train.txt"
        train_data_file = self.dataset_dir + "/train2id.txt"#"/triple2id.txt" #"/train2id.txt"
        test_data_file = self.dataset_dir + "/test2id.txt"
        vaidation_data_file = self.dataset_dir + "/valid2id.txt"
        
        self.entity2id_wn = np.loadtxt(open(entity2id_file, "rb"), delimiter="\t",usecols=(1), skiprows=1, dtype="int") #usecols=range(0, 1))
        self.relation2id_wn = np.loadtxt(open(relation2id_file, "rb"), delimiter="\t", usecols=(1),skiprows=1, dtype="int")#,    usecols=range(0, 100))
        self.train_data_wn = np.loadtxt(open(train_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int" )#,   usecols=range(0, 100))
        self.test_data_wn = np.loadtxt(open(test_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int" )#,   usecols=range(0, 100))

        self.validation_data_wn = np.loadtxt(open(vaidation_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int" )

    def read_FB15K(self, dataset_dir_ = ""):
        if dataset_dir_ == "":
            self.dataset_dir = self.input_directory + "FB15K"  # "FB15K" is the FB15K made by bordes
        else:
            self.dataset_dir = dataset_dir_

        entity2id_file =  self.dataset_dir + "/entity2id.txt"
        relation2id_file =  self.dataset_dir + "/relation2id.txt"
        train_data_file = self.dataset_dir + "/train2id.txt"
        validation_data= self.dataset_dir + "/valid2id.txt"
        test_data_file = self.dataset_dir + "/test2id.txt"
        self.entity2id_fb = np.loadtxt(open(entity2id_file, "rb"), delimiter="\t", usecols=(1),skiprows=1, dtype="int")
        self.relation2id_fb = np.loadtxt(open(relation2id_file, "rb"), delimiter="\t", usecols=(1),skiprows=1, dtype="int")
        self.train_data_fb = np.loadtxt(open(train_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int")
        self.validation_data_fb = np.loadtxt(open(validation_data, "rb"), delimiter=" ", skiprows=1, dtype="int")
        self.test_data_fb = np.loadtxt(open(test_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int")

    def read_WN18RR(self, dataset_dir_=""):
        if dataset_dir_ == "":
            self.dataset_dir = self.input_directory + "WN18RR"
        else:
            self.dataset_dir = dataset_dir_
        entity2id_file = self.dataset_dir + "/entity2id.txt"
        relation2id_file = self.dataset_dir + "/relation2id.txt"
        train_data_file = self.dataset_dir + "/train2id.txt"
        test_data_file = self.dataset_dir + "/test2id.txt"
        vaidation_data_file = self.dataset_dir + "/valid2id.txt"
        self.entity2id_wn = np.loadtxt(open(entity2id_file, "rb"), delimiter="\t", usecols=(1), skiprows=1,
                                     dtype="int")
        self.relation2id_wn = np.loadtxt(open(relation2id_file, "rb"), delimiter="\t", usecols=(1), skiprows=1,
                                       dtype="int")
        self.train_data_wn = np.loadtxt(open(train_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int")
        self.test_data_wn = np.loadtxt(open(test_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int")
        self.validation_data_wn = np.loadtxt(open(vaidation_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int")

    def read_countries_s1(self, dataset_dir_=""):
        if dataset_dir_ == "":
            self.dataset_dir = self.input_directory + "countries_S1"
        else:
            self.dataset_dir = self.input_directory+ dataset_dir_
        entity2id_ = self.dataset_dir + "/entities.dict"
        relation2id_ = self.dataset_dir + "/relations.dict"
        region_to_id = self.dataset_dir + "/regions.list"
        train_data_file = self.dataset_dir + "/train.txt"
        test_data_file = self.dataset_dir + "/test.txt"
        vaidation_data_file = self.dataset_dir + "/valid.txt"
        entity2id_ = np.loadtxt(open(entity2id_, "rb"), delimiter="\t", skiprows=0,dtype="str")

        relation2id_ = np.loadtxt(open(relation2id_, "rb"), delimiter="\t", skiprows=0,dtype="str")

        train_data = np.loadtxt(open(train_data_file, "rb"), delimiter="\t", skiprows=0, dtype="str")
        test_data = np.loadtxt(open(test_data_file, "rb"), delimiter="\t", skiprows=0, dtype="str")
        validation_data = np.loadtxt(open(vaidation_data_file, "rb"), delimiter="\t", skiprows=0, dtype="str")

        self.entity2id_ = np.zeros(entity2id_.shape[0], dtype='int')
        self.relation2id_ = np.zeros(relation2id_.shape[0], dtype='int')
        entity_dic = {}
        relation_dic = {}
        for entity_line in entity2id_:
            entity_dic[entity_line[1]] = int(entity_line[0])
            self.entity2id_[int (entity_line[0])] = int (entity_line[0])
        for rel_line in relation2id_:
            relation_dic[rel_line[1]] = int(rel_line[0])
            self.relation2id_[int(rel_line[0])] = int(rel_line[0])

        self.train_data_ = np.zeros(train_data.shape, dtype='int')
        self.test_data_ = np.zeros(test_data.shape, dtype='int')
        self.validation_data_ = np.zeros(validation_data.shape)
        for t_num in range(0,self.train_data_.shape[0]):
            self.train_data_[t_num,0] = entity_dic[ train_data[t_num,0]]
            self.train_data_[t_num, 1] = entity_dic[ train_data[t_num, 2]]
            self.train_data_[t_num, 2] = relation_dic[ train_data[t_num, 1]]
        for t_num in range(0,self.test_data_.shape[0]):
            self.test_data_[t_num, 0] = entity_dic[ test_data[t_num, 0]]
            self.test_data_[t_num, 1] = entity_dic[ test_data[t_num, 2]]
            self.test_data_[t_num, 2] = relation_dic[ test_data[t_num, 1]]
        for t_num in range(0,self.validation_data_.shape[0]):
            self.validation_data_[t_num, 0] = entity_dic[ validation_data[t_num, 0]]
            self.validation_data_[t_num, 1] = entity_dic[ validation_data[t_num, 2]]
            self.validation_data_[t_num, 2] = relation_dic[ validation_data[t_num, 1]]

        regions = list()
        with open(region_to_id) as file_c:
            for line in file_c:
                region = line.strip()
                regions.append(entity_dic[region])
        self.regions = regions

    def read_FB15K237(self, dataset_dir_=""):
        if dataset_dir_ == "":
            self.dataset_dir = self.input_directory + "FB15K237"
        else:
            self.dataset_dir = dataset_dir_
            self.dataset_dir = dataset_dir_
        entity2id_file = self.dataset_dir + "/entity2id.txt"
        relation2id_file = self.dataset_dir + "/relation2id.txt"
        train_data_file = self.dataset_dir + "/train2id.txt" 
        validation_data = self.dataset_dir + "/valid2id.txt"
        test_data_file = self.dataset_dir + "/test2id.txt"
        self.entity2id_fb = np.loadtxt(open(entity2id_file, "rb"), delimiter="\t", usecols=(1), skiprows=1,
                                     dtype="int")
        self.relation2id_fb = np.loadtxt(open(relation2id_file, "rb"), delimiter="\t", usecols=(1), skiprows=1,
                                       dtype="int")
        self.entity_and_id_ = np.loadtxt(open(entity2id_file, "rb"), delimiter="\t", usecols=(0, 1), skiprows=1,
                                         dtype="str")
        self.relation_and_id_ = np.loadtxt(open(relation2id_file, "rb"), delimiter="\t", usecols=(0, 1), skiprows=1,
                                           dtype="str")

        self.train_data_fb = np.loadtxt(open(train_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int")
        self.validation_data_fb = np.loadtxt(open(validation_data, "rb"), delimiter=" ", skiprows=1, dtype="int")
        self.test_data_fb = np.loadtxt(open(test_data_file, "rb"), delimiter=" ", skiprows=1, dtype="int")

    def create_hash_tables_of_entities(relationToId, entityToId, training_triplesInRaw):

        training_triples = np.zeros(training_triplesInRaw.shape)

        relationToId_Hash = {}
        entityToId_Hash = {}
        for i in range(0, relationToId.shape[0]):
            relationToId_Hash[relationToId[i, 0]] = int(relationToId[i, 1])

        for i in range(0, entityToId.shape[0]):
            entityToId_Hash[entityToId[i, 0]] = int(entityToId[i, 1])

        for i in range(0, training_triplesInRaw.shape[0]):
            training_triples[i, 0] = entityToId_Hash[training_triplesInRaw[i, 0]]
            training_triples[i, 1] = entityToId_Hash[training_triplesInRaw[i, 1]]
            training_triples[i, 2] = relationToId_Hash[training_triplesInRaw[i, 2]]

        Y = {}
        for i in range(0, training_triples.shape[0]):
            en1 = int(training_triples[i, 0])
            en2 = int(training_triples[i, 1])
            a = Y.get((en1, en2), None)
            if a is None:
                a = np.zeros((relationToId.shape[0]))  # 18
            a[int(training_triples[i, 2])] = 1
            Y[en1, en2] = a
        return Y
