# -*- coding = utf-8 -*-
'''
COMP9417 2019S2 project: recommendation system
Writen @ Laser, JacK, Aedgen and Alex
python 3 
'''

'''
usage:  python3 CF.py model_number recommend_uer_number recommend_movies

Model_type:
1. user_based
2. user_based_iif
3. item_based
4. item_based_iuf
5.  most_popular
6.  random

Gobal variables:
K: similarity item(user) value eg. k=10 we consider 10 similiar user
N: the number of movie recommend to a user
Ratio: the ratio of training set and testing set if ratio=0.3
       which means training_set=0.7*total_data testing_set=0.3*total_data
random_seed: the seed for data split and random recommendation

Dataset
1. Data_rating_1m 
2. Data_rating_100k


'''


import os
import math
import sys
import random
import pickle
import shutil
import itertools
import collections
from collections import namedtuple
from collections import defaultdict
from operator import itemgetter



K=20
N=20
Ratio=0.2
model_type=1
Dataset=2
random_seed=8888
recommend_user_number=20
random.seed(random_seed)

class DataSet:
  
    def __init__(self):
        random.seed(random_seed)
    @classmethod
    def load_dataset( ClearSystem, name='ml-100k'):
        create_datasets={}
        BuiltinDataset = namedtuple('BuiltinDataset', ['path', 'sep', 'params'])
        create_datasets = {
            'ml-100k':
                BuiltinDataset(
                path='data/ml-100k/u.data',
                sep='\t',
                params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='\t')
            ),
            'ml-1m'  :
                BuiltinDataset(
                path='data/ml-1m/ratings.dat',
                sep='::',
                params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='::')
            ),
        }
        
       
        try:
            dataset = create_datasets[name]
        except KeyError:
            raise ValueError('unknown dataset ' + name )
        if not os.path.isfile(dataset.path):
            raise OSError(
                "Dataset data/" + name + " could not be found \n")
        with open(dataset.path) as f:
            ratings = [ClearSystem.parse_line(line, dataset.sep) for line in itertools.islice(f, 0, None)]
        print("Load " + name + " dataset success.")
        return ratings

    @classmethod
    def parse_line(ClearSystem, line: str, sep: str):
        user, movie, rate = line.strip('\r\n').split(sep)[:3]
        return user, movie, rate

    @classmethod
    def train_test_split(ClearSystem, ratings, TestingSet_Size=0.2):
        train, test = collections.defaultdict(dict), collections.defaultdict(dict)
        trainingset_len = 0
        testingset_len = 0
        for user, movie, rate in ratings:
            if random.random() <= TestingSet_Size:
                test[user][movie] = int(rate)
                testingset_len += 1
            else:
                train[user][movie] = int(rate)
                trainingset_len += 1
        return train, test

class ModelManager:
    directory_name = ''
    @classmethod
    def __init__(ClearSystem, dataset_name='ml-1m', TestingSet_Size=0.3):
        if not ClearSystem.directory_name:
            ClearSystem.directory_name = 'model/' + dataset_name + '-testsize' + str(TestingSet_Size)

    @classmethod
    def save_model(ClearSystem, model, saved_name):
        if 'pkl' not in saved_name:
            saved_name += 'pkl'
        if not os.path.exists('model'):
            os.mkdir('model')
        pickle.dump(model, open(ClearSystem.directory_name + saved_name, "wb"))

    @classmethod
    def load_model(ClearSystem, model_name):
        if 'pkl' not in model_name:
            model_name += '.pkl'
        if not os.path.exists(ClearSystem.directory_name + model_name):
            raise OSError('wroing model named %s' % model_name)
        return pickle.load(open(ClearSystem.directory_name + model_name, "rb"))

    @staticmethod
    def clean_workspace(clean=False):
        if clean and os.path.exists('model'):
            shutil.rmtree('model')

class UserBasedCF:
    '''
    variables: K: the number of K similiar users
               N: the number of N moives recommend to a user
               use_iif: True-decrease the popular item weight, False-all items have the same weigh t            
    '''
  
    def __init__(self, K=20, N=10,use_iif=False):
       
        self.K = K
        self.N = N
        self.trainingset = None
        self.use_iif = use_iif

    def training_Model(self, trainingset):
        
        mManager = ModelManager()
        try:
            self.user_similarity_matrix = mManager.load_model(
                'user_similarity_matrix-iif' if self.use_iif else 'user_similarity_matrix')
            self.popular_movies = mManager.load_model('popular_movies')
            self.movie_numbers = mManager.load_model('movie_numbers')
            self.trainingset = mManager.load_model('trainingset')
         
        except OSError:
        
            self.user_similarity_matrix, self.popular_movies, self.movie_numbers = \
                compute_sim_users(trainingset=trainingset,
                                                     use_iif=self.use_iif)
            self.trainingset = trainingset
            mManager.save_model(self.user_similarity_matrix,
                                 'user_similarity_matrix-iif' if self.use_iif else 'user_similarity_matrix')
            mManager.save_model(self.popular_movies, 'popular_movies')
            mManager.save_model(self.movie_numbers, 'movie_numbers')
            
    def recommend(self, user):
        K = self.K
        N = self.N
        predict_score = collections.defaultdict(int)
        if user not in self.trainingset:
            print('The user (%s)does not exist in the training set.' % user)
            return
        watched_movies = self.trainingset[user]
        for SimilarUser, similarity_factor in sorted(self.user_similarity_matrix[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for movie, rating in self.trainingset[SimilarUser].items():
                if movie in watched_movies:
                    continue              
                predict_score[movie] += similarity_factor * rating                
        return [movie for movie, _ in sorted(predict_score.items(), key=itemgetter(1), reverse=True)[0:N]]

    def evaluation(self, testingset):
     
        self.testingset = testingset
        N = self.N
        hit = 0
        recommend_numbers = 0
        testing_numbers = 0
        total_recom_moives = set()
        popular_sum = 0
        for i, user in enumerate(self.trainingset):
            test_movies = self.testingset.get(user, {})
            rec_movies = self.recommend(user) 
            for movie in rec_movies:
                if movie in test_movies:
                    hit += 1
                total_recom_moives.add(movie)
                popular_sum += math.log(1 + self.popular_movies[movie])
            recommend_numbers += N
            testing_numbers += len(test_movies)
        precision = hit / (1.0 * recommend_numbers)
        recall = hit / (1.0 * testing_numbers)
        coverage = len(total_recom_moives) / (1.0 * self.movie_numbers)
        popularity = popular_sum / (1.0 * recommend_numbers)
        print("\n")
        print('-' * 108)
        print('\n\t\t\t\tCF SYSTEM PERFOMANCE RESULTS:\n\n\tprecision=%.4f  \trecall=%.4f    \tcoverage=%.4f    \tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))
       
        print('-' * 108)
        

class ItemBasedCF:
    def __init__(self, K=20, N=10, use_iuf=False):
       
        self.K = K
        self.N = N
        self.trainingset = None
        self.use_iuf = use_iuf

    '''
    If the itemBased model does not exist then train the model and save it, otherwise load the model
    '''
    def training_Model(self, trainingset):
        mManager = ModelManager()
        try:
            self.movie_sim_mat = mManager.load_model(
                'movie_sim_mat-iif' if self.use_iuf else 'movie_sim_mat')
            self.popular_movies = mManager.load_model('popular_movies')
            self.movie_numbers = mManager.load_model('movie_numbers')
            self.trainingset = mManager.load_model('trainingset')
            
        except OSError:
            #The model does not exist, train the model and save it
            self.movie_sim_mat, self.popular_movies, self.movie_numbers = \
                compute_sim_items(trainingset=trainingset,
                                                     use_iuf=self.use_iuf)
            self.trainingset = trainingset 
            mManager.save_model(self.movie_sim_mat,
                                     'movie_sim_mat-iif' if self.use_iuf else 'movie_sim_mat')
            mManager.save_model(self.popular_movies, 'popular_movies')
            mManager.save_model(self.movie_numbers, 'movie_numbers')
            mManager.save_model(self.trainingset, 'trainingset')
               

    def recommend(self, user):
       
        K = self.K
        N = self.N
        predict_score = collections.defaultdict(int)
        
        if user not in self.trainingset:
            print('The user (%s) not existed in our training set.' % user)
            return        
        watched_movies = self.trainingset[user]
        for movie, rating in watched_movies.items():
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                                                          key=itemgetter(1), reverse=True)[0:K]:
                if related_movie in watched_movies:
                    continue
                predict_score[related_movie] += similarity_factor * rating                    
        return [movie for movie, _ in sorted(predict_score.items(), key=itemgetter(1), reverse=True)[0:N]]

    def evaluation(self, testingset):               
        self.testingset = testingset
        N = self.N
        hit = 0
        recommend_numbers = 0
        testing_numbers = 0   
        total_recommend_numbers = set()      
        popular_sum = 0
        #print("***************************")
        for i, user in enumerate(self.trainingset):
            test_movies = self.testingset.get(user, {})
            rec_movies = self.recommend(user)  
            for movie in rec_movies:
                if movie in test_movies:
                    hit += 1
                total_recommend_numbers.add(movie)
                popular_sum += math.log(1 + self.popular_movies[movie])               
            recommend_numbers += N
            testing_numbers += len(test_movies)
        #print("---------------------------------------------")
        precision = hit / (1.0 * recommend_numbers)
        recall = hit / (1.0 * testing_numbers)
        coverage = len(total_recommend_numbers) / (1.0 * self.movie_numbers)
        popularity = popular_sum / (1.0 * recommend_numbers)
        
        print("\n")
        print('-' * 108)
        print('\n\t\t\t\tCF SYSTEM PERFOMANCE RESULTS:\n\n\tprecision=%.4f  \trecall=%.4f    \tcoverage=%.4f    \tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))
       
        print('-' * 108)
 
class PopularModel:   
    def __init__(self, N=10):        
        print("PopularModel start...\n")
        self.N = N
        self.trainingset = None        

    def training_Model(self, trainingset):
         #training_Model fuction is used to training_Model a model, if the model existed then load the model, otherwise train the mew model  
        mManager = ModelManager()
        try:
            #load the popular model if it exists, if it does not exist than train the new model
            self.popular_movies = mManager.load_model('popular_movies')
            self.movie_numbers = mManager.load_model('movie_numbers')
            self.trainingset = mManager.load_model('trainingset')
            self.total_movies = mManager.load_model('total_movies')
            self.popular_movies_sort = mManager.load_model('popular_movies_sort')
        except OSError:
            #the popular model does not exist,train the model and seve the model for further use
            self.trainingset = trainingset
            self.popular_movies, self.movie_numbers = calculate_popular_movies(trainingset)
            self.total_movies = list(self.popular_movies.keys())
            self.popular_movies_sort = sorted(self.popular_movies.items(), key=itemgetter(1), reverse=True)                  
            mManager.save_model(self.popular_movies, 'popular_movies')
            mManager.save_model(self.movie_numbers, 'movie_numbers')
            mManager.save_model(self.total_movies, 'total_movies')
            mManager.save_model(self.popular_movies_sort, 'popular_movies_sort')
                
    #most popular recommend mothod, apply to those use is not in the training set, so it recommeds N movies for the user
    def recommend(self, user):                
        N = self.N
        watched_movies = self.trainingset[user]
        recommend_list=list()
        for M,_ in self.popular_movies_sort:
            if len(recommend_list) <N and M  not in watched_movies:
                recommend_list.append(M)
        return recommend_list

    '''
    verify the popular mode, output the performance parameter
    precision:  =correct_recommend_numbers/total_recommend_numbers
    racall:     =correct_recommend_numbers/testing_numbers
    coverage:   =recommend_
    '''
    def evaluation(self, testingset):
        self.testingset = testingset
        N = self.N
        hit = 0
        testing_numbers = 0
        recommend_numbers = 0
        total_recommend_moives = set()
        popular_sum = 0
        for i, user in enumerate(self.trainingset):
            test_movies = self.testingset.get(user, {})
            rec_movies = self.recommend(user)  # type:list
            for movie in rec_movies:
                if movie in test_movies:
                    hit += 1
                total_recommend_moives.add(movie)
                popular_sum += math.log(1 + self.popular_movies[movie])
            recommend_numbers += N
            testing_numbers += len(test_movies)
        precision = hit / (1.0 * recommend_numbers)
        recall = hit / (1.0 * testing_numbers)
        coverage = len(total_recommend_moives) / (1.0 * self.movie_numbers)
        popularity = popular_sum / (1.0 * recommend_numbers)
        print("\n")
        print('-' * 108)
        print('\n\t\t\t\tCF SYSTEM PERFOMANCE RESULTS:\n\n\tprecision=%.4f  \trecall=%.4f    \tcoverage=%.4f    \tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))
       
        print('-' * 108)
class RandomModel:   
    def __init__(self, N=10, save_model=True):
        self.N = N
        self.trainingset = None
        self.save_model = save_model

    '''
    training_Model fuction is used to training_Model the random model, if the model existed then load the model, otherwise train the mew model  
    '''
    def training_Model(self, trainingset):
        mManager = ModelManager()
        try:
            self.popular_movies = mManager.load_model('popular_movies')
            self.movie_numbers = mManager.load_model('movie_numbers')
            self.trainingset = mManager.load_model('trainingset')
            self.total_movies = mManager.load_model('total_movies')            
        except OSError:
            self.trainingset = trainingset
            self.popular_movies, self.movie_numbers = calculate_popular_movies(trainingset)
            self.total_movies = list(self.popular_movies.keys())
            mManager.save_model(self.popular_movies, 'popular_movies')
            mManager.save_model(self.movie_numbers, 'movie_numbers')
            mManager.save_model(self.total_movies, 'total_movies')
                
    '''
    random recomend N moive to the user
    '''
    def recommend(self, user):                    
        recommend_movie = list()
        N = self.N
        watched_movies = self.trainingset[user]        
        while len(recommend_movie) < N:
            movie = random.choice(self.total_movies)
            if movie not in watched_movies:
                recommend_movie.append(movie)
        return recommend_movie[:N]

    '''
    evualate the random model, evaluation all testingset users
    '''
    def evaluation(self, testingset):
    
        if not self.N or not self.trainingset or not self.popular_movies or not self.movie_numbers:
            raise ValueError('random does not work.')
        self.testingset = testingset      
        N = self.N
        hit = 0
        recommend_numbers = 0
        all_rec_movies = set()
        testing_numbers = 0
        popular_sum = 0
        for i, user in enumerate(self.trainingset):
            test_movies = self.testingset.get(user, {})
            rec_movies = self.recommend(user) 
            for movie in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.popular_movies[movie])
            recommend_numbers += N
            testing_numbers += len(test_movies)           
        precision = hit / (1.0 * recommend_numbers)
        recall = hit / (1.0 * testing_numbers)
        coverage = len(all_rec_movies) / (1.0 * self.movie_numbers)
        popularity = popular_sum / (1.0 * recommend_numbers)

        print("\n")
        print('-' * 108)
        print('\n\t\t\t\tCF SYSTEM PERFOMANCE RESULTS:\n\n\tprecision=%.4f  \trecall=%.4f    \tcoverage=%.4f    \tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))
       
        print('-' * 108)
        
       
       
'''
Calculate the users similarity matrix
input: trainingset--train data (dict)
       use_iuf  --if use_iuf==True then give them a lower weight
output: similarity matrix       
'''

def compute_sim_users(trainingset, use_iif=True):
    
    #bulid the movie2users dict, the key is the moiviId, the value is the userId,record the movie who have watched it.
    #bulid the popular_movies dict,the key is the moiveId, the value is the number of users who have seen this moive
    movie2users = collections.defaultdict(set)
    popular_movies = defaultdict(int)
    #step 1: build movie2users tale
    for user, movies in trainingset.items():
        for movie in movies:
            popular_movies[movie] += 1
            movie2users[movie].add(user)
    Total_movie_num = len(movie2users)

    #caluate the similarity of users based on they dataset of movie2users
    #similarity algorithm fomular: W(u1,u2)=len(N(u1)^N(us))/ sqrt(len(N(u1))*len(N(u2)))
    #if we use the use_iuf then W(u1,u2)=sum((N(u1)^N(us)*1/log(1+N(i))/ sqrt(len(N(u1))*len(N(u2))) for i=0~n

    users_smilarity_matrix = {}

    #step 2 build similarity_matrix table the value w(u1,u2) is the number of movies u1 and u2 have been watched
    for movie, users in movie2users.items():
        for U_1 in users:
            users_smilarity_matrix.setdefault(U_1, defaultdict(int))
            for U_2 in users:
                if U_1 == U_2:
                    continue
                if use_iif==False:   # do not consider the popular factor
                    users_smilarity_matrix[U_1][U_2] += 1
                if use_iif==True:    # delow popularity item weight.
                    users_smilarity_matrix[U_1][U_2] += 1 / math.log(1 + len(users))
      
    # Calulate the similarity w(u1,u2)=w(u2,u1)
    # W(u1,u2)=len(common_movies_of_u1_u2)/sqrt(len(u1_wathed_moives)*len(u2_watched_movies))
    for U_1, related_users in users_smilarity_matrix.items():
        u1_wathed_moives = len(trainingset[U_1])
        for U_2, count in related_users.items():
            u2_wathed_moives = len(trainingset[U_2])
            users_smilarity_matrix[U_1][U_2] = count / math.sqrt(u1_wathed_moives * u2_wathed_moives)               
    return users_smilarity_matrix, popular_movies, Total_movie_num


#Calculate the movies similarity

def compute_sim_items(trainingset, use_iuf=True):
   
    popular_movies, Total_movie_num = calculate_popular_movies(trainingset)
    movie_similarity_matrix = {}
    #step one count the common user numbers for each pair of moives
    #W(m1,m2)=numer_of_users => the number of users who have been watched both moive1 and movie2 
    for user, movies in trainingset.items():
        for movie1 in movies:
            movie_similarity_matrix.setdefault(movie1, defaultdict(int))
            for movie2 in movies:
                if movie1 == movie2:
                    continue 
                if use_iuf==False:
                    movie_similarity_matrix[movie1][movie2] += 1
                if use_iuf==True:
                    movie_similarity_matrix[movie1][movie2] += 1 / math.log(1 + len(movies))

    #step 2 calulate the similarity for each moive pair
    # w(m1,m2)=common_users_numbers/ sqrt(viewers_of_moive1*viewers_of_moive2)
    for movie1, related_items in movie_similarity_matrix.items():
        viewers_of_moive1 = popular_movies[movie1]
        for movie2, count in related_items.items():
            viewers_of_moive2 = popular_movies[movie2]
            movie_similarity_matrix[movie1][movie2] = count / math.sqrt(viewers_of_moive1 * viewers_of_moive2)
    return movie_similarity_matrix, popular_movies, Total_movie_num


def calculate_popular_movies(trainingset):
    popular_movies = defaultdict(int)
    for user, movies in trainingset.items():
        for movie in movies:
            # count item popularity
            popular_movies[movie] += 1
    Total_movie_num = len(popular_movies)
    return popular_movies, Total_movie_num




def initia_model(model_type):
    '''
    model_name= {
        0 : "unkonwn",
        1 : "user_based",
        2 : "user_based_iif",
        3 : "item_based"
        4 : "item_based_iuf",
        5 : "most_popular",
        6 : "random",
        7 : "LFM"
    }
    '''
    if model_type==1:
        return UserBasedCF(K,N)
    if model_type==2:
        return UserBasedCF(K,N,use_iif=True)
    if model_type==3:
        return ItemBasedCF(K,N)
    if model_type==4:
        return ItemBasedCF(K,N,use_iuf=True)
    if model_type==5:
        return PopularModel(N)
    if model_type==6:
        return RandomModel(N)
    if model_type==7:
        return LFM(10, 10, 0.1, 0.01, 10)  

def do_recommend(model_number, dataset_name, TestingSet_Size=0.3, clean=False):
    model_type_dict= {
        1 : "user_based",
        2 : "user_based_iif",
        3 : "item_based",
        4 : "item_based_iuf",
        5 : "most_popular",
        6 : "random",
        7 : "LFM"
    }
    if model_number in model_type_dict.keys():
        model_type=model_type_dict[model_number]
    else:
        raise ValueError('No model named ' + model_type)

    print('-' * 108)
    print('\t\tThis is %s model trained on %s with TestingSet_Size = %.2f\n\t\trecommend_movies_number=%s K= %s recommend_user_number= %s' % (model_type, dataset_name, TestingSet_Size,N,K,recommend_user_number))
    print('-' * 108 + '\n')
                      
    mManager = ModelManager(dataset_name, TestingSet_Size)
    try:
        trainingset = mManager.load_model('trainingset')
        testingset = mManager.load_model('testingset')
    except OSError:
        ratings = DataSet.load_dataset(name=dataset_name)
        trainingset, testingset = DataSet.train_test_split(ratings, TestingSet_Size=TestingSet_Size)
        mManager.save_model(trainingset, 'trainingset')
        mManager.save_model(testingset, 'testingset')
    
    mManager.clean_workspace(clean)
    
 
    model=initia_model(model_number)
    model.training_Model(trainingset)

    #generate recommend user list,and recommend moives to them, print the result
    random.seed(random_seed)
    recommend_user_list=[]
    for i in range(recommend_user_number):
        random_number=random.randint(1,1000)
        if random_number not in recommend_user_list and str(random_number) in trainingset.keys():
            recommend_user_list.append(random_number)
    for user in recommend_user_list:
        recommend_result=model.recommend(str(user))
        print("recommend for userid = %s:" % user)
        print(recommend_result)
        print()    
   
    #recommend_test(model,recommend_user_list )

    #using the test set to evaluate the model, recommend user list is the all users in the testing set
    model.evaluation(testingset)



if __name__ == '__main__':
    
    #dataset_name = 'ml-100k'
    dataset_name = 'ml-1m'
    #model_number = 1 #'UserCF' 
    #model_number =2 #'UserCF-IIF'
    #model_number =3 #'ItemCF'
    #model_number =4 #'ItemCF-IUF'
    #model_number =5 #'PopularModel'
    #model_number =6 #'Random'
    #model_number =7 # 'LFM'
    TestingSet_Size = Ratio
    '''
    handle the inpute arguments 
    '''
    input_agr_num=5
    data_set_num=1
    if len(sys.argv)>5 or len(sys.argv)<1:
        print("\t\tInput error\n\t\tThe correct format is: \tpython3 CFSystem model_type number_of_users N DataSet\n\
                \n\t\tModel_type: \n\t\t1: user based Model\n\t\t2: imporved user based Model\
                \n\t\t3: item based Model\n\t\t4: imporved item based Model\
                \n\t\t5: popular based Model\n\t\t6: random Model \
                \n\n\t\t#users: the number of redcommend users (>=1)\
                    \n\t\tN: the number of recommend movies(>=1)  \n \
                    \n\t\tData set:\n\t\t1: ml_100k\n\t\t2: ml_1m \n")
        sys.exit()

    
    if len(sys.argv)==5:
        try:
            model_number=int(sys.argv[1])
            recommend_user_number=int(sys.argv[2])
            N=int(sys.argv[3])
            data_set_num=int(sys.argv[4])
        except ValueError:
            print("Input error, please input again.")
            sys.exit()
        if N<1 or recommend_user_number<1 or N>500 or recommend_user_number>800 or (data_set_num!=1 and data_set_num!=2):
            print("Input error, please input again.")
            sys.exit()
        if data_set_num==1:
            dataset_name='ml-100k'
        else:
            dataset_name='ml-1m'
    if len(sys.argv)==4:
        try:
            model_number=int(sys.argv[1])
            recommend_user_number=int(sys.argv[2])
            N=int(sys.argv[3])
            
        except ValueError:
            print("Input error, please input again.")
            sys.exit()
        if N<1 or recommend_user_number<1 or N>500 or recommend_user_number>800 :
            print("Input error, please input again.")
            sys.exit()
        
        dataset_name='ml-100k'

    if len(sys.argv)==3:
        try:
            model_number=int(sys.argv[1])
            recommend_user_number=int(sys.argv[2])
            N=10
            
        except ValueError:
            print("Input error, please input again.")
            sys.exit()
        if N<1 or recommend_user_number<1 or N>500 or recommend_user_number>800 or (data_set_num!=1 and data_set_num!=2):
            print("Input error, please input again.")
            sys.exit()
        
        dataset_name='ml-100k'
        
    if len(sys.argv)==2:
        try:
            model_number=int(sys.argv[1])
            recommend_user_number=10
            N=10
            
        except ValueError:
            print("Input error, please input again.")
            sys.exit()
        if N<1 or recommend_user_number<1 or N>500 or recommend_user_number>800 :
            print("Input error, please input again.")
            sys.exit()
        
        dataset_name='ml-100k'
    
    if len(sys.argv)==1: # only input: python3 CF.py
        dataset_name='ml-100k'
        model_number=1
        N=10
        recommend_user_number=10
        
    print("dataset--------------",dataset_name)
        
    

    
              
    do_recommend(model_number, dataset_name, TestingSet_Size, True)











    
