# Movie-recommedation-system

1. Introduction

This is the movie recommendation system project which was written by Laser, Jack, Aedgen and Alex. In this project we implemented the two kinds of KNN algorithms which are user_based and item_based CF. We also improve the use_based and item_based methods to improve the CF performance. Moreover, we provide two methods which are based on random and item_popularity. 
 
2. project contents
   --program file:  CF.py
   --data file:     data     
   --report file:   report.pdf
   --readme file:   readme.txt

 Data is a directory which contains two sub_directories ml-1m and ml_100k.
 These two databases can be downloaded in:
 https://grouplens.org/datasets/movielens/100k/
 https://grouplens.org/datasets/movielens/1m/
 Each sub_directory contain four files which are movies.dat, users.dat, ratings.dat and README.txt
 All our report is based on 1m database, 100k database is used for testing convenience.
    		
3. Running program
   --make sure the data file and CF.py in the same directory.
   --open a new terminal under the folder.
   --#python3 CF.py parameter_1  parameter_2  parameter_3  parameter_4

   parameter_1 is the Model type:
      1. user_based       (default)
      2. user_based_iif
      3. item_based
      4. item_based_iuf
      5. most_popular
      6. random

   parameter_2 is the number of recommend users.  (The default value is 10)

   Parameter_3 is the number of recommend movies.(The default value is 10)

   Parameter_4 is the size of dataset 
      1. ml-100k (default)
      2. ml-1m

   Eg. If we  choose model=user_based, recommend user number=10, recommend movie number=8
         and dataset=ml-100k then we input command like below.
        
    #  python3 CF.py 1 10 20 1

  If we want use the all default parameters then we only need to type command below:
  #  python3 CF.py
 

Note: In order to simplify testing, we use the number of users instead of import the user id list, the program will choose users randomly.

Note: If we use the ml_1m as the data set,  so  the program will take several minutes to output the results. 

4. fixed parameters

dataset ratio : After many times tests, we choose 80% data for training set and 20% for testing.

K nearest neighbours: we tried the K value for 5 to 160, finally we choose the 10 as K value in order to balance both precision and coverage.


5. Output
------------------------------------------------------------------------------------------------------------
		This is user_based model trained on ml-100k with TestingSet_Size = 0.20
		recommend_movies_number=10 K= 20 recommend_user_number= 10
------------------------------------------------------------------------------------------------------------
Load ml-100k dataset success.
recommend for userid = 591:
['211', '186', '318', '173', '234', '174', '168', '423', '153', '204']

recommend for userid = 792:
['117', '288', '25', '628', '286', '685', '273', '50', '111', '258']

recommend for userid = 336:
['64', '172', '22', '69', '98', '174', '12', '216', '763', '357']

recommend for userid = 272:
['318', '185', '89', '603', '181', '173', '182', '195', '180', '527']

recommend for userid = 515:
['313', '678', '333', '286', '302', '751', '326', '327', '272', '259']

recommend for userid = 299:
['124', '238', '79', '195', '172', '216', '8', '181', '425', '154']

recommend for userid = 326:
['208', '181', '191', '168', '28', '133', '175', '205', '197', '71']

recommend for userid = 291:
['181', '173', '56', '222', '568', '186', '239', '174', '7', '183']

recommend for userid = 661:
['176', '234', '98', '195', '186', '474', '56', '58', '194', '22']

recommend for userid = 580:
['117', '742', '237', '471', '127', '118', '313', '111', '281', '248']

------------------------------------------------------------------------------------------------------------

				CF SYSTEM PERFOMANCE RESULTS:

	precision=0.3113  	recall=0.1479    	coverage=0.2155    	popularity=5.4070

------------------------------------------------------------------------------------------------------------


