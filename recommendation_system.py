#Ke Xu

import pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error
import math
'----------------------------------------------data processing------------------------------------------------------------'
df = pd.read_csv('movies.csv')
df1 = pd.read_csv('ratings.csv')

total_number_of_users = max(df1['userId'])
total_number_of_movies = len(df)
title = df['title']
dataset = np.zeros((total_number_of_users, total_number_of_movies))

#form a rating matrix
movieId = df['movieId'].tolist()
for i in range(len(df1)):
    movieIndex = movieId.index(df1['movieId'][i])
    userId = df1['userId'][i]
    dataset[userId-1][movieIndex] = df1['rating'][i]

#Find mean ratings of each movie
mean_rating = []
for i in range(total_number_of_movies):
    temp = np.array(dataset[:,i])
    temp_sum = np.sum(temp)
    mean_rating.append(temp_sum/(len(np.nonzero(temp)[0])))
mean_rating = np.array(mean_rating)
moveratting = np.nonzero(dataset)

#find user weight
number_user = df1.groupby(['userId']).count()   
sumcount = np.sum(np.array(number_user['rating']))
userweight = []
temp_user = np.array(number_user['rating'])
for i in range(len(temp_user)):
    userweight.append(temp_user[i]/sumcount)

#find movie weight
popular = df1.groupby(['movieId']).count()
popular = popular['userId']
movie_popular= np.zeros(total_number_of_movies)
movie_popular1 = np.array(movieId)
movie_popular = np.column_stack((movie_popular1,movie_popular)).astype(int)
a=np.array(popular.index)
for i in range(len(movie_popular)):
    if movie_popular[i][0] in a:
        index = movie_popular[i][0]
        movie_popular[i][1] = popular[index]
popular_sum = np.sum(movie_popular[:,1])
popularity = movie_popular[:,1]/popular_sum

'-------------------------------------------------SGD, matrix factorization-----------------------------------------------'
#popularity is movie weight 
def recommender(r, p, q, bi, bu, k, mean_rating, popularity, userweight, iteration, beta):
    q = q.T
    #find all nonzero value's index
    nonzeros = np.nonzero(r)
    #SGD
    for it in range(iteration):
        temp_i = np.random.randint(len(nonzeros[0]))
        temp_j = np.random.randint(len(nonzeros[1]))
        i = nonzeros[0][temp_i]
        j = nonzeros[1][temp_j]
        
        #compute error
        eij1 = mean_rating[j]+ bi[i]+bu[i]+np.dot(p[i,:],q[:,j])
        eij = r[i][j] - (popularity[j]*userweight[i]*eij1)   
        
        #update p and q where k is number of latent factor
        for k1 in range(k):
            p[i][k1] = p[i][k1] + (1/(it+1)) *(popularity[j] * userweight[i] * eij * q[k1][j] - beta * p[i][k1])
            q[k1][j] = q[k1][j] + (1/(it+1)) *(popularity[j] * userweight[i] * eij * p[i][k1] - beta * q[k1][j])
        
        #update bi and bu
        bi[i] = bi[i] + (1/(it+1))* (popularity[j] * userweight[i]*eij  - beta * bi[i])
        bu[i] = bu[i] + (1/(it+1))* (popularity[j] * userweight[i]*eij  - beta * bu[i])
    return p, q.T, bi, bu

'---------------------------------------------------test--------------------------------------------------------------------'
r = np.array(dataset)
n = len(r)
m = len(r[0])

#latent factor
k=1

#initialize p and q (since k =1 so both of them are vectores)
p = np.random.randn(n, k)
q = np.random.randn(m, k)

#initialize bi and bu
bi = np.zeros((len(r),1))
bu = np.zeros((len(r),1))

#implement Algorithm and output the result with 1000000 iteration, lambda= 5 and k latent factors
p1, q1, bi1, bu1 = recommender(r,p,q, bi, bu, k, mean_rating, popularity, userweight, 1000000, 5)

#form a rating matrix
nr = np.dot(p1, q1.T)
for i in range(len(nr[:,0])):
    nr[:,i] = nr[:,i]+bi1[:,0]+ bu1[:,0]
for i in range(len(nr[:,0])):
    for l in range(len(nr[0])):
        nr[i][l] = nr[i][l] + mean_rating[l]

nozero = np.nonzero(r)
real =[]
test = []
for i in range(len(nozero[0])):
    temi = nozero[0][i]
    temj = nozero[1][i]
    real.append(r[temi][temj])
    test.append(nr[temi][temj])

#compute error rate
e = math.sqrt(mean_squared_error(real,test))
print('RMSE=', e)

'-------------------------------------------make recommendation for user1-----------------------------------------------------'
#find which movie user1 has watched
real_data = np.nonzero(r[:,0])

user1 = np.array(nr[0]).tolist()

#find 20 highest rating of user1
listindex = sorted(range(len(user1)), reverse=True , key=lambda k: user1[k])[:20]

#print the movies recommendation
print('recommendation for user1:')
for i in range(len(listindex)):
    if listindex[i] not in real_data[0]:
        print(df['title'][listindex[i]])
    

