import random
import math
from operator import itemgetter
import csv
import pandas as pd

def dataset(filename, random_zz = 0.8):

        trainset_len = 0
        testset_len = 0
        trainset = {}                 # 训练集
        testset = {}                  # 测试集
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:                                              # 去掉文件第一行的title
                    continue
                user, movie, rating, timestamp = line.split(',')         # 获取用户，电影， 评分， 时间截等信息           
                if(random.random() < random_zz):                             # 随机生成一个数来划分训练集和测试集
                    trainset.setdefault(user, {})                            # 如果键不存在于字典中，将会添加键并将值设为默认值
                    trainset[user][movie] = rating
                    trainset_len += 1
                else:
                    testset.setdefault(user, {})
                    testset[user][movie] = rating
                    testset_len += 1
        return trainset, testset
        print('划分训练集与测试集成功！')
        print('训练集长 = %s' % trainSet_len)
        print('测试集长 = %s' % testSet_len)


def movie_sim(trainset, testset):
    '''计算电影间的相似度矩阵'''
    movie_popular = {}
    sim_matrix = {}
    
    
    #  movie_popular字典键为电影名，值为所有用户总的观看数
    for user, movies in trainset.items():                
        # items() 方法把字典中每对 key 和 value 组成一个元组，并把这些元组放在列表中返回。
            for movie in movies:
            # 计算每一部电影的流行程度
                if movie not in movie_popular:
                    movie_popular[movie] = 0
                else:
                    movie_popular[movie] += 1
                    
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
            # 计算同一个人观看的不同电影的联系
            # 如果用户看了电影1，又看了2，则【1】【2】 += 1
                    sim_matrix.setdefault(m1, {})
                    sim_matrix[m1].setdefault(m2, 0)
                    sim_matrix[m1][m2] += 1 
    movie_count = len(movie_popular)
                    
    # 计算相似性矩阵，利用刚刚得到的movie_popular 和 sim_matrix               
    for m1, related_movies in sim_matrix.items():
            for m2, count in related_movies.items():
                # 注意0向量的处理，即某电影的用户数为0
                if movie_popular[m1] == 0 or movie_popular[m2] == 0:
                    sim_matrix[m1][m2] = 0
                else:
                    sim_matrix[m1][m2] = count / math.sqrt(movie_popular[m1] * movie_popular[m2])
    return sim_matrix

# 针对目标用户U，找到K部相似的电影，并推荐其N部电影
def recommend(user,sim_matrix,k):
    rank = {}
    watched_movies = trainset[user]
    for movie, rating in watched_movies.items():
        #对目标用户每一部看过的电影，从相似电影矩阵中取与这部电影关联值最大的前K部电影，若这K部电影用户之前没有看过，则把它加入rank字典中
        for related_movie, w in sorted(sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:100]:
            if related_movie in watched_movies:
                continue
            rank.setdefault(related_movie, 0)
            #计算推荐度，用电影相似矩阵的值乘以电影评分
            rank[related_movie] += w * float(rating)
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[:k]

#产生推荐并通过准确率、召回率和覆盖率进行评估
def evaluate(trainset, testset, user,sim_matrix,k):
    # 准确率和召回率
    hit = 0
    rec_count = 0
    test_count = 0
    # 覆盖率
    all_rec_movies = set()
    for user,m in list(trainset.items())[int(user)-1 : int(user)]:
        test_moives = testset.get(user, {})
        rec_movies = recommend(user,sim_matrix,k)
        print("用户 %s 的电影推荐列表为：" % user)
        precommend(rec_movies)
        #注意，这里的w与上面recommend的w不要一样，上面的w是指计算出的相似电影矩阵的权值，而这里是这推荐字典rank对应的推荐度
        for movie, w in rec_movies:
            if movie in test_moives:
                hit += 1
            all_rec_movies.add(movie)
        rec_count += 10
        test_count += len(test_moives)

    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    print('准确率=%.4f\t召回率=%.4f' % (precision, recall))
    print ('='*100)

def precommend(rec_m):
    csv_file = "movies.csv"
    csv_data = pd.read_csv(csv_file, low_memory = False)
    df = pd.DataFrame(csv_data)
    for movieid,w in rec_m:
        print('电影名称:',df.loc[df['movieId']==int(movieid),'title'].values,'推荐度:',w)

if __name__ == '__main__':
  user = input("请输入要推荐的用户： ")
  k = int(input("请输入要推荐的电影数： "))
  trainset, testset = dataset('ratings.csv', 0.8)
  sim_matrix = movie_sim(trainset, testset)
  evaluate(trainset, testset, user,sim_matrix, k)