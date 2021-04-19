import pandas as pd
from math import *
import numpy as np
import csv

def read(movies_path, ratings_path, data_path):
  '''读取用户和电影的数据并保存'''
  movies = pd.read_csv(movies_path)               # 读取电影相关的信息
  ratings = pd.read_csv(ratings_path)             # 读取用户对电影的打分
  data = pd.merge(movies,ratings,on = 'movieId')  # 通过两数据框之间的movieId连接
  data[['userId','rating','movieId','title']].sort_values('userId').to_csv(data_path,index=False)   # 将数据用csv文件保存

def dataset(dataset_path):
  file = open(dataset_path,'r', encoding='UTF-8')      # 读取整合后的数据
  data = {}                                          # 用字典存放每位用户评论的电影和评分
  for line in file.readlines():
    line = line.strip().split(',')
    # 如果字典中没有某位用户，则使用用户ID来创建这位用户
    if not line[0] in data.keys():
        data[line[0]] = {line[3]:line[1]}
    # 否则直接添加以该用户ID为key字典中
    else:
        data[line[0]][line[3]] = line[1]
  return data 

def Euclidean(user1,user2):
    #取出两位用户评论过的电影和评分
    user1_data=data[user1]
    user2_data=data[user2]
    distance = 0
    #找到两位用户都评论过的电影，并计算欧式距离
    for key in user1_data.keys():
        if key in user2_data.keys():
            #注意，distance越大表示两者越相似
            distance += pow(float(user1_data[key])-float(user2_data[key]),2)
 
    return 1/(1+sqrt(distance))#这里返回值越小，相似度越大
 
#计算某个用户与其他用户的相似度
def top10_simliar(userID):
    res = []
    for userid in data.keys():
        #排除与自己计算相似度
        if not userid == userID:
            simliar = Euclidean(userID,userid)
            res.append((userid,simliar))
    res.sort(key=lambda val:val[1])
    return res[:4]                        #返回与该用户最相近的几个用户

#根据用户推荐电影给其他人
def recommend(user):
    #相似度最高的用户
    top_sim_user = top10_simliar(user)[0][0]
    #相似度最高的用户的观影记录
    items = data[top_sim_user]
    recommendations = []
    #筛选出该用户未观看的电影并添加到列表中
    for item in items.keys():
        if item not in data[user].keys():
            recommendations.append((str(item),str(items[item])))
    recommendations.sort(key=lambda val:val[1],reverse=True)#按照评分排序
    #返回评分最高的10部电影
    return recommendations[:10]

if __name__ == '__main__' :
  movies_path = 'movies.csv'
  ratings_path = 'ratings.csv'
  data_path = 'data.csv'

  read(movies_path, ratings_path, data_path)
  data = dataset(data_path)
  row = []
  films = []

  for i in data.keys():            
    RES = top10_simliar(i)                 # 获取每个用户相关的邻居(这里设置为4个)
    row.append(RES)        
    film = recommend(i)                    # 获取每个用户可能喜欢的电影(这里设置为10个)
    films.append(film)

  with open('user.csv','w', encoding = 'utf-8') as f:
    write=csv.writer(f)
    write.writerows(row)
  with open('films.csv','w', encoding = 'utf-8') as f1:
    write=csv.writer(f1)
    write.writerows(films)
  
      






