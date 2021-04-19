import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder

def dataset(path):
  '''数据获取以及初步处理'''

  data = pd.read_csv(path)
  cols = data.columns.values                            # 获取数据的标签，方便之后进行数据处理
  dense_feats = [i for i in cols if i[0] == "I"]        # 数据主要分为两种类型
  sparse_feats = [j for j in cols if j[0] == "C"]       

  #首先对标签为dense_feats的数据进行处理
  data_dense = data.copy()                                       # 保留原数据
  data_dense = data_dense[dense_feats].fillna(0.0)               # 缺失值填补
  for f in dense_feats:
    data_dense[f] = data_dense[f].apply(lambda x: np.log(x+1) if x > -1 else -1)      
  
  # 再对标签为sparse_feats的数据进行处理
  data_sparse = data.copy()
  data_sparse = data_sparse[sparse_feats].fillna("-1")                   # 空白值补-1
  for f in sparse_feats:
    label_encoder = LabelEncoder()          # 将数据离散化
    data_sparse[f] = label_encoder.fit_transform(data_sparse[f])

  total_data = pd.concat([data_dense, data_sparse], axis=1)               #再将两种数据联立起来
  total_data['label'] = data['label']

  return dense_feats, sparse_feats, data_dense, data_sparse, total_data

def data_deal(dense_feats, sparse_feats, data_dense, data_sparse):
  '''对数据进行特征处理'''

  # 先构造 dense 特征的输入
  dense_inputs = []
  for f in dense_feats:
    _input = Input([1], name=f)
    dense_inputs.append(_input)
  # 将输入拼接到一起，方便连接 Dense 层
  # 若axis=0，则要求除了a.shape[0]和b.shape[0]可以不等之外，其它维度必须相等。此时c.shape[0] = a.shape[0]+b.shape[0]
  # 若axis=1，则要求除了a.shape[1]和b.shape[1]可以不等之外，其它维度必须相等。此时c.shape[1] = a.shape[1]+b.shape[1]
  concat_dense_inputs = Concatenate(axis=1)(dense_inputs)  
  # 然后连上输出为1个单元的全连接层，表示对 dense 变量的加权求和
  fst_order_dense_layer = Dense(1)(concat_dense_inputs)

  # 再对 sparse 特征构造输入，目的是方便后面构造二阶组合特征
  sparse_inputs = []
  for f in sparse_feats:
    _input = Input([1], name=f)
    sparse_inputs.append(_input)
    
  sparse_1d_embed = []
  for i, _input in enumerate(sparse_inputs):
    f = sparse_feats[i]
    voc_size = total_data[f].nunique()
    # 使用 l2 正则化防止过拟合
    reg = tf.keras.regularizers.l2(0.5)
    _embed = Embedding(voc_size, 1, embeddings_regularizer=reg)(_input)
    # 由于 Embedding 的结果是二维的，
    # 因此如果需要在 Embedding 之后加入 Dense 层，则需要先连接上 Flatten 层
    _embed = Flatten()(_embed)
    sparse_1d_embed.append(_embed)
  # 对每个 embedding lookup 的结果 wi 求和
  fst_order_sparse_layer = Add()(sparse_1d_embed)
  # 再将dense 和 sparse的特征联合在一起
  linear_part = Add()([fst_order_dense_layer, fst_order_sparse_layer])    

  return linear_part,dense_inputs,sparse_inputs

def embedding(sparse_inputs, sparse_feats, total_data):
  '''进行特征组合之前，每个 sparse 特征需要先进行 embedding'''

  k = 8
  # 只考虑sparse的二阶交叉
  sparse_kd_embed = []
  for i, _input in enumerate(sparse_inputs):
    f = sparse_feats[i]
    voc_size = total_data[f].nunique()
    reg = tf.keras.regularizers.l2(0.7)                 # 防止过拟合
    # embedding为嵌入层，嵌入层可以降维，也可以升维，这里是降维
    _embed = Embedding(voc_size, k, embeddings_regularizer=reg)(_input)  
    sparse_kd_embed.append(_embed)

  #接下来就是要进行特征组合，如果对 n 个 sparse 特征两两组合，那么复杂度应该是O（n^），但是可以用计算技巧降为O(kn)
  # 1.将所有sparse的embedding拼接起来，得到 (n, k)的矩阵，其中n为特征数，k为embedding大小
  concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)
  # 2.先求和再平方
  sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(concat_sparse_kd_embed) 
  square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])  
  # 3.先平方再求和
  square_kd_embed = Multiply()([concat_sparse_kd_embed, concat_sparse_kd_embed]) 
  sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed)  
  # 4.相减除以2
  sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed]) 
  sub = Lambda(lambda x: x*0.5)(sub)  
  snd_order_sparse_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub)  

  return snd_order_sparse_layer, concat_sparse_kd_embed

def DNN(concat_sparse_kd_embed,linear_part, snd_order_sparse_layer):
  '''接下来构造DNN部分'''
  flatten_sparse_embed = Flatten()(concat_sparse_kd_embed)  
  fc_layer = Dropout(0.5)(Dense(256, activation='relu')(flatten_sparse_embed)) 
  fc_layer = Dropout(0.3)(Dense(256, activation='relu')(fc_layer))  
  fc_layer = Dropout(0.1)(Dense(256, activation='relu')(fc_layer))  
  fc_layer_output = Dense(1)(fc_layer) 
  output_layer = Add()([linear_part, snd_order_sparse_layer, fc_layer_output])
  output_layer = Activation("sigmoid")(output_layer)

  return output_layer, fc_layer_output

def add_layer(linear_part, snd_order_sparse_layer, fc_layer_output):
  #将dnn和fm联立起来
  output_layer = Add()([linear_part, snd_order_sparse_layer, fc_layer_output])
  output_layer = Activation("sigmoid")(output_layer)

  model = Model(dense_inputs+sparse_inputs, output_layer)
  model.compile(optimizer="adam", 
              loss="binary_crossentropy", 
              metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])
  
  return model 

def train(total_data,dense_feats,sparse_feats,model):
  train_data = total_data.loc[:500000-1]
  valid_data = total_data.loc[500000:]

  train_dense_x = [train_data[f].values for f in dense_feats]
  train_sparse_x = [train_data[f].values for f in sparse_feats]
  train_label = [train_data['label'].values]

  val_dense_x = [valid_data[f].values for f in dense_feats]
  val_sparse_x = [valid_data[f].values for f in sparse_feats]
  val_label = [valid_data['label'].values]

  model.fit(train_dense_x+train_sparse_x, 
          train_label, epochs=7, batch_size=256,
          validation_data=(val_dense_x+val_sparse_x, val_label),
          )
  model.save('my_mode.h5')

if __name__ == '__main__':
  dense_feats, sparse_feats, data_dense, data_sparse, total_data = dataset('criteo_sampled_data.csv')
  linear_part,dense_inputs,sparse_inputs = data_deal(dense_feats, sparse_feats, data_dense, data_sparse)
  snd_order_sparse_layer, concat_sparse_kd_embed = embedding(sparse_inputs, sparse_feats, total_data)
  output_layer, fc_layer_output = DNN(concat_sparse_kd_embed,linear_part, snd_order_sparse_layer)
  model = add_layer(linear_part, snd_order_sparse_layer, fc_layer_output)
  train(total_data,dense_feats,sparse_feats,model)


