# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:14:48 2017

@author: 90662
"""
#3 概率为线性变化的数 P = A*X+B
#WS小世界模型构建
import random
#import numpy 
from numpy import *
import networkx as nx
import matplotlib.pyplot as plt
        

#每个节点都与左右相邻的各K/2节点相连，K为偶数
def CreateNetwork(n,k,matrix):   
    for i in range(n):
        for j in range( k // 2 + 1):
            if i-j >= 1 and i+j <= (n-2):
                matrix[i][i-j] = matrix[i][i+j] = 1
            elif i-j < 1:
                matrix[i][(n-1)+i-j] = matrix[i][i+j] = 1
            elif i+j > (n-1):
                matrix[i][i+j-(n-1)] = matrix[i][i-j] = 1
                       
#第一次随机化重连     
def SmallWorld_1(n,k,x,matrix):
    #随机产生一个概率p_change，如果p_change < p, 重新连接边 
    A = 1.0
    B = 0.0
    p = A * x + B
    
    p_change = 0.0
    edge_change = 0
    for i in range(n):
        #t = int(k/2)
        for j in range( k // 2 + 1):
            #需要重新连接的边
            p_change = (random.randint(0,n-1)) / (double)(n)        
            #重新连接
            if p_change < p:
                #随机选择一个节点，排除自身连接和重边两种情况
                while(1):
                    node_NewConnect = (random.randint(0,n-1)) + 1                    
                    if matrix[i][node_NewConnect] == 0 and node_NewConnect != i:
                        break
                if (i+j) <= (n-1):
                    matrix[i][i+j] = matrix[i+j][i] = 0
                else:
                    matrix[i][i+j-(n-1)] = matrix[i+j-(n-1)][i] = 0
                matrix[i][node_NewConnect] = matrix[node_NewConnect][i] = 1
                edge_change += 1
                    
#第二次随机化重连
def SmallWorld_2(n,k,x,matrix):
    #随机产生一个概率p_change，如果p_change < p, 重新连接边 
    A = 1.5
    B = 0.0
    p = A * x + B
    
    p_change = 0.0
    edge_change = 0
    for i in range(n):
        #t = int(k/2)
        for j in range( k // 2 + 1):
            #需要重新连接的边
            p_change = (random.randint(0,n-1)) / (double)(n)        
            #重新连接
            if p_change < p:
                #随机选择一个节点，排除自身连接和重边两种情况
                while(1):
                    node_NewConnect = (random.randint(0,n-1)) + 1                    
                    if matrix[i][node_NewConnect] == 0 and node_NewConnect != i:
                        break
                if (i+j) <= (n-1):
                    matrix[i][i+j] = matrix[i+j][i] = 0
                else:
                    matrix[i][i+j-(n-1)] = matrix[i+j-(n-1)][i] = 0
                matrix[i][node_NewConnect] = matrix[node_NewConnect][i] = 1
                edge_change += 1
                
#第三次随机化重连
def SmallWorld_3(n,k,x,matrix):
    #随机产生一个概率p_change，如果p_change < p, 重新连接边 
    A = 1.5
    B = 0.01
    p = A * x + B
    
    p_change = 0.0
    edge_change = 0
    for i in range(n):
        #t = int(k/2)
        for j in range( k // 2 + 1):
            #需要重新连接的边
            p_change = (random.randint(0,n-1)) / (double)(n)        
            #重新连接
            if p_change < p:
                #随机选择一个节点，排除自身连接和重边两种情况
                while(1):
                    node_NewConnect = (random.randint(0,n-1)) + 1                    
                    if matrix[i][node_NewConnect] == 0 and node_NewConnect != i:
                        break
                if (i+j) <= (n-1):
                    matrix[i][i+j] = matrix[i+j][i] = 0
                else:
                    matrix[i][i+j-(n-1)] = matrix[i+j-(n-1)][i] = 0
                matrix[i][node_NewConnect] = matrix[node_NewConnect][i] = 1
                edge_change += 1
                
# 画图
def Drawmap(n,matrix,G):
#添加n个节点
    for i in range(n):
        G.add_node(i)

#添加边,if = 1 then return [(i,j)]
    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                G.add_edge(i,j)
                         
    #定义一个布局，采用circular布局方式    
    pos = nx.circular_layout(G)        
    #绘制图形
    nx.draw(G,pos,with_labels=False,node_size = 30)
    #输出方式1: 将图像存为一个png格式的图片文件      
    plt.savefig("WS-Network-byme.png")           
    #输出方式2: 在窗口中显示这幅图像
    plt.show()    

#平均群聚系数
def average_clustering(n,matrix):  
    #三元组 
    number_three_tuple = 0.0
    #三角形
    Triangle = 0.0
    #聚类系数
    clustering_coefficient = 0.0
    
    for i in range(n):
        three_tuple = 0.0
        sum_edge = 0       
        
        for j in range(n):
            if matrix[i][j] == 1 or matrix[j][i] == 1:                  
                sum_edge += 1                   
        float(sum_edge)
        #计算每个节点的三元组个数
        three_tuple = int((sum_edge*(sum_edge-1.0))/2.0)
             
        #节点i的边组成列表mylist，并且每次循环之前初始为空值
        myList = []
        for j in range(i,n):
            if matrix[i][j] == 1 or matrix[j][i] == 1:  
                myList.append(j)
        
        #如果myList中的边（i，j）等于1，则形成三角形
        for k in range(len(myList)):
            for q in range(k,len(myList)):
                if matrix[myList[k]][myList[q]] == 1 or matrix[myList[q]][myList[k]] == 1:
                    Triangle += 1                
        if three_tuple != 0:
            clustering_coefficient += (Triangle/three_tuple)
    clustering_coefficient = clustering_coefficient/n
    print('clustering_coefficient = ',clustering_coefficient)
        
  
#Floyd算法求最短路径                
def Ford(n,matrix):
    #出发点v
    #到达点w
    #中转点K  
    #初始化新的邻接矩阵new_m,路径矩阵dis
    dis = zeros((n,n),int) 
    new_m = zeros((n,n),int)
    for v in range(n):
        for w in range(n):
            dis[v][w] = w
            if matrix[v][w] == 0:
                new_m[v][w] = 6666666
            elif matrix[v][w] == 1:
                new_m[v][w] = 1
                dis[v][w] = 1
                       
    for k in range(n):
        for v in range(n):
            for w in range(n):
                #如果经过中转点的路径比两点路径短
                if (new_m[v][k] + new_m[k][w]) < new_m[v][w]:
                    new_m[v][w] = new_m[v][k] + new_m[k][w]
                    #dis[v][w] = dis[v][k]
                    dis[v][w] = 2
    #打印节点  
    sum = 0.0              
    for v in range(n):
            for w in range(v+1,n):
                #print('v= ,',v,'w = ',w)
                #print('dis[v][w] = ',dis[v][w])
                sum = sum + dis[v][w]
                float(n)

    average_shortest_path_length = sum/(n*(n-1.0)/2)
    print('average_shortest_path_length = ',average_shortest_path_length)
    
#节点度分布
def node_degree_distribution(n,matrix):
    #求节点的度
    degree = []
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += matrix[i][j]
        #print(sum)
        degree.append(sum)
    #print(degree)
    degree.sort()
    print('degree = ',degree)
    
    sum_degree= 0.0
    for i in range(n):
        sum_degree += degree[i]
    #print(sum_degree)
    
    #生成x轴序列，从1到最大度         
    x = range(len(degree))
    #将频次转换为频率，这用到Python的一个小技巧：列表内涵                      
    y = [z/sum_degree for z in degree]
    #在双对数坐标轴上绘制度分布曲线
    plt.loglog(x,y,color="blue",linewidth=2)
    #显示图表 
    plt.show()
    
#动态行为
#抗故意攻击  robustness against intentional attack 
def node_robustness(n):
    #求出度最大的点
    degree = []
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += matrix[i][j]
        degree.append(sum)
    #将度最大的点删除边
    node_flag = degree.index(max(degree))
    for i in range(n):
        matrix[node_flag][i] = 0 
        matrix[i][node_flag] = 0
    
#随机攻击  random attack
def node_random(n):
    #产生一个随机数：0到n-1
    node_flag = random.randint(0,n-1)
    print(node_flag)
    for i in range(n):
        matrix[node_flag][i] = 0 
        matrix[i][node_flag] = 0       
        

if __name__=="__main__":
    print("main")
    #输入三个参数：节点数N,参数K,概率P
    n = input("请输入节点数 n = ",)
    k = input("请输入参数（偶数） k = ",)
    x = input("请输入概率 x = ",)
    n=int(n)
    k=int(k)
    x=float(x)
    
    
    matrix = zeros((n,n),int)
    G = nx.Graph()
    
    #value = [n,k,p]
    #print("\n")
    
    CreateNetwork(n,k,matrix)
    
    #SmallWorld_1(n,k,x,matrix)
    #Drawmap(n,matrix,G)
    
    #SmallWorld_2(n,k,x,matrix)
    #Drawmap(n,matrix,G)
    
    SmallWorld_3(n,k,x,matrix)
    Drawmap(n,matrix,G)
    
    
    #被攻击前的网络特性
    #群聚系数
    average_clustering(n,matrix)
    #平均最短路径
    Ford(n,matrix)
    #节点度分布
    node_degree_distribution(n,matrix)
    
    #抗故意攻击  robustness against intentional attack 
    #重新定义图
    node_robustness(n)
    G = nx.Graph()
    Drawmap(n,matrix,G)
    
    #随机攻击  random attack
    #node_random(n)
    #重新定义图
    #G = nx.Graph()
    #Drawmap(n,matrix,G)
    
    #被攻击后的网络特性
    #群聚系数
    average_clustering(n,matrix)
    #平均最短路径
    Ford(n,matrix)
    #节点度分布
    node_degree_distribution(n,matrix)
            