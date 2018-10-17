# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:14:48 2017

@author: 90662
"""


#WS小世界模型构建
import random
#import numpy 
from numpy import *
import networkx as nx
import matplotlib.pyplot as plt
        
#每个节点都与左右相邻的各K/2节点相连，K为偶数
def CreateNetwork(n,k,p,matrix):
    for i in range(n):
        for j in range( k // 2 + 1):
            if i-j >= 1 and i+j <= (n-2):
                matrix[i][i-j] = matrix[i][i+j] = 1
            elif i-j < 1:
                matrix[i][(n-1)+i-j] = matrix[i][i+j] = 1
            elif i+j > (n-1):
                matrix[i][i+j-(n-1)] = matrix[i][i-j] = 1
                
    for i in range(n):
        for j in range(n):
            print(matrix[i][j])
        print("\n")
            
        
def SmallWorld(n,k,p,matrix):
    #随机产生一个概率p_change，如果p_change < p, 重新连接边
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
                
            else:
                print("no change\n",i+j)
    #test
    print("small world network\n")
    for i in range(n):
        for j in range(n):
            print(matrix[i][j])
        print("\n")
    print("edge_change = ",edge_change)
    print("ratio = ",(double)(edge_change)/(n*k/2))
     
          
#将matrix写入文件
def DataFile(n,k,p,matrix):
    # 打开一个文件
    f = open("C:/0network/data.txt", "w")
    for i in range(n):
        for j in range(n):     
            netdata = ','.join(str(matrix[i][j]))
            f.write(netdata)
        f.write('\n')

    # 关闭打开的文件
    f.close()
    print('end')


# 画图
def Drawmap(n,matrix,G):
#添加n个节点
    for i in range(n):
        G.add_node(i)

#添加边
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

#主函数
if __name__=="__main__":
    print("main")
    #输入三个参数：节点数N,参数K,概率P
    n = input("请输入节点数 n = ",)
    k = input("请输入参数（偶数） k = ",)
    p = input("请输入概率 p = ",)
    n=int(n)
    k=int(k)
    p=float(p)
    matrix = zeros((n,n),int)
    G = nx.Graph()
    
    value = [n,k,p]
    
    CreateNetwork(n,k,p,matrix)
    SmallWorld(n,k,p,matrix)

    print(matrix)
    #导出到一个文件中
    DataFile(n,k,p,matrix)
    #画图
    Drawmap(n,matrix,G)