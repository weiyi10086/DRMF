import sys
import numpy as np
# 引入ElementTree
import xml.etree.ElementTree as et
# 导入random包
import random
import os
import time

# 将每次检测出的异常数量写入文件
'''
    result:返回的结果集
    n:插入的异常数量
    生成文件 DRMFnFalse.txt
'''
def write_file(result,n):
    filename = "DRMF"+str(n)+"False.txt"
    file = open(filename, "w+")
    sum=0
    length = len(result)
    for i in result:
        sum+=i
        file.write(str(i)+'\n')
    aver = sum / (length*n)
    file.write('平均为: '+str(aver))
    file.close()



# 随机生成插入异常值的位置
# num:异常值数量
# m:矩阵行列数
def generateLocal(num,m):
    # 生成num个[1,m)的随机整数
    i = random.sample(range(0, m), num)
    j = random.sample(range(0, m), num)
    return i,j

# 生成噪声
# num:生成噪声数量
def generateGuass(num):
    x = []
    # 循环生成高斯噪声
    for i in range(num):
        y = np.random.normal(loc=1.0, scale=2.0, size=None)
        x.append(y)
    return x

# 获取文件路径
def getFileName(number):
    # 要获取文件名的文件夹路径
    folder_path = "./traffic-matrices"
    # 使用os.listdir()函数获取文件夹下的所有文件名
    file_names = os.listdir(folder_path)
    # 打印所有文件名
    result = []
    for file_name in file_names:
        file_name=folder_path+"/"+file_name
        n = readxml(file_name,number)

        result.append(n)
        print(file_name)
    result = np.array(result)
    write_file(result,number)

# 找出前k大的元素
def top_k(matrix, k):
    """
    :param matrix: 输入的矩阵
    :param k: 选出前k大的元素
    :return: 由前k大元素组成的矩阵
    """
    # 将矩阵展成一个向量，并返回降序排序的下标
    # 记录shape
    _shape = matrix.shape
    # 把二维矩阵拉直
    broad_matrix = matrix.flatten()
    # 算出最大值的索引
    ind = np.argpartition(broad_matrix, -k)[-k:]
    # 创建全0向量
    output = np.zeros_like(broad_matrix)
    # topk赋值S
    output[ind] = broad_matrix[ind]
    output = np.reshape(output, _shape)

    return output

# 读入文件数据，转换为矩阵
def readxml(filename,number):
    # 读入xml
    tree = et.ElementTree(file=filename)
    # tree = et.ElementTree(file="./IntraTM-2005-01-01-01-00.xml")
    # 获取根节点
    root = tree.getroot()
    # 生成一个空矩阵
    graph = np.zeros((23, 23))

    # #循环获取数据，生成为一个矩阵
    for tag in root:
        for a_tag in tag:
            if len(a_tag.attrib) != 0:
                source_node = int(a_tag.attrib['id'])
            else:
                continue
            for b_tag in a_tag:
                target_node = int(b_tag.attrib['id'])
                weight = float(b_tag.text)
                graph[source_node - 1][target_node - 1] = weight
    # 获取矩阵的最大和最小值，归一化矩阵
    max = np.max(graph)
    min = np.min(graph)
    m,n = graph.shape
    for i in range(m):
        for j in range(n):
            graph[i,j] = (graph[i,j]-min)/(max-min)

    # 生成噪声
    # 噪声数量
    x = generateGuass(number)
    i,j = generateLocal(number,m)
    graph[i,j] = x

    L, S = DRMF(graph, 5, 20)
    Y = S[i,j]
    n = np.count_nonzero(Y)
    return n

def DRMF(X, k, e, num_iter=5):
    keepLF = sys.maxsize
    keepSF = sys.maxsize
    # 初始化U,V,E
#     获取矩阵的行列数
    m,n=X.shape
#     初始化离群矩阵S
    S=np.zeros((m,n))


    for i in range(num_iter):
        C=X-S
        # 返回前 k 大的奇异值和奇异向量
        try:
            U, L, Vt = np.linalg.svd(C,k)
        except:
            return keepL,keepS

        con = np.zeros(m-k,dtype=float)
        new_L = L[:k]
        new_L = np.append(new_L,con)
        L = np.diag(new_L)
        # 求解C-L的F范数，并保留F范数最大的L
        LF = np.linalg.norm(C-L)

        # 记录min l
        if(LF<keepLF):
            keepLF=LF
            keepL=L
        E=X-L
        E2=np.square(E)
        S=top_k(E2,e)
        SF=np.linalg.norm(E-S)
        if (SF < keepSF):
            keepSF = SF
            keepS = S
    return keepL,keepS

def DRMF_main(n):
    # 获取程序开始时间
    start = time.perf_counter()
    # 获取文件名字
    getFileName(n)
    # 程序结束时间
    end = time.perf_counter()
    # 将程序运行时间写入文件中
    filename = "DRMF" + str(n) + "False.txt"
    print(filename)
    file = open(filename, "a")
    file.write('程序运行时间为: %s Seconds' % (end - start))
    file.close()

def main():
    n=11
    DRMF_main(n)


if __name__ == '__main__':
    main()