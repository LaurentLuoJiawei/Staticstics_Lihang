#-*- coding:utf-8-*-
import math
import numpy as np
import time
import random
class KdTreeNode(object):
    def __init__(self,data,dim,lf_child,rt_child,label,parent):
        self.data = data 
        self.range = dim #分割域
        self.lf_child= lf_child # 左子树
        self.rt_child = rt_child #右子树
        self.parent = parent
        self.label = label 

class KdTree(object):
    def __init__(self,dataList,labelList):
        self._length = 0
        self._root = self._createTree(dataList,labelList)
    def _createTree(self,dataList,labelList,parentnoede=None):
        """
        对kd 树初始化
        :param dataList: list
        :param labelList: list
        :return:
        """
        # assert len(dataList) == len(labelList)
        #获取样本集对大小
        dataArray = np.array(dataList)
        sample_num,dim_num = dataArray.shape
        labelArray = np.array(labelList).reshape(-1,1)
        if sample_num == 0:
            return None
        else:
            # 获取方差最大的维度,并获取最大方差维度下的中位数的样本位置
            max_var_index = np.argmax([np.var(dataArray[:,col]) for col in range(dim_num)],0)
            # print("max_var",max_var_index)
            median_index_list = np.argsort(dataArray[:,max_var_index])
            # print("median_list",len(median_index_list))
            median_index = median_index_list[sample_num//2]
            # print("median_index",median_index)
            if sample_num == 1:
                self._length += 1
                node = KdTreeNode(data=dataList[median_index,:], dim=max_var_index, lf_child=None, rt_child=None, label= labelList[median_index], parent=parentnoede)
                return node

            label = labelList[median_index_list[median_index]]
            node = KdTreeNode(data=dataArray[median_index,:],dim=max_var_index,lf_child=None,rt_child=None, label= label, parent=parentnoede)
            #构建当前节点
            lf_child_data = dataArray[median_index_list[:sample_num//2]]
            if lf_child_data.size == dim_num:
                lf_child_data=lf_child_data.reshape(1,dim_num)
            lf_child_label = labelArray[median_index_list[:sample_num//2]]
            lf_subtree = self._createTree(lf_child_data,lf_child_label,parentnoede=node)
            if sample_num  == 2:
                rt_subtree = None
            else:
                rt_child_data = dataArray[median_index_list[sample_num//2+1:]]
                if rt_child_data.size == dim_num:
                    rt_child_data = rt_child_data.reshape(1,dim_num)
                rt_child_label = labelArray[median_index_list[sample_num//2+1:]]
                rt_subtree = self._createTree(rt_child_data, rt_child_label,parentnoede=node)
            node.lf_child = lf_subtree
            node.rt_child = rt_subtree
            self._length += 1
            return node
    # 构造kd树的属性
    @property
    def get_len(self):
        return self._length
    @property
    def root(self):
        return self._root

    def transferTree2Dict(self,node):
        """
        将树结构转化为字典格式输出，使用节点的data作为key，然后[key1=data]->['key2='属性'']
        :param node: 父节点
        :return:
        """
        if node is None:
            return None
        treeDict = {}
        treeDict[tuple(node.data)] = {}
        treeDict[tuple(node.data)]['label'] = node.label[0]
        treeDict[tuple(node.data)]['dim'] = node.range
        treeDict[tuple(node.data)]['parent'] = tuple(node.parent.data) if node.parent else None
        treeDict[tuple(node.data)]['left_child'] = self.transferTree2Dict(node.lf_child)
        treeDict[tuple(node.data)]['right_child'] = self.transferTree2Dict(node.rt_child)
        return  treeDict
    def transferTree2List(self,node:KdTreeNode,datalist:list)->list:
        """
        将树转化为列表输出 首先还是字典形式保存节点内容，但是外部数据结构是list
        :param dictlist:
        :return: 返回列表形式的树结构
        """
        if node is None:
            return None
        treeDict = {}
        treeDict['data'] = tuple(node.data)
        treeDict['dim'] = node.range
        treeDict['label'] = node.label
        treeDict['parent'] = tuple(node.parent.data) if node.parent else None
        treeDict['right_child'] = tuple(node.lf_child.data) if node.lf_child else None
        treeDict['left_child'] = tuple(node.rt_child.data) if node.rt_child else None
        datalist.append(treeDict)
        self.transferTree2List(node.lf_child,datalist)
        self.transferTree2List(node.rt_child,datalist)
        return datalist


def nearest_search(kdtree:KdTree,X:list):
    """
    从根节点出发，进行搜索,找到最邻近的点
    :param kdtree:
    :param X:
    :return:
    """
    if kdtree.get_len == 1:
        return kdtree.root
    node = kdtree.root
    dim = node.range
    while 1:
        if node.data[dim] == X[dim]:
            return node
        if node.data[dim] > X[dim]:
            if node.lf_child is None:
                return node
            else:
                node = node.lf_child
        elif node.data[dim] < X[dim]:
            if node.rt_child is None:
                return node
            else:
                node = node.rt_child


def k_search(kdtree:KdTree,X:list,k):
    """
    1. 判断k与树的长度，若k大于树的总长度，直接返回整颗树里面的节点，并统计类别总量
    2. 树>k时，构建candidate_list（也可以用小顶堆）
    3. 首先找到树分支意义上最近的节点，放入list，并计算distance，如果这个节点不是叶子节点就把子节点加入计算list，判断是不是小雨最小距离。
    4. 回溯父亲节点，判断父节点的距离是否小于最小距离，如果小于，则把父亲节点的另一个子节点放入list，开始遍历
    5. 直到根节点
    :param kdTree:
    :param X: 样本
    :param K: 搜索的数量 1-5
    :return:
    """
    if k > kdtree.get_len:
        """
        如果k> 树的总长度，则直接遍历整个树获取类别的数量
        """
        kd_list = []
        labels = {}
        kdtree.transferTree2List(kdtree.root,kd_list) # 获得list形式的kd树
        for i in kd_list:
            if i['label'] in labels:
                labels[i['label']] +=1
            else:
                labels[i['label']] = 1
        #给标签排序
        sorted_label = sorted(labels.items(),key=lambda x:x[1])
        return sorted_label[0][0]
    """
    对于树总长度>k的情况，先找到直观上的最近节点，并计算最小距离
    """
    X = np.array(X)
    node = nearest_search(kdtree,X)
    if node is None:
        return None
    else:
        print("距离 X{} 最近的点为 {}".format(X,node.data))
    min_dist = dis(X,treenode=node)
    candidate_node_list = []
    candidate_node = [min_dist,node.data,node.label[0]]
    candidate_node_list.append(candidate_node)
    if node.lf_child is not None:
        lf_dist = dis(X,node.lf_child)
        if k > len(candidate_node_list) or lf_dist < min_dist:
            candidate_node_list.append([lf_dist,tuple(node.lf_child.data),node.lf_child.label[0]])
            sorted_list = sorted(candidate_node_list,key= lambda x:x[0],reverse=True)
            min_dist = sorted_list[-1][0] if k>len(candidate_node_list) else sorted_list[k-1][0]
    while True:
        """
        若返回到了根节点则搜索结束
        """
        if node == kdtree.root:
            break
        parent_node = node.parent
        p_dist = dis(X,parent_node)
        if k>len(candidate_node_list) or p_dist < min_dist:
            candidate_node_list.append([p_dist,tuple(parent_node.data),parent_node.label[0]])
            sorted_list = sorted(candidate_node_list,key= lambda x:x[0],reverse=True)
            min_dist = sorted_list[-1][0] if k>len(candidate_node_list) else sorted_list[k-1][0]
        #关于是否要遍历另外一颗子树的判定条件还是没有理解！！
        if k>len(candidate_node_list) or abs(parent_node.data[parent_node.range] -X[parent_node.range])< min_dist:
            other_child_node = parent_node.lf_child if parent_node.lf_child != node else parent_node.rt_child
            if other_child_node is not None:
                if X[parent_node.range] - parent_node.data[parent_node.range] >= 0:
                    # 则为左子树
                    lf_search(other_child_node,X,candidate_node_list,k)
                else:
                    #为右子树进行遍历
                    rt_search(other_child_node,X,candidate_node_list,k)
        node = parent_node
    """
    结束循环后，对candidate_list 进行排序，选出前k个样本的类别进行统计。
    """
    labels = {}
    k_node_list = candidate_node_list[:k]
    for i in k_node_list:
        # print(i,labels)
        if i[2] in labels:
            labels[i[2]] +=1
        else:
            labels[i[2]] = 1
    sorted_labels = sorted(labels.items(),key=lambda x:x[1],reverse=True)
    return sorted_labels[0][1],k_node_list



def lf_search(node,X,candidate_node_list,k):
    """
    1。 按照前序遍历左中右，左搜索下左子树的左子树与圆相交，那么左子树的右子树距离待测点更近。所以通过左中右遍历能够更快找到圆相交的边界。
    2。 首先对list排序，然后判断是否是叶子节点，如果是叶子节点，判断是否需要更新list，如果不需要则返回上一层
    2。 返回出来，先进行排序，然后比对根节点，判断是否要更新。然后判断是否需要遍历根节点的另一子树
    3。 如果需要遍历另一边子树，继续递归
    :param node:
    :param X:
    :param min_dist:
    :param candidate_node_list:
    :param k:
    :return:
    """
    candidate_node_list = sorted(candidate_node_list,key=lambda x:x[0],reverse=True)
    min_dist = candidate_node_list[-1][0] if k > len(candidate_node_list) else candidate_node_list[k-1][0]
    if node.lf_child is None and node.rt_child is None:
        temp_dis = dis(X,node)
        if temp_dis< min_dist:
            candidate_node_list.append([temp_dis,tuple(node.data),node.label[0]])
        return
    if node.lf_child is None:
        temp_dis = dis(X,node)
        if temp_dis< min_dist:
            candidate_node_list.append([temp_dis,tuple(node.data),node.label[0]])
        return
    lf_search(node.lf_child,X,candidate_node_list,k)
    candidate_node_list = sorted(candidate_node_list, key=lambda x: x[0],reverse=True)
    min_dist = candidate_node_list[-1][0] if k > len(candidate_node_list) else candidate_node_list[k-1][0]
    temp_dist = dis(X,node)
    if k>len(candidate_node_list) or temp_dist<min_dist:
        candidate_node_list.append([temp_dist,tuple(node.data),node.label[0]])
    # 判断是否要进入另一个子树
    # print(node.data)
    if k>len(candidate_node_list) or abs(node.data[node.range] - X[node.range])<min_dist:
        if node.rt_child is not None:
            lf_search(node.rt_child,X,candidate_node_list,k)

    return  candidate_node_list
def rt_search(node,X,candidate_node_list,k):
    """
    同理若要搜索右子，按照右边中左，去寻找圆覆盖的最大区域。
    :param node:
    :param X:
    :param min_dist:
    :param candidate_node_list:
    :param k:
    :return:
    """
    candidate_node_list = sorted(candidate_node_list, key=lambda x: x[0],reverse=True)
    min_dist = candidate_node_list[-1][0] if k > len(candidate_node_list) else candidate_node_list[k - 1][0]
    # print(node.data)
    if node.lf_child is None and node.rt_child is None:
        temp_dis = dis(X, node)
        if temp_dis < min_dist:
            candidate_node_list.append([temp_dis, tuple(node.data), node.label[0]])
        return
    if node.rt_child is None:
        temp_dis = dis(X, node)
        if temp_dis < min_dist:
            candidate_node_list.append([temp_dis, tuple(node.data), node.label[0]])
        return
    rt_search(node.rt_child, X,candidate_node_list, k)
    candidate_node_list = sorted(candidate_node_list, key=lambda x: x[0],reverse=True)
    min_dist = candidate_node_list[-1][0] if k > len(candidate_node_list) else candidate_node_list[k - 1][0]
    temp_dist = dis(X, node)
    if k > len(candidate_node_list) or temp_dist < min_dist:
        candidate_node_list.append([temp_dist, tuple(node.data), node.label[0]])
    # 判断是否要进入另一个子树
    if k > len(candidate_node_list) or abs(node.data[node.range] - X[node.range]) < min_dist:
        if node.lf_child is not None:
            rt_search(node.lf_child, X, candidate_node_list, k)

    return candidate_node_list


def dis( X,treenode:KdTreeNode):
    return np.sum(np.square(X-treenode.data))




def Klinearsearch(datalist,labellist,X,k):
    """
    构造用于线性list 用于对比kd树的搜索效果
    :param seed:
    :return: 按照随机种子数生成的列表
    """
    t1 = time.time()
    x = np.array(X)
    dis_list = []
    for i in datalist:
        distance = np.sum(np.square(X,i))
        dis_list.append(distance)
    sorted_list = sorted(dis_list,reverse=True)
    return sorted_list[:k]








if __name__ == '__main__':
    # 构造原始数据
    t1 = time.time()
    dim_num = 100
    length = 50000
    seed = 5
    K = 80
    kdsearch_avg_cost = 0.0
    linearsearch_avg_cost = 0.0
    test_node =[]
    for i in range(dim_num):
        test_node.append(random.randint(1,20))
    for i in range(10):
        print("开始循环")
        np.random.seed(seed+i)
        data = np.random.randint(0,20,size=(length,dim_num))
        labels = np.random.randint(0,3,size=(length,1))
        kdtree = KdTree(dataList=data,labelList=labels)
        # kd_dict = kdtree.transferTree2Dict(kdtree.root)
        # print('kd_dict:',kd_dict)
        # kdlist =[]
        # kd_list = kdtree.transferTree2List(kdtree.root,kdlist)
        # print(len(kdlist))
        # print('kd_list',kd_list)

        # node = nearest_search(kdtree,test_node)
        # print('%s最近的叶结点:%s'%(test_node,tuple(node.data)))
        t2 = time.time()
        label, node_list = k_search(kdtree,test_node,k=K)
        # print('点%s的最接近的前k个点为:%s' % (test_node, node_list))
        # print('点%s的标签:%s' % (test_node, label))
        t3 = time.time()
        # print('创建树耗时：', t2 - t1)
        print('搜索前k个最近邻点耗时：', t3 - t2)
        t4 = time.time()
        Klinearsearch(datalist=data,labellist=labels,X=test_node,k=K)
        t5 = time.time()
        print("线性搜索k个最邻近耗时： ",t5-t4)
        one_ksearch_cost = t3-t2
        kdsearch_avg_cost+= one_ksearch_cost
        one_linearsearch_cost = t5-t4
        linearsearch_avg_cost+=one_linearsearch_cost
    print("测试结束，kd树10次平均搜索花费：" ,(kdsearch_avg_cost/10))
    print("测试结束，线性搜索平均花费：",(linearsearch_avg_cost/10))


