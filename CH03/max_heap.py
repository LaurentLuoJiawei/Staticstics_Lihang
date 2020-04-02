# -*- utf-8 -*-
class max_heap(object):
    """
    通过list构建大顶堆，实现找到k个距离最近的元素
    数据格式[kdnode,distance,index]
    """
    def __init__(self,k):
        """
        初始化
        :param k:
        """
        self._heap = []
        self.k = k
    @property
    def get_heap(self):
        """
        返回顶堆列表
        :return:
        """
        return self._heap
    def get_parent_index(self,index):
        """
        通过自身节点索引找到父节点
        :param index:
        :return:
        """
        p_index = index//2
        return p_index

    def get_left_child_idx(self,index):
        return 2*index
    def get_right_child_index(self,index):
        return 2*index+1
    def get_dis(self,index,X):
        return np.sum(np.square(self._heap[index].data -X))
    def get_len(self):
        return len(self._heap)

    def max_heapify(self,index,X):
        """
        保持顶堆的顶部元素的距离是最大的
        :param index:
        :param X:
        :return:
        """
        left_child_idx= self.get_left_child_idx(index)
        right_child_idx = self.get_right_child_index(index)
        len = self.get_len()
        largest =index
        if left_child_idx<len and self.get_dis(left_child_idx,X) > self.get_dis(largest,X):
            largest = left_child_idx
        if right_child_idx<len and self.get_dis(largest,X) > self.get_dis(right_child_idx,X):
            largest = right_child_idx
        if largest != index:
            temp = self._heap[index]
            self._heap[index] = self._heap[largest]
            self._heap[largest] = temp
            self.max_heapify(index)
    def propage_up(self,index,X):
        """
        在index位置(尾部)插入新节点后。更新顶堆保持，顶部元素距离最大
        :param index:
        :return:
        """
        parent_idx = self.get_parent_index(index)
        while index != 0 and self.get_dis(parent_idx,X) < self.get_dis(index,X):
            self._heap[index],self._heap[parent_idx] = self._heap[parent_idx],self._heap[index]
            index = parent_idx

    def add(self, k, node:KdTreeNode, X):
        """在顶堆尾部插入新节点
        如果顶堆未满，直接插，
        如果已经满了，那么删除顶部元素，进行树的旋转，再插入尾部，再进行位置调整。
        :param k:
        :param node:
        :param X:
        :return:
        """
        if k > self.get_len():
            self._heap.append(node)
        else:
            self.delete_top(X)
            self.heap_append(node,X)

    def heap_append(self,node:KdTreeNode,X):
        self._heap.append(node)
        self.propage_up(self.get_len()-1,X)

    def delete_top(self,X):
        if self.get_len() == 0:
            print("堆栈已空")
        max = self._heap[0]
        data = self._heap.pop()
        if self.get_len()>0:
            self._heap[0] = data
            self.max_heapify(0,X=X)


