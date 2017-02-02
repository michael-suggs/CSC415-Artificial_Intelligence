__author__ = "Michael J. Suggs // mjs3607@uncw.edu"

class KDTree:

    def __init__(self):
        self.root = None
        self.size = 0

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, item):
        pass

    def treeify_data(self, data):
        data.sort(key=lambda x: int(x[0]))
        mid = data[len(data)//2]
        line_len = len(data[0])

        self.add_node(mid)
        self._treeify_data(data[:mid], 1, line_len)
        self._treeify_data(data[mid:], 1, line_len)

    def _treeify_data(self, data_sub, i, line_len):
        if len(data_sub) == 0:
            return
        elif len(data_sub) == 1:
            self.add_node(data_sub[0])
            return
        else:
            data_sub.sort(key=lambda x: int(x[i % line_len]))
            mid = data_sub[len(data_sub)//2]
            self.add_node(mid)
            self._treeify_data(data_sub[:mid], i + 1, line_len)
            self._treeify_data(data_sub[mid:], i + 1, line_len)

    def add_node(self, point):
        if self.root:
            self._add_node(point, self.root)
        else:
            self.root = KDNode(point)
        self.size += 1

    def _add_node(self, point, parent):
        pass


class KDNode:

    def __init__(self, point, parent=None, l_child=None, r_child=None):
        self.point = point
        self.parent = parent
        self.l_child = l_child
        self.r_child = r_child

    def depth(self):
        if self.parent is None:
            return 0
        else:
            return self.depth(self.parent) + 1
