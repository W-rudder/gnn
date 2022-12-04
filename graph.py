import numpy as np
import torch

class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
       
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        
        self.off_set_l = off_set_l
        
        self.uniform = uniform
        
    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1]) #按边顺序排序
            n_idx_l.extend([x[0] for x in curr])#点
            e_idx_l.extend([x[1] for x in curr])#边
            n_ts_l.extend([x[2] for x in curr])#时间戳
           
            
            off_set_l.append(len(n_idx_l))
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l, e_idx_l, off_set_l
        
    def find_before(self, src_idx, cut_time):
        """
    
        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]# 取出所有相邻点
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]# 取出所有相邻边的时间戳
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]# 取出所有相邻边
        
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_idx

        left = 0
        right = len(neighbors_idx) - 1
        
        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid
        
        # not include cut_time
        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]# 该时间之前的

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32) # [bs,20]
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)# [bs,20]
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)# [bs,20]
        
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]#第i个节点的邻居中随机sample出20个
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx
                    
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def extract_subgraph(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Sampling the k-hop sub graph
        
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        time_array = np.array(cut_time_l)
        
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]

        # max_node = max(src_idx_l)
        # max_node = max(max_node, np.max(x))

        for _ in range(k-1):
            src_idx_list = node_records[-1].flatten()
            time_array = time_array.reshape(np.size(time_array), 1)
            time_array = np.pad(time_array, ((0, 0), (0, 19)), mode='edge').flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(src_idx_list, time_array, num_neighbors)
            # max_node = max(max_node, np.max(out_ngh_node_batch)) # [bs*num_neighbors, num_neighbors]

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)

        # print(node_records[0], max_node)
        # get temporal adj list
        temporal_adj_list = [[set() for _ in range(num_neighbors * (num_neighbors + 1) + 1)] for _ in range(len(src_idx_l))]
        node_dict = [dict() for _ in range(len(src_idx_l))]

        for hop in range(k):
            # cnt = 0
            if hop == 0:
                src_idx_list = src_idx_l
            else:
                src_idx_list = node_records[hop-1].flatten()

            ngn_node_array = node_records[hop].reshape(-1, num_neighbors)
            ngn_eidx_array = eidx_records[hop].reshape(-1, num_neighbors)
            ngn_t_array = t_records[hop].reshape(-1, num_neighbors)

            # print('{}:{} {}'.format(hop, src_idx_list, len(src_idx_list)))
            for i in range(len(src_idx_list)):
                # 区分是哪个节点的子图
                node_index = i // np.power(num_neighbors, hop)
                node_id = node_index

                src_idx = src_idx_list[i]
                ngn_node = ngn_node_array[i]
                ngn_edix = ngn_eidx_array[i]
                ngn_t = ngn_t_array[i]
                assert(len(ngn_node) == num_neighbors)
                for j in range(len(ngn_node)):
                    # 跳过mask节点
                    if ngn_edix[j] == 0:
                        continue
                    # cnt += 1
                    idx_src = get_index(node_dict=node_dict, node_id=node_id, src_id=src_idx)
                    idx_ngh = get_index(node_dict=node_dict, node_id=node_id, src_id=ngn_node[j])
                    temporal_adj_list[node_id][idx_src].add((ngn_node[j], ngn_edix[j], ngn_t[j]))
                    temporal_adj_list[node_id][idx_ngh].add((src_idx, ngn_edix[j], ngn_t[j]))
            # print(cnt)
        for i in range(5):
            cnt1 = np.count_nonzero(node_records[0][i, :])
            cnt2 = np.count_nonzero(node_records[1][i*20:(i+1)*20, :])
            res = set(np.concatenate((node_records[0][i, :].flatten(), node_records[1][i*20:(i+1)*20, :].flatten())).flatten())
            res.remove(0)
            res.add(src_idx_l[i])
            res = len(list(res))
            print(cnt1, cnt2, res)
        return temporal_adj_list, node_dict
    def subgragh_to_adj(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        temporal_list, node_dict = self.extract_subgraph(k, src_idx_l, cut_time_l, num_neighbors)
        ls = temporal_list
        print(len(src_idx_l), len(ls))
        assert(len(src_idx_l) == len(ls))

        max_node_cnt = len(ls[0])
        print("max_cnt:{}".format(max_node_cnt))
        src_list = np.zeros((len(src_idx_l), max_node_cnt)).astype(np.int32)
        src_t_list = np.array(cut_time_l).reshape(len(cut_time_l), 1)
        src_t_list = np.pad(src_t_list, ((0, 0), (0, max_node_cnt-1)), 'edge')
        ngh_list = np.zeros((len(src_idx_l), max_node_cnt, num_neighbors)).astype(np.int32)
        ngh_t_list = np.zeros((len(src_idx_l), max_node_cnt, num_neighbors)).astype(np.int32)
        ngh_e_list = np.zeros((len(src_idx_l), max_node_cnt, num_neighbors)).astype(np.int32)
        for i in range(len(src_idx_l)):
            for j in range(max_node_cnt):
                if len(ls[i][j]) != 0:
                    tmp_ls = list(ls[i][j])
                    if len(ls[i][j]) > 20:
                        tmp_ls.sort(key=lambda x: x[2])
                        tmp_ls = tmp_ls[-20:]
                    
                    src_list[i, j] = list(node_dict[i].keys())[j]

                    ngh = [idx for idx, eidx, t in tmp_ls]
                    ngh_t = [t for idx, eidx, t in tmp_ls]
                    ngh_e = [eidx for idx, eidx, t in tmp_ls]
                    ngh_list[i, j, (num_neighbors - len(ngh)):] = np.array(ngh)
                    ngh_t_list[i, j, (num_neighbors - len(ngh)):] = np.array(ngh_t)
                    ngh_e_list[i, j, (num_neighbors - len(ngh)):] = np.array(ngh_e)
        return src_list, src_t_list, ngh_list, ngh_t_list, ngh_e_list

def get_index(node_dict, node_id, src_id):
    if src_id in node_dict[node_id].keys():
        return node_dict[node_id][src_id]
    else:
        assert(len(node_dict[node_id].keys()) < 421)
        node_dict[node_id][src_id] = len(node_dict[node_id].keys())
        return node_dict[node_id][src_id]

