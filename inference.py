import numpy as np

def edge_to_str(edges, classes=['0','1','2']):
    result = ''
    for i in range(3):
        result += classes[i]+'\n'
        result += str(edges[:,:,:,i])+'\n\n'
    return result


def _densify_matrix(edges):
    edge_presence = np.sign(edges.copy())
    for _ in range(edges.shape[1]):
        new_edges = np.tensordot(edge_presence, edges, axes=[[2],[1]]).squeeze(2)
        new_edges = new_edges * (edges==0)
        if (new_edges==0).all():
            break
        edges += new_edges
    return edges

def _sparsify_matrix(edges):
    dense_edges = edges.copy()
    remove = np.tensordot(dense_edges, dense_edges, axes=[[2],[1]]).squeeze(2)
    return dense_edges * (remove==0).astype(int)

def densify(edges, edge_classes):
    null = np.ones(edges.shape[0:3])/2
    in_idx, on_idx, close_idx = edge_classes.index("INSIDE"), edge_classes.index("ON"), edge_classes.index("CLOSE")
    edges_on_in = np.stack([null, edges[:,:,:,in_idx], edges[:,:,:,on_idx]], axis=-1).argmax(axis=-1)
    edges_on_in_dense = _densify_matrix(edges_on_in)
    dense_edges = edges.copy()
    dense_edges[:,:,:,in_idx] = (edges_on_in_dense == 1).astype(int)
    dense_edges[:,:,:,on_idx] = (edges_on_in_dense == 2).astype(int)
    close_from_on_in = np.sign(edges_on_in_dense + np.swapaxes(edges_on_in_dense,1,2))
    dense_edges[:,:,:,close_idx] = np.sign(close_from_on_in+dense_edges[:,:,:,close_idx])
    return dense_edges

def sparsify(edges, edge_classes):
    null = np.ones(edges.shape[0:3])/2
    in_idx, on_idx, close_idx = edge_classes.index("INSIDE"), edge_classes.index("ON"), edge_classes.index("CLOSE")
    edges_on_in = np.stack([null, edges[:,:,:,in_idx], edges[:,:,:,on_idx]], axis=-1).argmax(axis=-1)
    edges_on_in_dense = _densify_matrix(edges_on_in)
    edges_on_in_sparse = _sparsify_matrix(edges_on_in_dense)
    sparse_edges = edges.copy()
    sparse_edges[:,:,:,in_idx] = (edges_on_in_sparse == 1).astype(int)
    sparse_edges[:,:,:,on_idx] = (edges_on_in_sparse == 2).astype(int)
    close_from_on_in = np.sign(edges_on_in_dense + np.swapaxes(edges_on_in_dense,1,2))
    sparse_edges[:,:,:,close_idx] = sparse_edges[:,:,:,close_idx] * (close_from_on_in == 0)
    return sparse_edges

def remove_close_edges(edges, edge_classes):
    in_idx, on_idx, close_idx = edge_classes.index("INSIDE"), edge_classes.index("ON"), edge_classes.index("CLOSE")
    close_from_on_in = np.sign(edges[:,:,:,in_idx] + edges[:,:,:,on_idx] + np.swapaxes(edges[:,:,:,in_idx],1,2)  + np.swapaxes(edges[:,:,:,on_idx],1,2))
    edges[:,:,:,close_idx] = edges[:,:,:,close_idx]  * (close_from_on_in == 0)
    return edges

def add_close_edges(edges, edge_classes):
    in_idx, on_idx, close_idx = edge_classes.index("INSIDE"), edge_classes.index("ON"), edge_classes.index("CLOSE")
    close_from_on_in = np.sign(edges[:,:,:,in_idx] + edges[:,:,:,on_idx] + np.swapaxes(edges[:,:,:,in_idx],1,2)  + np.swapaxes(edges[:,:,:,on_idx],1,2))
    edges[:,:,:,close_idx] = np.sign(edges[:,:,:,close_idx]  + (close_from_on_in == 0))
    return edges

class Test1():
    def __init__(self):
        self.edge_classes = {"CLOSE":0, "ON":1, "INSIDE":2}
        self.nodes = np.eye(3)
        self.sparse_edges = np.stack([np.zeros_like(self.nodes)]*len(self.edge_classes), axis=-1)
        self.sparse_edges[1,1,self.edge_classes["CLOSE"]] = 1
        self.sparse_edges[1,0,self.edge_classes["INSIDE"]] = 1
        self.sparse_edges[2,1,self.edge_classes["ON"]] = 1
        self.dense_edges = self.sparse_edges.copy()
        self.dense_edges[2,0,self.edge_classes["INSIDE"]] = 1
        self.dense_edges[2,0,self.edge_classes["CLOSE"]] = 1
        self.dense_edges[0,2,self.edge_classes["CLOSE"]] = 1
        self.dense_edges[1,0,self.edge_classes["CLOSE"]] = 1
        self.dense_edges[0,1,self.edge_classes["CLOSE"]] = 1
        self.dense_edges[2,1,self.edge_classes["CLOSE"]] = 1
        self.dense_edges[1,2,self.edge_classes["CLOSE"]] = 1
        self.sparse_edges = np.expand_dims(self.sparse_edges, axis=0)
        self.dense_edges = np.expand_dims(self.dense_edges, axis=0)
    def run(self):
        edge_classes_list = list(self.edge_classes.keys())
        actual_dense = densify(self.sparse_edges, edge_classes_list)
        assert np.array_equal(actual_dense, self.dense_edges), 'Test1 failed for densify \n Actual \n'+edge_to_str(actual_dense, edge_classes_list)+'\n Expected \n'+edge_to_str(self.dense_edges, edge_classes_list)
        print('Test1 passed for densify')
        actual_sparse = sparsify(self.dense_edges, edge_classes_list)
        assert np.array_equal(actual_sparse, self.sparse_edges), 'Test1 failed for sparsify \n Actual \n'+edge_to_str(actual_sparse, edge_classes_list)+'\n Expected \n'+edge_to_str(self.sparse_edges, edge_classes_list)
        print('Test1 passed for sparsify')


class Test2():
    def __init__(self):
        self.edge_classes = {"CLOSE":0, "ON":1, "INSIDE":2}
        self.nodes = np.eye(10)
        k = -1
        self.sparse_edges = np.stack([np.zeros_like(self.nodes)]*len(self.edge_classes), axis=-1)
        self.sparse_edges[:,:,self.edge_classes["INSIDE"]] = np.diag(np.ones((self.nodes.shape[0]-abs(k),)),k=k)
        self.dense_edges = np.stack([np.zeros_like(self.nodes)]*len(self.edge_classes), axis=-1)
        self.dense_edges[:,:,self.edge_classes["INSIDE"]] = np.tri(self.nodes.shape[0], k=k)
        self.dense_edges[:,:,self.edge_classes["CLOSE"]] = self.dense_edges[:,:,self.edge_classes["INSIDE"]] + np.swapaxes(self.dense_edges[:,:,self.edge_classes["INSIDE"]],0,1)
        self.sparse_edges = np.expand_dims(self.sparse_edges, axis=0)
        self.dense_edges = np.expand_dims(self.dense_edges, axis=0)
    def run(self):
        edge_classes_list = list(self.edge_classes.keys())
        actual_dense = densify(self.sparse_edges, edge_classes_list)
        assert np.array_equal(actual_dense, self.dense_edges), 'Test2 failed for densify \n Actual \n'+edge_to_str(actual_dense, edge_classes_list)+'\n Expected \n'+edge_to_str(self.dense_edges, edge_classes_list)
        print('Test2 passed for densify')
        actual_sparse = sparsify(self.dense_edges, edge_classes_list)
        assert np.array_equal(actual_sparse, self.sparse_edges), 'Test2 failed for sparsify \n Actual \n'+edge_to_str(actual_sparse, edge_classes_list)+'\n Expected \n'+edge_to_str(self.sparse_edges, edge_classes_list)
        print('Test2 passed for sparsify')
