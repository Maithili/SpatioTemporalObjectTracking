import numpy as np
from numpy.core.numeric import full

edge_classes = ["CLOSE", "ON", "INSIDE"]

nodes = np.load('nodes.npy')
edges_full = np.load('edges.npy')[2,:,:,:].squeeze()

node_classes = [
{"id": 1, "class_name": "bathroom", "category": "Rooms"}, 
{"id": 67, "class_name": "bedroom", "category": "Rooms"}, 
{"id": 201, "class_name": "dining_room", "category": "Rooms"}, 
{"id": 319, "class_name": "home_office", "category": "Rooms"}, 
{"id": 1001, "class_name": "coffee", "category": "placable_objects"}, 
{"id": 1002, "class_name": "kitchen_cabinet", "category": "placable_objects"}, 
{"id": 1006, "class_name": "coffee_cup", "category": "placable_objects"}, 
{"id": 226, "class_name": "table", "category": "Furniture"}, 
{"id": 231, "class_name": "sink", "category": "Furniture"}, 
{"id": 1000, "class_name": "food_bread", "category": "placable_objects"}, 
{"id": 289, "class_name": "freezer", "category": "Appliances"}, 
{"id": 1005, "class_name": "plate", "category": "placable_objects"}, 
{"id": 1003, "class_name": "chair", "category": "Furniture"}]



class Test1():
    def __init__(self):
        self.nodes = np.eye(3)
        self.sparse_edges = np.zeros_like(self.nodes)
        self.sparse_edges[1,0] = 2
        self.sparse_edges[2,1] = 1
        self.dense_edges = self.sparse_edges.copy()
        self.dense_edges[2,0] = 2

class Test2():
    def __init__(self):
        self.nodes = np.eye(10)
        k = -1
        self.sparse_edges = np.diag(np.ones((nodes.shape[0],)),k=k)
        self.dense_edges = np.tri(nodes.shape[0], k=k)

def densify(edges):
    dense_edges = edges
    print(dense_edges)
    edge_presence = np.sign(edges.copy())
    print(edge_presence)
    for _ in range(edges.shape[0]):
        new_edges = np.matmul(edge_presence, dense_edges)
        new_edges = new_edges * (dense_edges==0)
        if (new_edges==0).all():
            break
        dense_edges += new_edges
    return dense_edges

def sparsify(edges):
    fully_dense = densify(edges.copy())
    remove = np.matmul(fully_dense, fully_dense)
    sparse = fully_dense * (remove==0).astype(int)
    return sparse

def postprocess(edges, edge_classes):
    null = np.ones(nodes.shape)/2
    edges_on_in = np.stack([null, edges[:,:,edge_classes.index("INSIDE")], edges[:,:,edge_classes.index("ON")]]).argmax(axis=-1)
    edges_on_in_dense = densify(edges_on_in)
    edges_on_in_sparse = sparsify(edges_on_in)
    edges_close = edges[:,:,edge_classes.index("CLOSE")]
    edges_close_dense = np.sign(edges_on_in_dense+edges_on_in_dense.transpose()+edges_close+edges_close.transpose())
    ## What should this do?!
    edges_close_sparse = edges_close

    sparse_edges = np.zeros_like(edges)
    sparse_edges[:,:,edge_classes.index("CLOSE")] = edges_close_sparse
    sparse_edges[:,:,edge_classes.index("INSIDE")] = (edges_on_in_sparse == 1).astype(int)
    sparse_edges[:,:,edge_classes.index("ON")] = (edges_on_in_sparse == 2).astype(int)

    dense_edges = np.zeros_like(edges)
    dense_edges[:,:,edge_classes.index("CLOSE")] = edges_close_dense
    dense_edges[:,:,edge_classes.index("INSIDE")] = (edges_on_in_dense == 1).astype(int)
    dense_edges[:,:,edge_classes.index("ON")] = (edges_on_in_dense == 2).astype(int)

    return sparse_edges, dense_edges


t = Test2()
print(sparsify(t.dense_edges))
print()
print(t.sparse_edges)

