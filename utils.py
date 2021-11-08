import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

EDGE_THRESH = 0.9
COLORS = ['green','blue','gray','red','yellow']

def _get_node_list(graph_nodes, node_classes):
    node_names = []
    for node in list(graph_nodes):
        node_names.append(node_classes[node.argmax()])
    return node_names

def _visualize_graph(graph_nodes, graph_edges, node_classes, edge_classes):
    G = nx.Graph()
    graph_node_names = _get_node_list(graph_nodes, node_classes)
    G.add_nodes_from(graph_node_names)
    print('Class colors are :')
    for e,c in zip(edge_classes,COLORS):
        print(e,' : ',c)
    
    edges_at = list(np.argwhere(graph_edges > EDGE_THRESH))

    for n_from, n_to, edg_idx in zip(edges_at[0], edges_at[1], edges_at[2]):
        G.add_edge(graph_node_names[n_from], graph_node_names[n_to], color=COLORS[edg_idx])
    
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]

    nx.draw_networkx(G, edge_color=colors)
    # plt.title('; '.join([(e,c) for e,c in zip(edge_classes,COLORS)]))
    plt.show()

def visualize_datapoint(model, dataloader, node_classes, edge_classes):
    data_list = list(dataloader)
    inp = input('Do you want to visualize any output? (y/n)')

    while(inp == 'y' and len(data_list)>0):
        data = data_list.pop()
        edges = data['edges']
        nodes = data['nodes']
        context_curr = data['context_curr']
        context_query = data['context_query']
        y = data['y']

        x_hat = model.forward(edges, nodes, context_curr, context_query)
        _visualize_graph(nodes.squeeze(), x_hat.squeeze(), node_classes, edge_classes)

        inp = input('Do you want to visualize another output? (y/n)')
