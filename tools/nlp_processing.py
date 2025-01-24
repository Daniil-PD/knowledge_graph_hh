import pandas as pd
import networkx as nx
from pprint import pprint
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import itertools
import re
import spacy
import numpy as np
import math
from pyvis.network import Network
import pickle


nlp = spacy.load('ru_core_news_sm')


dependencies_display = {
    "obj": "Что или кого?",  # Прямое или косвенное дополнение
    "subj": "Кто или что?",  # Номинальное подлежащее
    "nmod": "Какое? Чье?",  # Модификатор, который уточняет существительное
    "mod": "Какой? Как?",  # Общий модификатор, может относиться к прилагательным или наречиям
}

def get_entities_and_relation(sent:list[str, int]):
    doc = nlp(sent[0])

    ret_list = []
    for token in doc:
        if token.dep_ in ["ROOT", "punct"]:
            continue
        if token.dep_.find("obj") >= 0:
            class_dep = "obj"
        elif token.dep_.find("subj") >= 0:
            class_dep = "subj"
        elif token.dep_.endswith("nmod"):
            class_dep = "nmod"
        elif token.dep_.endswith("mod"):
            class_dep = "mod"
        else:
            continue
        ret_list.append([ token.head.lemma_, token.lemma_, class_dep, token.dep_, sent[1]])
    return ret_list

def modifier_valve(in_dict:dict, modifier:dict):
        ret_dict = {}
        for key, value in in_dict.items():
            if value in modifier.keys():
                ret_dict[key] = modifier[value]
            else:
                ret_dict[key] = value
        return ret_dict

def make_nlp_graph(df_raw):

    # # Фильтрация
    df_mask_have_skills = df_raw["key_skills"].notna()
    print(f"filter: {df_mask_have_skills.sum()}/{df_mask_have_skills.shape[0]}  {df_mask_have_skills.sum()/df_mask_have_skills.shape[0] * 100:.3}%")
    df_have_skills = df_raw[df_mask_have_skills]

    # # Подготовка данных

    # ### Подготовка столбца навыков
    df_have_skills.loc[:, "key_skills_pars"] = df_have_skills.loc[:, "key_skills"].apply(lambda x: x.split("\n"))

    # ### Осмотр данных по навыкам
    full_skills = df_have_skills["key_skills_pars"].explode()\
        .apply(lambda x: re.sub("[^\w ]", "", x.lower()))\
        .value_counts().to_dict()


    # # nlp обработка    
    edges_to_graf = [edges for skill in tqdm(full_skills.items()) for edges in get_entities_and_relation(skill)]



    # # Построение графа
    df_edges_to_graf = pd.DataFrame({'source':[i[0] for i in edges_to_graf],
                'target':[i[1] for i in edges_to_graf],
                'relation_class':[i[2] for i in edges_to_graf],
                'relation':[i[3] for i in edges_to_graf],
                'count': [i[4] for i in edges_to_graf]
                })

    df_edges_to_graf = df_edges_to_graf.groupby(["source", "target", "relation"]).agg({
        "relation_class":"first",
        "count":"sum"
    }).reset_index()

    df_edges_to_graf["weight"] = (df_edges_to_graf["count"]/df_edges_to_graf["count"].max())**0.3 * 20
    
    G=nx.from_pandas_edgelist(df_edges_to_graf[:10000], "source", "target",
                            edge_attr=True, create_using=nx.MultiDiGraph())

    dict_degree = G.degree()
    for node in G.nodes:
        G.nodes[node]["degree"] = dict_degree[node]
        G.nodes[node]["size"] = 5 + math.sqrt(dict_degree[node])*2
    return G



def get_biggest_subgraph(G):

    components = list(nx.connected_components(G.to_undirected()))

    # Находим компонент с наибольшим количеством узлов
    largest_component = max(components, key=len)

    # Создаем подграф из самого большого компонента
    H = G.subgraph(largest_component).copy()
    return H

def get_center_node(G):
    center_node = max(list(G.degree()), key=lambda x: x[1])
    return center_node


def get_too_level_neighbors_subgraph(H: nx.MultiDiGraph, center_node):
    first_level_neighbors = [*H.neighbors(center_node[0])]
    first_level_neighbors.append(center_node[0])

    second_level_neighbors = set(first_level_neighbors)
    for neighbor in tqdm(first_level_neighbors):
        second_level_neighbors.update(H.neighbors(neighbor))

    second_level_neighbors = second_level_neighbors.difference(first_level_neighbors)
    D = H.subgraph(set(first_level_neighbors + [*second_level_neighbors] + [center_node[0]])).copy()
    return D


def get_optimal_subgraph(G, start_num, end_num):
    G_degree = [*G.degree]
    G_degree.sort(key=lambda x: x[1], reverse=True)
    G_opt = G.subgraph([node for node, _ in G_degree][start_num:end_num])
    return G_opt


def draw_graphs(G, dependencies_display):
    # ## отрисовка графов методами NetworkX
    plt.figure(figsize=(12,12))

    # pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='red', edge_cmap=plt.cm.Blues)

    edge_labels = nx.get_edge_attributes(G, 'relation')
    edge_labels = modifier_valve(edge_labels, dependencies_display)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.show()


def pos_on_circular_layout(l:list, r=1, center = (0, 0)):
    angle_step = math.pi*2/(len(l) -1)
    angle = 0
    pos = {}
    for node in l:
        pos[node] = np.array((math.sin(angle)*r + center[0], math.cos(angle)*r + center[1]))
        # pos[node] = (1, 0)
        angle += angle_step
    return pos


def draw_graphs_pyvis(graf_to_show):
    # Отрисовка графа методами pyvis

    net = Network(filter_menu=True, notebook=False) # создаём объект графа
    # net.show_buttons(filter_=["physics"])
    net.from_nx(graf_to_show)

    net.set_options("""
    const options = {
    "physics": {
        "barnesHut": {
        "gravitationalConstant": -26750,
        "centralGravity": 4.9,
        "springLength": 25,
        "springConstant": 0.055,
        "damping": 0.86
        },
        "maxVelocity": 18,
        "minVelocity": 0.75,
        "timestep": 0.2
    }
    }""")

    net.show('graph.html', notebook=False)  # save visualization in 'graph.html'


if __name__ == "__main__":

    with open('paths_data.txt', 'r') as f:
        paths = [path.replace("\n", "") for path in f.readlines()]
    print(paths)


    for path in paths:
        print("Processing", path)
        df_raw = pd.read_csv(path)
        print("Length of dataframe:", len(df_raw))
        G = make_nlp_graph(df_raw)
        # Сохранение графа в формате Pickle
        with open("graphs_pkl\\" + path.split("\\")[-1].split(".")[0] + "_nlp.pkl", 'wb') as f:
            pickle.dump(G, f)


