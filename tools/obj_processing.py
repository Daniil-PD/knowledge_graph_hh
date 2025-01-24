import pandas as pd
import networkx as nx 
from tqdm.auto import tqdm
from pyvis.network import Network  
import time
import pickle


# цвета узлов
__COLORS_NODES__ = {
    "vacancies": "#62e1ff",
    "employer": "#d77833",
    "specializations": "#d83377",
    "employer_industries": "#d833d8",
    "areas": "#f3e00d",
    "city": "#8d33d8",
    "address": "#d85fd8",
    "classes": "#a3bfdf",
    "terms": "#d8f3d8"
}

def add_edges_from_dataframe(G, df, source_col='source', target_col='target'):
    edges = []
    for index, row in df.iterrows():
        # Извлекаем начальный и конечный узлы
        source = row[source_col]
        target = row[target_col]

        # Извлекаем остальные атрибуты
        attributes = row.drop([source_col, target_col]).to_dict()

        # Создаем кортеж с ребром и атрибутами
        edges.append((source, target, attributes))

    # Добавляем рёбра в граф
    G.add_edges_from(edges)

def make_obj_graph(df_raw, add_centers_group=False):
# Загрузка данных

    for col in ["id", "employer_id"]:
        df_raw[col] = df_raw[col].astype("Int64")

    
    for col in ["employer_name", "address_city", "address_street", "address_building", "address_lat", "address_lng"]:
        df_raw[col] = df_raw[col].astype(str)

    
    for col in ["id", "employer_id"]:
        df_raw[col] = df_raw[col].astype("Int64")

    # Подготовка списков
    

    # ### Список вакансий
    df_nodes_vacancies = df_raw.loc[:,["id", "name", "key_skills",
                                    "schedule_id", "schedule_name", "accept_handicapped",
                                    "accept_kids", "experience_id", "contacts", "billing_type_id",
                                    "allow_messages", "premium", "driver_license_types",
                                    "accept_incomplete_resumes", "response_letter_required",
                                    "type_id", "salary_from", "salary_to", "salary_gross",
                                    "salary_currency", "created_at", "published_at", "employment_id"]]
    df_nodes_vacancies = df_nodes_vacancies.rename({"name":"label"}, axis="columns")
    df_nodes_vacancies["id"] = df_nodes_vacancies["id"].astype(int)
    df_nodes_vacancies = df_nodes_vacancies.set_index("id")
    df_nodes_vacancies.index = "vacan_" + df_nodes_vacancies.index.astype(str)
    df_nodes_vacancies["color"] = __COLORS_NODES__["vacancies"]
    df_nodes_vacancies["type"] = "vacancies"

    # ### Список работодателей
    df_nodes_employer = df_raw.loc[:,["employer_id", "employer_name", "employer_trusted", "employer_industries"]]\
        .drop_duplicates(["employer_id"]) # TODO: Короче, ничто не уникально!
    df_nodes_employer = df_nodes_employer.set_index("employer_id")
    df_nodes_employer.index = "employer_" + df_nodes_employer.index.astype(str)
    df_nodes_employer["color"] = __COLORS_NODES__["employer"]
    df_nodes_employer["type"] = "employer"
    df_nodes_employer["label"] = df_nodes_employer["employer_name"]

    df_edges_employer = df_raw.loc[:,["id", "employer_id"]].rename({"id":"source", "employer_id":"target"},axis=1)
    df_edges_employer["source"] = "vacan_" + df_edges_employer["source"].astype(str)
    df_edges_employer["target"] = "employer_" + df_edges_employer["target"].astype(str)

    # ### Список специализаций
    df_nodes_specializations = df_raw.loc[:,["specializations"]]
    df_nodes_specializations["specializations"] = df_nodes_specializations["specializations"].str.split("\n")
    df_nodes_specializations = df_nodes_specializations.explode("specializations")
    df_nodes_specializations = df_nodes_specializations.drop_duplicates().set_index("specializations")
    df_nodes_specializations.index = "special_" + df_nodes_specializations.index.astype(str)
    df_nodes_specializations["color"] = __COLORS_NODES__["specializations"]
    df_nodes_specializations["type"] = "specializations"
    df_edges_specializations = df_raw.loc[:,["id", "specializations"]].rename({"id":"source", "specializations":"target"},axis=1)
    df_edges_specializations["target"] = df_edges_specializations["target"].str.split("\n")
    df_edges_specializations = df_edges_specializations.explode("target")
    df_edges_specializations["target"] = "special_" + df_edges_specializations["target"].astype(str)
    df_edges_specializations["source"] = "vacan_" + df_edges_specializations["source"].astype(str)


    # ### Список индустрий
    df_nodes_employer_industries = df_raw.loc[:,["employer_industries"]].drop_duplicates()
    df_nodes_employer_industries["employer_industries"] = df_nodes_employer_industries["employer_industries"].str.split("\n")
    df_nodes_employer_industries = df_nodes_employer_industries.explode("employer_industries")
    df_nodes_employer_industries = df_nodes_employer_industries.drop_duplicates().set_index("employer_industries")
    df_nodes_employer_industries["label"] = df_nodes_employer_industries.index.astype(str)
    df_nodes_employer_industries.index = "ind_" + df_nodes_employer_industries.index.astype(str)
    df_nodes_employer_industries["color"] = __COLORS_NODES__["employer_industries"]
    df_nodes_employer_industries["type"] = "employer_industries"

    df_edges_employer_industries = df_raw.loc[:,["employer_id", "employer_industries"]].rename({"employer_id":"source", "employer_industries":"target"},axis=1)
    df_edges_employer_industries["target"] = df_edges_employer_industries["target"].str.split('\n')
    df_edges_employer_industries = df_edges_employer_industries.explode("target")
    df_edges_employer_industries["target"] = "ind_" + df_edges_employer_industries["target"].astype(str)
    df_edges_employer_industries["source"] = "employer_" + df_edges_employer_industries["source"].astype(str)

    # ### Список областей
    df_nodes_areas= df_raw.loc[:,["area_id", "area_name"]].drop_duplicates().set_index("area_id")
    df_nodes_areas.index = "area_" + df_nodes_areas.index.astype(str)
    df_nodes_areas["color"] = __COLORS_NODES__["areas"]
    df_nodes_areas["label"] = df_nodes_areas["area_name"]
    df_nodes_areas["type"] = "areas"
    df_edges_employer_areas = df_raw.loc[:,["employer_id", "area_id"]].rename({"employer_id":"source", "area_id":"target"},axis=1).drop_duplicates()
    df_edges_employer_areas["target"] = "area_" + df_edges_employer_areas["target"].astype(str)
    df_edges_employer_areas["source"] = "employer_" + df_edges_employer_areas["source"].astype(str)


    # ### Список адресов
    df_nodes_address = df_raw.loc[:,["id", "address_city", "address_street", "address_building", "address_lat", "address_lng"]]\
        .groupby(["address_city", "address_street", "address_building", "address_lat", "address_lng"]).agg({
            "id":list
        }).reset_index().reset_index(names="id_address_node")
    df_nodes_address["id_address_node"] = "address_" + df_nodes_address["id_address_node"].astype(str)
    df_nodes_address["label"] = df_nodes_address["address_city"] + ", " +\
        df_nodes_address["address_street"] + " " +\
        df_nodes_address["address_building"]

    df_nodes_city = df_nodes_address.groupby(["address_city"]).agg({
        "id_address_node":list
    }).reset_index(names="city_name").reset_index(names="id_city_node")
    df_nodes_city["id_city_node"] = "city_" + df_nodes_city["id_city_node"].astype(str)
    df_nodes_city["label"] = df_nodes_city["city_name"].astype(str)


    df_edges_address = df_nodes_address.loc[:, ["id_address_node", "id"]]\
        .explode("id")\
        .rename({"id":"source", "id_address_node":"target"},axis=1)
    df_edges_address["source"] = "vacan_" + df_edges_address["source"].astype(str)


    df_edges_city = df_nodes_city.loc[:, ["id_address_node", "id_city_node"]]\
        .explode("id_address_node")\
        .rename({"id_city_node":"source", "id_address_node":"target"},axis=1)


    df_nodes_city = df_nodes_city.drop(columns=["id_address_node"]).set_index("id_city_node")
    df_nodes_address = df_nodes_address.drop(columns=["address_city", "id"]).set_index("id_address_node")

    df_nodes_city["type"] = "city"
    df_nodes_address["type"] = "address"
    df_nodes_city["color"] = __COLORS_NODES__["city"]
    df_nodes_address["color"] = __COLORS_NODES__["address"]


    # ### Список профессиональных областей
    df_nodes_classes= df_raw.loc[:,["prof_classes_found"]].drop_duplicates().set_index("prof_classes_found")
    df_nodes_classes["label"] = df_nodes_classes.index
    df_nodes_classes.index = "classes_" + df_nodes_classes.index.astype(str)
    df_nodes_classes["type"] = "classes"
    df_nodes_classes["color"] = __COLORS_NODES__["classes"]

    df_edges_classes = df_raw.loc[:,["id", "prof_classes_found"]].rename({"id":"source", "prof_classes_found":"target"},axis=1).drop_duplicates()  
    df_edges_classes["target"] = "classes_" + df_edges_classes["target"].astype(str)
    df_edges_classes["source"] = "vacan_" + df_edges_classes["source"].astype(str)

    # ### Список найденных определений
    df_nodes_terms= df_raw.loc[:,["terms_found"]].drop_duplicates().set_index("terms_found")
    df_nodes_terms["label"] = df_nodes_terms.index
    df_nodes_terms.index = "terms_" + df_nodes_terms.index.astype(str)
    df_nodes_terms["type"] = "terms"
    df_nodes_terms["color"] = __COLORS_NODES__["terms"]

    df_edges_terms = df_raw.loc[:,["id", "terms_found"]].rename({"id":"source", "terms_found":"target"},axis=1).drop_duplicates()
    df_edges_terms["target"] = "terms_" + df_edges_terms["target"].astype(str)
    df_edges_terms["source"] = "vacan_" + df_edges_terms["source"].astype(str)

    # # Загрузка данных
    del df_raw
    list_df_nodes = {
        "vacancies": df_nodes_vacancies,
        "employer": df_nodes_employer,
        "specializations": df_nodes_specializations,
        "employer_industries": df_nodes_employer_industries,
        "employer_areas": df_nodes_areas,
        "city": df_nodes_city,
        "address": df_nodes_address,
        "classes": df_nodes_classes,
        "terms": df_nodes_terms
    }

    list_df_edges = {
        "employer": df_edges_employer,
        "specializations": df_edges_specializations,
        "employer_industries": df_edges_employer_industries,
        "employer_areas": df_edges_employer_areas,
        "city": df_edges_city,
        "address": df_edges_address,
        "classes": df_edges_classes,
        "terms": df_edges_terms
    }

    __RADIUS__ = 4000

    group_positions = {
        "vacancies": (0, 0),
        "employer": (0, __RADIUS__),
        "specializations": (__RADIUS__, 0),
        "employer_industries": (-__RADIUS__, __RADIUS__),
        "employer_areas": (__RADIUS__, __RADIUS__),
        "city": (-__RADIUS__, -__RADIUS__),
        "address": (-__RADIUS__, 0),
        "classes":(0, -__RADIUS__),
        "terms": (__RADIUS__, -__RADIUS__)
    }

    G = nx.DiGraph()


    

    for df_name, df in list_df_nodes.items():
        time_start = time.time()
        print(f"add nodes {df_name}...", end="\r")
        # ser_title = df.drop(columns=["color", "label"], errors="ignore")\
        ser_title = df\
            .astype(str)\
            .apply(lambda x: "\n".join([f"{name}: {valve}" for name, valve in x.to_dict().items()]),
                axis=1)
        ser_title.name = "title"
        G.add_nodes_from((n, dict(d)) for n, d in df.assign(title=ser_title).iterrows())
        print(f"add nodes {df_name}: done load {len(df)} nodes in {time.time()-time_start:.2f} sec")
    print("total nodes:", len(G), "\n")

    for df_name, df in list_df_edges.items():
        time_start = time.time()
        print(f"add edges {df_name}...", end="\r")
        add_edges_from_dataframe(G, df)
        print(f"add edges {df_name}: done load {len(df)} edges in {time.time()-time_start:.2f} sec")
    print("total edges:", len(G.edges()), "\n")


    if add_centers_group:
        for df_name, df in list_df_nodes.items():
            time_start = time.time()
            print(f"add center {df_name}...", end="\r")
            G.add_node(df_name,
                    shape="star",
                    physics=False,
                    x=group_positions[df_name][0],
                    y=group_positions[df_name][1],
                    color="red",
                    hidden=True
                    )
            df_new_adges = pd.DataFrame({
                "target": df_name,
                "source": df.index
                })

            add_edges_from_dataframe(G, df_new_adges)
            print(f"add center {df_name}: done in {time.time()-time_start:.2f} sec")
    return G

def show_graph_pyvis(graf_to_show: nx.DiGraph):
    net = Network(filter_menu=True) # создаём объект графа
    net.show_buttons(filter_=["physics"])
    net.from_nx(graf_to_show)
    net.toggle_physics(False)

    net.show('graph.html', local=False)




if __name__ == "__main__":

    with open('paths_data.txt', 'r') as f:
        paths = [path.replace("\n", "") for path in f.readlines()][6:]
    print(paths)


    for path in paths:
        print("\n\n" + " #"*10 + " Processing", path, " #"*10)
        df_raw = pd.read_csv(path)
        print("Length of dataframe:", len(df_raw))
        G = make_obj_graph(df_raw)
        # Сохранение графа в формате Pickle
        with open("graphs_pkl\\" + path.split("\\")[-1].split(".")[0] + "_obj.pkl", 'wb') as f:
            pickle.dump(G, f)
