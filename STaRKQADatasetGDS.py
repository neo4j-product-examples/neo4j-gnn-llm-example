import os

import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from graphdatascience import GraphDataScience
from pandas import DataFrame
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from compute_metrics import compute_intermediate_metrics


def get_nodes_by_vector_search(query_embedding: np.ndarray, k_nodes: int, driver: Driver) -> list[int]:
    res = driver.execute_query("""
    CALL db.index.vector.queryNodes($index, $k, $query_embedding) YIELD node
    RETURN node.nodeId AS nodeId
    """,
                               parameters_={
                                   "index": "text_embeddings",
                                   "k": k_nodes,
                                   "query_embedding": query_embedding})
    return [rec.data()['nodeId'] for rec in res.records]

def cypher_retrieval(node_ids: list[int], driver: Driver):
    res = driver.execute_query("""
                    UNWIND $nodeIds AS nodeId
                    MATCH (m {nodeId:nodeId})-[r]->(n)
                    RETURN
                    m.nodeId as sourceNodeId, n.nodeId as targetNodeId, type(r) as relationshipType,
                    labels(m)[0] as sourceNodeType, labels(n)[0] as targetNodeType
                """,
                               parameters_={'nodeIds': node_ids})
    return pd.DataFrame([rec.data() for rec in res.records])

def get_textual_nodes(node_ids: list[int], driver: Driver) -> DataFrame:
    res = driver.execute_query("""
    UNWIND $nodeIds AS nodeId
    MATCH(node:_Entity_ {nodeId:nodeId})
    RETURN node.nodeId AS nodeId, node.name AS name, node.details AS description, node.textEmbedding AS textEmbedding
    """,
                               parameters_={"nodeIds": node_ids})
    return pd.DataFrame([rec.data() for rec in res.records])

def get_textual_edges(node_pairs: list[tuple[int, int]], driver: Driver) -> DataFrame:
    res = driver.execute_query("""
    UNWIND $node_pairs AS pair
    MATCH(src:_Entity_ {nodeId:pair[0]})-[e]->(tgt:_Entity_ {nodeId:pair[1]})
    RETURN src.nodeId AS src, type(e) AS edge_attr, tgt.nodeId AS dst
    """,
                               parameters_={"node_pairs": node_pairs})
    return pd.DataFrame([rec.data() for rec in res.records])

def textualize_graph(textual_nodes_df, textual_edges_df):
    textual_nodes_df.description.fillna("")
    textual_nodes_df['node_attr'] = textual_nodes_df.apply(
        lambda row: f"name: {row['name']}, description: {row['description']}", axis=1)
    textual_nodes_df.rename(columns={'nodeId': 'node_id'}, inplace=True)
    nodes_desc = textual_nodes_df.drop(['name', 'description', 'textEmbedding'], axis=1).to_csv(index=False)
    edges_desc = textual_edges_df.to_csv(index=False)
    return nodes_desc + '\n' + edges_desc

def assign_node_prizes(nodes_df, topn_nodes):
    nodes = nodes_df['nodeId'].tolist()
    node_prizes = {node: len(topn_nodes) - rank for rank, node in enumerate(topn_nodes)}
    node_prizes = [4 / len(topn_nodes) * node_prizes.get(node, 0) for node in nodes]
    nodes_df['nodePrize'] = node_prizes

def assign_edge_costs(relationships_df, topn_edges=None):
    edge_costs = .5 - np.zeros(len(relationships_df))
    relationships_df['edgeCost'] = edge_costs # No edge prizes for now (recall drops 3pts, f1 is about the same)

def convert_pcst_output(pcst_output) -> (np.array, np.array):
    pcst_src = pcst_output['nodeId'].values
    pcst_tgt = pcst_output['parentId'].values
    pcst_nodes = np.unique(np.concatenate((pcst_src, pcst_tgt)))
    pcst_edges = np.stack((pcst_src, pcst_tgt), axis=1)
    return pcst_nodes, pcst_edges


class STaRKQADataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        raw_dataset: Dataset,
        retrieval_config_version: int,
        algo_config_version: int,
        split: str = "train",
        force_reload: bool = False,
    ) -> None:
        self.split = split
        self.raw_dataset = raw_dataset
        self.retrieval_config_version = retrieval_config_version
        self.algo_config_version = algo_config_version
        self.query_embedding_dict = torch.load(os.path.join(os.path.dirname(__file__), 'data-loading/emb/prime/text-embedding-ada-002/query/query_emb_dict.pt')) # load from parent directory of this file

        super().__init__(root, force_reload=force_reload)

        path = self.processed_paths[0]
        self.load(path)

    @property
    def processed_file_names(self) -> list[str]:
        return [self.split + '_data.pt']

    def process(self) -> None:
        load_dotenv('db.env', override=True)
        NEO4J_URI = os.getenv('NEO4J_URI')
        NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

        retrieval_data = []

        dataframe = self.raw_dataset.data.loc[self.raw_dataset.indices]
        answer_ids = {index : eval(qa_row[2]) for index, qa_row in dataframe.iterrows()}

        # Cypher query retrieval
        with open(f"configs/retrieval_config_v{self.retrieval_config_version}.yaml", "r") as f:
            cypher_config = yaml.safe_load(f)

        base_subgraph_folder = os.path.join(os.path.dirname(__file__), f'base_subgraphs/v{self.retrieval_config_version}/')
        base_subgraph_file = f"{base_subgraph_folder}{self.split}_data_base_subgraph.pt"

        if os.path.exists(base_subgraph_file):
            print(f"Load precomputed base subgraphs from disk...")
            base_subgraph = torch.load(base_subgraph_file)
        else:
            print("Retrieve base subgraphs for each question...")
            base_subgraph = {}
            for index, (question_id, _, _) in tqdm(dataframe.iterrows()):
                query_emb = self.query_embedding_dict[question_id].numpy()[0]
                with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                    topk_node_ids = get_nodes_by_vector_search(query_emb, 25*cypher_config['k_nodes'], driver)[:cypher_config['k_nodes']]
                    relationships_df = cypher_retrieval(topk_node_ids, driver) # Variations of cypher queries are supported here

                nodes = np.unique(np.concatenate((relationships_df['sourceNodeId'].values, relationships_df['targetNodeId'].values)))
                nodes_df = pd.DataFrame({'nodeId': nodes})

                base_subgraph[index] = (nodes_df, relationships_df)

            os.makedirs(base_subgraph_folder, exist_ok=True)
            torch.save(base_subgraph, base_subgraph_file)

            compute_intermediate_metrics(answer_ids, {k: v[0]['nodeId'].tolist() for k,v in base_subgraph.items()})

        # PCST subgraph pruning
        print(f"Compute PCST graphs...")

        with open(f"configs/algo_config_v{self.algo_config_version}.yaml", "r") as f:
            pcst_config = yaml.safe_load(f)

        all_pcst_nodes = {} # for metrics only
        for index, (question_id, prompt, _) in tqdm(dataframe.iterrows()):
            query_emb = self.query_embedding_dict[question_id].numpy()[0]
            nodes_df, relationships_df = base_subgraph[index]

            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                topn_nodes = get_nodes_by_vector_search(query_emb, pcst_config["prized_nodes"], driver)
                topk_nodes = get_nodes_by_vector_search(query_emb, pcst_config["topk_nodes"], driver) # for union

            assign_node_prizes(nodes_df, topn_nodes) #adds column 'nodePrizes'
            assign_edge_costs(relationships_df) #adds column 'edgeCosts'

            # Run the pcst algorithm
            gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            with gds.graph.construct(graph_name='pcst-graph', nodes=nodes_df, relationships=relationships_df.drop(['sourceNodeType','targetNodeType'], axis=1), undirected_relationship_types=['*']) as G:
                pcst_output = gds.prizeSteinerTree.stream(G, prizeProperty='nodePrize', relationshipWeightProperty='edgeCost')
            pcst_nodes, pcst_edges = convert_pcst_output(pcst_output)

            # Take union with top25
            pcst_nodes = np.unique(np.concatenate((pcst_nodes, topk_nodes)))

            # Retrieve node embedding, label and textual graph description
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                textual_nodes_df = get_textual_nodes(pcst_nodes, driver)
                textual_edges_df = get_textual_edges(pcst_edges, driver)
                answers = get_textual_nodes(answer_ids[index], driver)['name'].tolist()

            # Order nodes by similarity to question
            textual_nodes_df['vector_similarity'] = textual_nodes_df.apply(lambda row: row['textEmbedding'] @ query_emb, axis=1)
            textual_nodes_df = textual_nodes_df.sort_values(by=['vector_similarity'], ascending=False)
            all_pcst_nodes[index] = textual_nodes_df['nodeId'].tolist() # for metrics only

            # Generate textualized graph
            desc = textualize_graph(textual_nodes_df, textual_edges_df)
            node_embedding = torch.tensor(textual_nodes_df['textEmbedding'].tolist())
            consecutive_map = {id : i for i, id in enumerate(textual_nodes_df['node_id'].values)}
            edge_index = torch.tensor([(consecutive_map[src], consecutive_map[tgt]) for src, tgt in pcst_edges], dtype=torch.int32).T #when dtype is not specified, it becomes a float tensor when unserialized, weird.
            enriched_data = Data(
                x=node_embedding,
                edge_index=edge_index,
                edge_attr=None,
                question=f"Question: {prompt}\nAnswer: ",
                label=('|').join(answers).lower(),
                desc=desc,
            )
            retrieval_data.append(enriched_data)
        compute_intermediate_metrics(answer_ids, all_pcst_nodes)
        self.save(retrieval_data, self.processed_paths[0])