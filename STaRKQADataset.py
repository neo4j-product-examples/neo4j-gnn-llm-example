import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from neo4j import Driver, GraphDatabase
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

from compute_pcst import compute_pcst


class STaRKQADataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        raw_dataset: Dataset,
        split: str = "train",
        force_reload: bool = False,
    ) -> None:
        self.split = split
        self.raw_dataset = raw_dataset
        #self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        #embedding_dimension is fixed as 1536
        # load from parent directory of this file
        self.reltype_embedding_dict = torch.load(os.path.join(os.path.dirname(__file__), 'data-loading/emb/prime/text-embedding-ada-002/doc/reltype_emb_dict.pt'))
        self.query_embedding_dict = torch.load(os.path.join(os.path.dirname(__file__), 'data-loading/emb/prime/text-embedding-ada-002/query/query_emb_dict.pt'))

        super().__init__(root, force_reload=force_reload)

        path = self.processed_paths[0]
        self.load(path)

    @property
    def processed_file_names(self) -> List[str]:
        return [self.split + '_data.pt']

    def process(self) -> None:
        load_dotenv('db.env', override=True)
        NEO4J_URI = os.getenv('NEO4J_URI')
        NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

        retrieval_data = []

        dataframe = self.raw_dataset.data.loc[self.raw_dataset.indices]
        skipped_queries = 0
        for index, qa_row in dataframe.iterrows():
            t = time.time()
            print(f"Retrieving and constructing base subgraph for row {index}...")
            print(f"Current time is: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            prompt = qa_row[1]
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                query_emb = self.query_embedding_dict[qa_row[0]].numpy()[0]
                topk_node_ids = self.get_nodes_by_vector_search(query_emb, driver)
                subgraph_rels = self.get_subgraph_rels(topk_node_ids, driver)
                if len(subgraph_rels) < 1:
                    # topk_node_ids can't form a small connected graph, skip this query
                    skipped_queries += 1
                    continue
                subgraph_rels['textEmbedding'] = self._embed(subgraph_rels['relType'])
                topk_edge_ids = self.get_edges_by_vector_search(qa_row[0], subgraph_rels)

            # process ids to consecutive tensor
            src = subgraph_rels['src'].values
            tgt = subgraph_rels['tgt'].values
            unique_nodes = np.unique(np.concatenate([src, tgt]))
            id_map = {node: i for i, node in enumerate(unique_nodes)}
            src_consecutive = [id_map[node] for node in src]
            tgt_consecutive = [id_map[node] for node in tgt]
            pcst_base_graph_topology = Data(edge_index=torch.tensor([src_consecutive, tgt_consecutive], dtype=torch.long))

            # Some topk_node_ids may not be in subgraph_rels. Drop them for now.
            mapped_topk_node_ids = [id_map[node] for node in topk_node_ids if node in id_map.keys()]

            pcst, inner_id_mapping, selected_nodes, selected_edges = compute_pcst(pcst_base_graph_topology, mapped_topk_node_ids, topk_edge_ids)
            print(f"PCST computed in {time.time() - t} seconds.")
            print("Enriched PCST with embeddings and textual description.")
            t = time.time()

            # Retrieve node embedding, label and textual graph description
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                reverse_id_map = {v: k for k, v in id_map.items()}
                pcst_nodes_original_ids = [reverse_id_map[intermediate_id] for intermediate_id in selected_nodes]

                textual_nodes_df = self.get_textual_nodes(pcst_nodes_original_ids, driver)

                node_embedding = torch.tensor(textual_nodes_df['textEmbedding'].tolist())

                textual_nodes_df.description.fillna("")
                textual_nodes_df['node_attr'] = textual_nodes_df.apply(lambda row: f"name: {row['name']}, description: {row['description']}", axis=1)
                textual_nodes_df.rename(columns={'nodeId': 'node_id'}, inplace=True)
                nodes_desc = textual_nodes_df.drop(['name', 'description', 'textEmbedding'], axis=1).to_csv(index=False)

                original_edges = [(reverse_id_map[src.item()], reverse_id_map[tgt.item()]) for src, tgt in selected_edges.t()]
                textual_edges_df = self.get_textual_edges(original_edges, driver)
                edges_desc = textual_edges_df.to_csv(index=False)

                desc = nodes_desc + '\n' + edges_desc

                answer_ids = eval(qa_row[2])
                answers = self.get_textual_nodes(answer_ids, driver)['name'].tolist()

            enriched_data = Data(
                x=node_embedding,
                edge_index=pcst.edge_index,
                edge_attr=None,  # add edge_attr if needed
                question=f"Question: {prompt}\nAnswer: ",
                label=('|').join(answers).lower(),
                desc=desc,
            )

            retrieval_data.append(enriched_data)
            print(f"Finished enriched PCST for row {index} in {time.time() - t} seconds.")

        print(f"Skipped {skipped_queries} queries due to insufficient subgraph data.")
        self.save(retrieval_data, self.processed_paths[0])


    def get_nodes_by_vector_search(self, query_embedding: np.ndarray, driver: Driver) -> List:
        """
        Given a prompt, encode it with OpenAI's API and search for similar nodes in the SKB graph in Neo4j

        :param driver:
        :return: A list of 4 node-ids that are most similar to the prompt
        """
        res = driver.execute_query("""
        CALL db.index.vector.queryNodes($index, $k, $query_embedding) YIELD node
        RETURN node.nodeId AS nodeId
        """,
                                   parameters_={
                                       "index": "text_embeddings",
                                       "k": 4,
                                       "query_embedding": query_embedding})
        return [rec.data()['nodeId'] for rec in res.records]

    def get_subgraph_rels(self, node_ids: List, driver: Driver):
        res = driver.execute_query("""
        UNWIND $nodeIds AS nodeId
        MATCH(node:_Entity_ {nodeId:nodeId})
        // create filtered cartesian product
        WITH collect(node) AS sources, collect(node) AS targets
        UNWIND sources as source
        UNWIND targets as target
        WITH source, target
        WHERE source > target //how is this calculated? on element id?...it works

        // find connecting paths
        MATCH (source)-[rl]->{0,2}(target)

        //get rels
        UNWIND rl AS r
        WITH DISTINCT r
        MATCH (m)-[r]->(n)
        RETURN
        m.nodeId AS src,
        n.nodeId AS tgt,
        type(r) AS relType
        """,
                                   parameters_={"nodeIds": node_ids})
        return pd.DataFrame([rec.data() for rec in res.records])

    def get_edges_by_vector_search(self, qa_row_id: int, subgraph_rels: DataFrame) -> np.ndarray:
        """
        Given a prompt find the most similar edges in the subgraph

        :param driver:
        :return: A list of 4 edges (node pairs) that are most similar to the prompt
        """

        prompt_emb = self.query_embedding_dict[qa_row_id]
        sims = cosine_similarity(prompt_emb, np.vstack(subgraph_rels["textEmbedding"].values))
        k = min(4, len(subgraph_rels))
        indices = np.argpartition(sims[0], -k)[-k:]

        return indices[np.argsort(sims[0][indices])[::-1]]

    def _chunks(self, xs, n=500):
        n = max(1, n)
        return [xs[i:i + n] for i in range(0, len(xs), n)]

    # def _embed(self, doc_list, chunk_size=500):
    #     embeddings = []
    #     for docs in self._chunks(doc_list, chunk_size):
    #         try:
    #             embeddings.extend(self.embedding_model.embed_documents(docs))
    #         except Exception as e:
    #             print(f"Error while embedding the documents: {e}")
    #             exit()
    #         time.sleep(1)
    #     return embeddings

    def _embed(self, reltype_list: List[str]):
        return [self.reltype_embedding_dict[reltype] for reltype in reltype_list]


    def get_textual_nodes(self, node_ids: List, driver: Driver) -> DataFrame:
        res = driver.execute_query("""
        UNWIND $nodeIds AS nodeId
        MATCH(node:_Entity_ {nodeId:nodeId})
        RETURN node.nodeId AS nodeId, node.name AS name, node.details AS description, node.textEmbedding AS textEmbedding
        """,
                                   parameters_={"nodeIds": node_ids})
        return pd.DataFrame([rec.data() for rec in res.records])

    def get_textual_edges(self, node_pairs: List[Tuple[int, int]], driver: Driver) -> DataFrame:
        res = driver.execute_query("""
        UNWIND $node_pairs AS pair
        MATCH(src:_Entity_ {nodeId:pair[0]})-[e]->(tgt:_Entity_ {nodeId:pair[1]})
        RETURN src.nodeId AS src, type(e) AS edge_attr, tgt.nodeId AS dst
        """,
                                   parameters_={"node_pairs": node_pairs})
        return pd.DataFrame([rec.data() for rec in res.records])