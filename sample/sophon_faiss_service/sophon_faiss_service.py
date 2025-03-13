import sophon.sail as sail
from transformers import AutoTokenizer
import numpy as np
import csv
import enum
import faiss

class Embedding:
    def __init__(self, embedding_bmodel_path, tokenizer_path, tpu_id=0):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_token_length = 512
        
        self.handle = sail.Handle(tpu_id)
        self.embedding_model = sail.Engine(embedding_bmodel_path, tpu_id, sail.IOMode.SYSIO)
        self.graph_name = self.embedding_model.get_graph_names()[0]
        self.input_names = self.embedding_model.get_input_names(self.graph_name)
        self.output_names = self.embedding_model.get_output_names(self.graph_name)
        self.embedding_length = self.embedding_model.get_output_shape(self.graph_name, self.output_names[1])[1]
    
    def text2vec(self, text: str):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_token_length, return_tensors="pt")

        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        input_ids, attention_mask = input_ids.numpy(), attention_mask.numpy()

        if input_ids.shape[1] > self.max_token_length:
            input_ids = input_ids[:, :self.max_token_length]
            attention_mask = attention_mask[:, :self.max_token_length]
        elif input_ids.shape[1] < self.max_token_length:
            input_ids = np.pad(input_ids,
                                ((0, 0), (0, self.max_token_length - input_ids.shape[1])),
                                mode='constant', constant_values=0)
            attention_mask = np.pad(attention_mask,
                                    ((0, 0), (0, self.max_token_length - attention_mask.shape[1])),
                                    mode='constant', constant_values=0)
            
        input_data = { self.input_names[0]: input_ids, self.input_names[1]: attention_mask }
        outputs = self.embedding_model.process(self.graph_name, input_data)

        embeddings = outputs[self.output_names[0]][:, 0]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.squeeze(embeddings) #[768]

    
class SearchType(enum.Enum):
    IndexFlatIP = 0
    IndexFlatL2 = 1
    IndexPQADC = 2
    IndexPQSDC = 3
    
class IndexFlat:
    def __init__(self, embedding_bmodel_path, tokenizer_path, tpu_id=0, search_type=SearchType.IndexFlatIP):
        self.handle = sail.Handle(tpu_id)
        self.bmcv = sail.Bmcv(self.handle)
        
        self.embedding = Embedding(embedding_bmodel_path, tokenizer_path, tpu_id)
        self.embedding_length = self.embedding.embedding_length
        
        self.database_max_length = 100000
        self.database_vectors_tensor = sail.Tensor(self.handle, [self.database_max_length, self.embedding_length], dtype=sail.Dtype.BM_FLOAT32, own_sys_data=False, own_dev_data=True) #固定长度的device memory，用于存放数据库。
        self.database_vectors = []
        self.database_vectors_l2norm_tensor = sail.Tensor(self.handle, [self.database_max_length], dtype=sail.Dtype.BM_FLOAT32, own_sys_data=False, own_dev_data=True) #固定长度的device memory，用于存放数据库的l2_norm。
        self.database_vectors_l2norm = []
        self.database_ids = []
        self.use_official_faiss = False
        
        self.search_type = search_type
        if self.use_official_faiss:
            if search_type == SearchType.IndexFlatIP:
                print("Using IndexFlatIP...")
                self.faiss_service = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_length))
            elif search_type == SearchType.IndexFlatL2:
                self.faiss_service = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_length))
                print("Using IndexFlatL2...")
            else:
                print("unsupport search_type:", search_type)
                exit(1)
    def load_from_npy(self, vectors_file: str, ids_file: str):
        vectors = np.load(vectors_file).astype(np.float32).squeeze()
        ids = np.load(ids_file).astype(np.int64).squeeze()
        assert vectors.shape[0] == ids.shape[0], "num of vectors and ids must be equal."
        load_length = min(self.database_max_length, vectors.shape[0])
        self.database_vectors = [vectors[i] for i in range(load_length)]
        self.database_ids = [int(ids[i]) for i in range(load_length)]
        if self.search_type == SearchType.IndexFlatL2:
            self.database_vectors_l2norm = [np.sum(np.square(vectors[i])) for i in range(load_length)]
        self.renew_dev_mem()

        if self.use_official_faiss:
            self.faiss_service.add_with_ids(vectors, ids)

    def save_npy(self, vectors_file_name: str, ids_file_name: str):
        np.save(vectors_file_name, np.ascontiguousarray(self.database_vectors))
        np.save(ids_file_name, np.ascontiguousarray(self.database_ids))
    
    def renew_dev_mem(self, ):
        tensor = sail.Tensor(self.handle, np.ascontiguousarray(self.database_vectors))
        self.database_vectors_tensor.sync_s2d(tensor, 0, 0, len(self.database_vectors) * self.embedding_length)
        if self.search_type == SearchType.IndexFlatL2:
            tensor_l2norm = sail.Tensor(self.handle, np.ascontiguousarray(self.database_vectors_l2norm))
            self.database_vectors_l2norm_tensor.sync_s2d(tensor_l2norm, 0, 0, len(self.database_vectors))
            
    def add_text(self, text: str, id: int):
        if len(self.database_vectors) >= self.database_max_length:
            print("add_text failed, database is full!")
            return 
        if id in self.database_ids:
            print("add_text failed, id:{} already exists.".format(id))
        else:            
            vec = self.embedding.text2vec(text)
            vec_l2norm = np.expand_dims(np.sum(np.square(vec)), axis=0)
            #update device memory:
            tensor = sail.Tensor(self.handle, vec)
            tensor_l2norm = sail.Tensor(self.handle, vec_l2norm)
            self.database_vectors_tensor.sync_s2d(tensor, 0, len(self.database_vectors) * self.embedding_length, self.embedding_length)
            if self.search_type == SearchType.IndexFlatL2:
                self.database_vectors_l2norm_tensor.sync_s2d(tensor_l2norm, 0, len(self.database_vectors_l2norm), 1)

            self.database_vectors.append(vec)
            if self.search_type == SearchType.IndexFlatL2:
                self.database_vectors_l2norm.append(vec_l2norm)
            self.database_ids.append(id)
            
            if self.use_official_faiss:
                self.faiss_service.add_with_ids(np.expand_dims(vec, axis=0), np.expand_dims(id, axis=0))
    
    def delete_by_ids(self, ids):
        indexes = [i for i, id in enumerate(ids) if id in self.database_ids]
        for index in sorted(indexes, reverse=True):
            del self.database_ids[index]
            del self.database_vectors[index]
            if self.search_type == SearchType.IndexFlatL2:
                del self.database_vectors_l2norm[index]
        self.renew_dev_mem()

        if self.use_official_faiss:
            self.faiss_service.remove_ids(np.array(indexs))
            
    def similarity_search(self, query: str, top_k: int = 1, search_type=None):
        similarities, ids = self.similarity_search_by_queries([query], top_k, search_type)
        return similarities[0], ids[0]
        
    def similarity_search_by_queries(self, queries: list, top_k: int = 1, search_type=None):
        if top_k > len(self.database_ids):
            top_k = len(self.database_ids)
        query_vecs = []
        for query in queries:
            query_vec = self.embedding.text2vec(query)
            query_vecs.append(query_vec)
        query_vecs = np.array(query_vecs)

        similarities, idxs = [], []
        if self.use_official_faiss:
            similarities, idxs = self.faiss_service.search(query_vecs, top_k)
            return similarities, idxs
        else:
            if search_type is None:
                search_type = self.search_type
            if self.search_type == SearchType.IndexFlatIP and search_type == SearchType.IndexFlatL2:
                print("Now reset searchtype to indexflatip! You cannot use indexflatL2 on a class which initial searchtype is indexflatip.")
                search_type = self.search_type
                
            if search_type == SearchType.IndexFlatIP:
                similarities, idxs = self.bmcv.faiss_indexflatIP(query_vecs, self.database_vectors_tensor, self.embedding_length, len(queries), len(self.database_vectors), top_k)
            elif search_type == SearchType.IndexFlatL2:
                query_vecs_l2norm = np.sum(np.square(query_vecs))
                similarities, idxs = self.bmcv.faiss_indexflatL2(query_vecs, 
                                                                 query_vecs_l2norm, 
                                                                 self.database_vectors_tensor, 
                                                                 self.database_vectors_l2norm_tensor, 
                                                                 self.embedding_length, len(queries), len(self.database_vectors), top_k)
            ids = []
            for idxs_ in idxs:
                ids.append([self.database_ids[int(i)] for i in idxs_])
            
            return similarities, ids
        

class IndexPQ:
    def __init__(self, embedding_bmodel_path, tokenizer_path, tpu_id=0, search_type=SearchType.IndexPQADC, slice_num=8, n_bits=8, metric=faiss.METRIC_L2):
        self.handle = sail.Handle(tpu_id)
        self.bmcv = sail.Bmcv(self.handle)
        
        self.embedding = Embedding(embedding_bmodel_path, tokenizer_path, tpu_id)
        self.embedding_length = self.embedding.embedding_length
        self.subvec_length = self.embedding_length // slice_num
        self.slice_num = slice_num
        self.n_bits = n_bits
        self.centroids_num = 2 ** n_bits
        self.ip_metric = 0 
        if metric == faiss.METRIC_L2:
            self.ip_metric = 0
        elif metric == faiss.METRIC_INNER_PRODUCT:
            self.ip_metric = 1
        else:
            print("Unsupport metric, use L2 as default.")
        
        self.database_max_length = 100000
        self.database_vectors_pq_tensor = sail.Tensor(self.handle, 
                                                      [self.database_max_length, self.n_bits], 
                                                      dtype=sail.Dtype.BM_UINT8, 
                                                      own_sys_data=False, 
                                                      own_dev_data=True) #固定长度的device memory，用于存放数据库pq结果。
        self.database_vectors_pq = None
        self.database_vectors = []
        self.database_ids = []
        self.centroids_vectors_tensor = None
        self.centroids_vectors = None
        self.sdc_table_tensor = None
        self.sdc_table = None
        self.is_trained = False
        
        self.search_type = search_type
        self.faiss_service = faiss.IndexIDMap(faiss.IndexPQ(self.embedding_length, self.slice_num, self.n_bits, metric))
        self.faiss_internal_index = faiss.downcast_index(self.faiss_service.index)
        self.use_official_faiss = False

    def load_from_npy(self, vectors_file: str, ids_file: str):
        vectors = np.load(vectors_file).astype(np.float32).squeeze()
        ids = np.load(ids_file).astype(np.int64).squeeze()
        assert vectors.shape[0] == ids.shape[0], "num of vectors and ids must be equal."
        self.database_vectors = [vectors[i] for i in range(min(self.database_max_length, vectors.shape[0]))]
        self.database_ids = [int(ids[i]) for i in range(min(self.database_max_length, ids.shape[0]))]

    def save_npy(self, vectors_file_name: str, ids_file_name: str):
        np.save(vectors_file_name, np.ascontiguousarray(self.database_vectors))
        np.save(ids_file_name, np.ascontiguousarray(self.database_ids))
    
    def add_text(self, text: str, id: int):
        if len(self.database_vectors) >= self.database_max_length:
            print("add_text failed, database is full!")
            return 
        if id in self.database_ids:
            print("add_text failed, id:{} already exists.".format(id))
        else:            
            vec = self.embedding.text2vec(text)
            self.database_vectors.append(vec)
            self.database_ids.append(id)
        if self.is_trained:
            if self.use_official_faiss:
                self.faiss_service.add_with_ids(np.expand_dims(vec, axis=0), np.expand_dims(id, axis=0))
            else:
                pq_tensor = self.bmcv.faiss_indexPQ_encode(vec,
                                                           self.centroids_vectors,
                                                           1,
                                                           self.embedding_length,
                                                           self.slice_num,
                                                           self.centroids_num,
                                                           self.ip_metric)
                self.database_vectors_pq_tensor.sync_d2d(pq_tensor, 0, len(self.database_vectors) * self.n_bits, self.n_bits)

    def delete_by_ids(self, ids):
        indexes = [i for i, id in enumerate(ids) if id in self.database_ids]
        for index in sorted(indexes, reverse=True):
            del self.database_ids[index]
            del self.database_vectors[index]
            
        if self.is_trained:
            if self.use_official_faiss:
                self.faiss_service.remove_ids(np.array(indexs))
            else:
                self.bmcv.faiss_indexPQ_encode(self.database_vectors,
                                               self.centroids_vectors_tensor,
                                               self.database_vectors_pq_tensor,
                                               len(self.database_vectors),
                                               self.embedding_length,
                                               self.slice_num,
                                               self.centroids_num,
                                               self.ip_metric)
    
    def train(self):
        self.faiss_service.train(np.array(self.database_vectors))
        self.centroids_vectors = faiss.vector_float_to_array(self.faiss_internal_index.pq.centroids)
        self.faiss_internal_index.pq.compute_sdc_table()
        
        if self.use_official_faiss:
            self.faiss_service.add_with_ids(np.array(self.database_vectors), np.array(self.database_ids))
        else:
            self.centroids_vectors_tensor = sail.Tensor(self.handle, self.centroids_vectors, False, True)
            self.sdc_table = faiss.vector_float_to_array(self.faiss_internal_index.pq.sdc_table)
            self.sdc_table_tensor = sail.Tensor(self.handle, self.sdc_table, False, True)
            self.database_vectors_pq = self.faiss_internal_index.pq.compute_codes(np.array(self.database_vectors))
            db_pq_tensor = sail.Tensor(self.handle, self.database_vectors_pq)
            self.database_vectors_pq_tensor.sync_s2d(db_pq_tensor, 0, 0, len(self.database_vectors) * self.n_bits)
            # Multi-vectors encoding has bug now.
            # self.bmcv.faiss_indexPQ_encode(self.database_vectors, 
            #                                self.centroids_vectors_tensor,
            #                                self.database_vectors_pq_tensor,
            #                                len(self.database_vectors),
            #                                self.embedding_length,
            #                                self.slice_num,
            #                                self.centroids_num,
            #                                self.ip_metric)
        self.is_trained = True

    def similarity_search(self, query: str, top_k: int = 1, search_type=None):
        similarities, ids = self.similarity_search_by_queries([query], top_k, search_type)
        return similarities[0], ids[0]
    
    def similarity_search_by_queries(self, queries: list, top_k: int = 1, search_type=None):
        if self.is_trained == False:
            raise RuntimeError("Index is not trained yet!")
        if search_type == None:
            search_type = self.search_type
        if top_k > len(self.database_ids):
            top_k = len(self.database_ids)
        query_vecs = []
        for query in queries:
            query_vec = self.embedding.text2vec(query)
            query_vecs.append(query_vec)
        query_vecs = np.array(query_vecs)

        similarities, idxs = [], []
        if self.use_official_faiss:
            if search_type == SearchType.IndexPQADC:
                self.faiss_internal_index.search_type = faiss.IndexPQ.ST_PQ
                similarities, idxs = self.faiss_service.search(query_vecs, top_k)
            elif search_type == SearchType.IndexPQSDC:
                self.faiss_internal_index.search_type = faiss.IndexPQ.ST_SDC
                similarities, idxs = self.faiss_service.search(query_vecs, top_k)
            else:
                raise TypeError("Invalid search_type:{}".format(search_type))            
            return similarities, idxs
        else:
            if search_type == SearchType.IndexPQADC:
                similarities, idxs = self.bmcv.faiss_indexPQ_ADC(query_vecs,
                                                                 self.centroids_vectors_tensor,
                                                                 self.database_vectors_pq_tensor,
                                                                 self.embedding_length,
                                                                 self.slice_num,
                                                                 self.centroids_num,
                                                                 len(self.database_vectors),
                                                                 len(queries),
                                                                 top_k,
                                                                 self.ip_metric)
            elif search_type == SearchType.IndexPQSDC:
                # query_pq_tensor = self.bmcv.faiss_indexPQ_encode(query_vecs,
                #                                                  self.centroids_vectors_tensor,
                #                                                  len(queries),
                #                                                  self.embedding_length,
                #                                                  self.slice_num,
                #                                                  self.centroids_num,
                #                                                  self.ip_metric)
                query_pq = self.faiss_internal_index.pq.compute_codes(query_vecs)
                query_pq_tensor = sail.Tensor(self.handle, query_pq, False, True)
                similarities, idxs = self.bmcv.faiss_indexPQ_SDC(query_pq_tensor,
                                                                 self.database_vectors_pq_tensor,
                                                                 self.sdc_table_tensor,
                                                                 self.slice_num,
                                                                 self.centroids_num,
                                                                 len(self.database_vectors),
                                                                 len(queries),
                                                                 top_k,
                                                                 self.ip_metric)
            else:
                raise TypeError("Invalid search_type:{}".format(search_type))
            ids = []
            for idxs_ in idxs:
                ids.append([self.database_ids[int(i)] for i in idxs_])
            
            return similarities, ids                
