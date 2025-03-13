import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from sophon_faiss_service import IndexFlat,IndexPQ,SearchType
import time

#768
embedding_model_path = "bce-embedding-base_1b.bmodel"
tokenizer_path = "token_config"
db_vectors_path = "db_vectors.npy"

indexflatl2 = IndexFlat(embedding_model_path,tokenizer_path,0,SearchType.IndexFlatL2)
indexpq = IndexPQ(embedding_model_path,tokenizer_path,0)
embeddings = HuggingFaceEmbeddings(model_name="maidalun1020/bce-embedding-base_v1", model_kwargs={'device': 'cpu'})
URI = "ready.db"
load_flag = False
if not os.path.exists(URI):
    load_flag = True
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    auto_id=True
)

if load_flag:
    from langchain_core.documents import Document
    from langchain_community.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(file_path='ready.txt')
    documents = loader.load()
    ids = vector_store.add_documents(documents=documents)
    for id, doc in zip(ids, documents):
        indexflatl2.add_text(doc.page_content, id)
    indexflatl2.save_npy('db_vectors','db_ids')
else:
    indexflatl2.load_from_npy(db_vectors_path, "db_ids.npy")
    indexpq.load_from_npy(db_vectors_path, "db_ids.npy")
    indexpq.train()
print("----milvus.similarity_search----")
start = time.time()
results = vector_store.similarity_search(
    "哈利波特猛然睡醒",
    k=5,
)
duration = time.time() - start
print("cpu search1 time(s): ", duration)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

start = time.time()
results = vector_store.similarity_search(
    "哈利波特与魔法石",
    k=5,
)
duration = time.time() - start
print("cpu search2 time(s): ", duration)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

start = time.time()
results = vector_store.similarity_search(
    "霍格沃茨魔法学校的校长",
    k=5,
)
duration = time.time() - start
print("cpu search3 time(s): ", duration)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
print("-------------------------------")

print("---------IndexFlatIP-----------")
start = time.time()
sim, sids = indexflatl2.similarity_search("哈利波特猛然睡醒", 5, search_type=SearchType.IndexFlatIP)
duration = time.time() - start
print("tpu search time(s): ", duration)

start = time.time()
results_ = vector_store.col.query(expr=f'pk in {[int(_id) for _id in sids]}', output_fields=["*"])
for res in results_:
    print(res['text'], res['source'], res['pk'])
duration = time.time() - start
print("query time(s): ", duration)
print("-------------------------------")


print("---------IndexFlatL2-----------")
start = time.time()
sim, sids = indexflatl2.similarity_search("哈利波特猛然睡醒", 5, search_type=SearchType.IndexFlatL2)
print(sids)
duration = time.time() - start
print("tpu search time(s): ", duration)

start = time.time()
results_ = vector_store.col.query(expr=f'pk in {[int(_id) for _id in sids]}', output_fields=["*"])

for res in results_:
    print(res['text'], res['source'], res['pk'])
duration = time.time() - start
print("query time(s): ", duration)
print("------------------------------")

print("---------IndexPQADC-----------")
start = time.time()
sim, sids = indexpq.similarity_search("哈利波特猛然睡醒", 5, search_type=SearchType.IndexPQADC)
print(sim, sids)
duration = time.time() - start
print("tpu search time(s): ", duration)

start = time.time()
results_ = vector_store.col.query(expr=f'pk in {[int(_id) for _id in sids]}', output_fields=["*"])
for res in results_:
    print(res['text'], res['source'], res['pk'])
duration = time.time() - start
print("query time(s): ", duration)
print("-------------------------------")


print("---------IndexPQSDC-----------")
start = time.time()
sim, sids = indexpq.similarity_search("哈利波特猛然睡醒", 5, search_type=SearchType.IndexPQSDC)
print(sim, sids)
duration = time.time() - start
print("tpu search time(s): ", duration)

start = time.time()
results_ = vector_store.col.query(expr=f'pk in {[int(_id) for _id in sids]}', output_fields=["*"])

for res in results_:
    print(res['text'], res['source'], res['pk'])
duration = time.time() - start
print("query time(s): ", duration)
print("------------------------------")