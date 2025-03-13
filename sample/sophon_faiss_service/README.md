# sophon_faiss_service
该功能仅支持BM1684X，可用于替换faiss或者milvus的similarity search接口。目前支持IndexFlat(IP/L2)、IndexPQ(ADC/SDC)两种检索方法。

用户可以直接调用本封装，也可以参考本封装进行二次开发。

# 环境准备：

pcie/soc都需要安装这些依赖库：
```bash
pip3 install dfss --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install setuptools --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install langchain_huggingface langchain_core langchain_milvus -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install scikit-learn sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple #indexpq计算聚类中心时会用到。
```

在pcie上，请参考[https://github.com/sophgo/sophon-demo/blob/release/docs/Environment_Install_Guide.md]配置对应平台的sophonsdk开发环境，即libsophon、sophon-mw、sophon-sail。
还需要做如下操作：
```bash
python3 -m dfss --url=open@sophgo.com:sophon-sail/sample/sophon_faiss_service/tpu_module_x86.tar.gz && tar xvf tpu_module_x86.tar.gz
sudo cp tpu_module_x86/libbm1684x_kernel_module.so /opt/sophon/libsophon-${x.y.z}/lib/tpu_module/ #${x.y.z}是libsophon的版本号，请根据具体情况填写。
```

在soc上，请先确保您的环境是官网v24.04.01版本或以上的版本，然后做如下操作：
```bash
python3 -m dfss --url=open@sophgo.com:sophon-sail/sample/sophon_faiss_service/sophon_arm-3.9.1-py3-none-any.whl
pip3 install sophon_arm-3.9.1-py3-none-any.whl --force-reinstall
python3 -m dfss --url=open@sophgo.com:sophon-sail/sample/sophon_faiss_service/tpu_module_aarch64.tar.gz && tar xvf tpu_module_aarch64.tar.gz
sudo cp tpu_module_aarch64/libbm1684x_kernel_module.so /opt/sophon/libsophon-${x.y.z}/lib/tpu_module/
```

在soc上还需要设置环境变量，以解决一个scikit_learn在aarch64上的bug：
```bash
export LD_PRELOAD=$LD_PRELOAD:~/.local/lib/python3.8/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0 #这一步只在当前终端生效
```

# 数据下载：
```bash
#向量长度768 embedding bmodel下载：
python3 -m dfss --url=open@sophgo.com:ezoo/chatdoc/bce-embedding-base_1b.bmodel
#向量长度768 token_config
python3 -m dfss --url=open@sophgo.com:ext_model_information/RAG/token_config.zip
unzip token_config.zip
#db下载
python3 -m dfss --url=open@sophgo.com:sophon-sail/sample/sophon_faiss_service/ready.db
#原始文本下载
python3 -m dfss --url=open@sophgo.com:sophon-sail/sample/sophon_faiss_service/ready.txt
#vectors/ids下载
python3 -m dfss --url=open@sophgo.com:sophon-sail/sample/sophon_faiss_service/db_vectors.npy
python3 -m dfss --url=open@sophgo.com:sophon-sail/sample/sophon_faiss_service/db_ids.npy
```

# 目录结构说明

下载完数据之后，目录结构如下：
```bash
├── bce-embedding-base_1b.bmodel #embedding model
├── db_ids.npy                   #一维数组，存储vector对应的id
├── db_vectors.npy               #二维数组，存储vector，shape为[vec_num, vec_dim]
├── langchain_milvus_test.py     #cpu/tpu search对比测试程序
├── README.md                    
├── ready.db                     #数据库
├── ready.txt                    #原始文本
├── sophon_arm-3.9.1-py3-none-any.whl #soc模式，封装了tpu faiss接口的sail安装包
├── sophon_faiss_service.py           #对sail的tpu faiss接口封装。
├── token_config                      #tokenizer
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── tpu_module_x86                #tpu算子库，x86版本
|    └── libbm1684x_kernel_module.so
└── tpu_module_aarch64                #tpu算子库，aarch64版本
    └── libbm1684x_kernel_module.so
```

# 运行：

sophon_faiss_service.py是我们的封装，langchain_milvus_test.py是langchain milvus原生接口和sophon_faiss_service封装的调用示例。
```bash
python3 langchain_milvus_test.py
```

# 性能测试

底库向量数50000+，向量维度768，top_k=5，性能测试结果如下：
| 测试平台 | 搜索方法                     |  单次搜索耗时(s)    |
| ---     |  ---                         |   ---------       |
| SE7-32  | sail.bmcv.faiss_indexflatIP |  0.0216           |
| SE7-32  | sail.bmcv.faiss_indexflatL2 |  0.106           |
| SE7-32  | sail.bmcv.faiss_indexPQ_ADC |  0.021           |
| SE7-32  | sail.bmcv.faiss_indexPQ_SDC |  0.020           |

## 6. 接口定义

```python
class IndexFlat:
    def __init__(self, embedding_bmodel_path, tokenizer_path, tpu_id=0, search_type=SearchType.IndexFlatIP):
        """
        功能：初始化
        参数解析：
        embedding_bmodel_path：词向量模型的路径。
        tokenizer_path：分词器的路径。
        tpu_id：设备号，soc只能为0。
        search_type：检索方式，如果初始化为IndexFlatIP，那么在search的时候只能使用IndexFlatIP；如果初始化为IndexFlatL2，那么search的时候可以用IndexFlatIP和IndexFlatL2。
        """

    def load_from_npy(self, vectors_file: str, ids_file: str):
        """
        功能：加载向量、索引库
        参数解析：
        vectors_file: 向量库的路径，只接受npy。        
        ids_file: 索引库的路径，只接受npy。
        注：每一个向量都对应一个索引，向量数和索引数必须相等，不能有重复索引。        
        """
    def save_npy(self, vectors_file_name: str, ids_file_name: str):
        """
        功能：将类中维护的向量和索引库保存为npy文件。
        参数解析：
        vectors_file_name：向量库的名字，不用带.npy的后缀。
        ids_file_name：索引库的名字，不用带.npy的后缀。
        """

    def renew_dev_mem(self, ):
        """
        功能：将系统内存的向量库更新到设备内存，用户一般用不到这个接口。
        """
    
    def add_text(self, text: str, id: int):
        """
        功能：往向量库里添加数据。
        参数解析：
        text：输入的文本，会经由词向量模型转换为向量，添加到向量库。
        id：文本对应的索引。
        """

    def delete_by_ids(self, ids):
        """
        功能：根据索引删除向量库中的数据。
        参数解析：
        ids：一个索引列表。
        """

    def similarity_search(self, query: str, top_k: int = 1, search_type=None):
        """
        功能：检索和要查找的输入文本最相似的向量，返回相似度列表、索引列表。
        参数解析：
        query：要查找的文本。
        top_k：要返回的索引个数，不可以比已有的索引个数多。
        search_type：如果不设置，默认和初始化的search_type相同，如果初始化用了IndexFlatL2，那么这里可以用IndexFlatIP和IndexFlatL2。
        """

    def similarity_search_by_queries(self, queries: list, top_k: int = 1, search_type=None):
        """
        功能：检索和要查找的输入文本最相似的向量列表，返回相似度二维列表、索引二维列表，多query的搜索效率比单query更高。
        参数解析：
        queries：要查找的文本列表。
        top_k：要返回的索引个数，不可以比已有的索引个数多。
        search_type：如果不设置，默认和初始化的search_type相同，如果初始化用了IndexFlatL2，那么这里可以用IndexFlatIP和IndexFlatL2。
        """

class IndexPQ:
    def __init__(self, embedding_bmodel_path, tokenizer_path, tpu_id=0, search_type=SearchType.IndexPQADC, slice_num=8, n_bits=8, metric=faiss.METRIC_L2):
        """
        功能：初始化
        参数解析：
        embedding_bmodel_path：词向量模型的路径。
        tokenizer_path：分词器的路径。
        tpu_id：设备号，soc只能为0。
        search_type：检索方式，支持SearchType.IndexPQADC、SearchType.IndexPQSDC，如果后续search的时候没有指定search_type，则使用初始化的search_type。
        slice_num：向量切分的子空间个数。
        n_bits：量化编码的位数。
        metric：计算sdc_table或检索时使用的距离类型，支持faiss.METRIC_L2、faiss.METRIC_INNER_PRODUCT。
        """
    def load_from_npy(self, vectors_file: str, ids_file: str):
        """
        同IndexFlat.load_from_npy
        """
    
    def save_npy(self, vectors_file_name: str, ids_file_name: str):
        """
        同IndexFlat.save_npy
        """

    def add_text(self, text: str, id: int):
        """
        同IndexFlat.add_text
        """
        
    def delete_by_ids(self, ids):
        """
        同IndexFlat.delete_by_ids
        """

    def train(self):
        """
        功能：训练量化中心向量库，load完或者add完之后，应当执行一次train，否则无法检索。
        """

    def similarity_search(self, query: str, top_k: int = 1, search_type=None):
        """
        功能：检索和要查找的输入文本最相似的向量，返回相似度列表、索引列表。
        参数解析：
        query：要查找的文本。
        top_k：要返回的索引个数，不可以比已有的索引个数多。
        search_type：如果不设置，默认和初始化的search_type相同，这里可以用IndexPQADC和IndexPQSDC。
        """

    def similarity_search_by_queries(self, queries: list, top_k: int = 1, search_type=None):
        """
        功能：检索和要查找的输入文本最相似的向量列表，返回相似度二维列表、索引二维列表，多query的搜索效率比单query更高。
        参数解析：
        queries：要查找的文本列表。
        top_k：要返回的索引个数，不可以比已有的索引个数多。
        search_type：如果不设置，默认和初始化的search_type相同，这里可以用IndexPQADC和IndexPQSDC。
        """
```
