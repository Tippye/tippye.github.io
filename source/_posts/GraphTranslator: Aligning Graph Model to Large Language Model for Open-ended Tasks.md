---
title: GraphTranslator\: Aligning Graph Model to Large Language Model for Open-ended Tasks
date: 2024-06-1T12:09:57+08:00
categories: 论文精读
tags:
    - 复杂网络
    - NLP

category_bar: true
math: true
mermaid: true
lazyload: true
---

- # GraphTranslator: Aligning Graph Model to Large Language Model for Open-ended Tasks


<center><a href='https://github.com/alibaba/GraphTranslator'><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" /></a><a href='https://arxiv.org/abs/2402.07197v4'><img src="https://img.shields.io/badge/arxiv-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=ffffff" referrerpolicy="no-referrer"></a><a href='https://mp.weixin.qq.com/s/qqbeiac-p5tKto4YSYSiNw'><img src="https://img.shields.io/badge/WeChat-07c168?style=for-the-badge&amp;logo=wechat&amp;logoColor=ffffff" referrerpolicy="no-referrer"></a></center>

## 论文要做什么

通过LLM(大语言模型)与GM(图模型)相结合，实现一个既能解决**预定义任务**又能解决**开放式任务**的模型。

实验结果表明，该方法在零样本分类等任务中表现出色，并具有广泛的应用前景。

## 论文方法

本文主要方法是将LLM与GM相结合。共包括四个模块：**冻结的GM**、**冻结的LLM**、**Producer**、**Translator**。

四个模块作用如下：

- 冻结的GM：
  - 编码节点嵌入信息
  - Producer阶段，通过图模型找到节点的邻居节点
- 冻结的LLM：
  - 提供自然语言描述能力
  - 生成节点嵌入和文本描述的配对数据
- Producer：
  - 利用LLM生成节点嵌入和文本描述的配对数据，并文本化节点信息
- Translator：
  - 将图节点嵌入转换为LLM可理解的token嵌入
  - 学习graph queries来提取语言信息并适配LLM

训练后，带有Translator的LLM能处理各种开放式任务

![model](https://cdn.jsdelivr.net/gh/alibaba/GraphTranslator@main/figure/model.jpg)

## 实现

### 环境

论文作者使用的配置如下：

- CPU: Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz
- GPU: Tesla V100-SXM2-32GB
- OS: Linux (Ubuntu 18.04.6 LTS)
- Python\=\=3.9, CUDA\=\=11.4, Pytorch\=\=1.12.1

我论文复现时使用的配置如下：

- CPU: 15 vCPU AMD EPYC 9754 128-Core Processor
- GPU: RTX 4090D(24GB) * 1
- 镜像: Miniconda conda3 Python 3.10(ubuntu22.04) Cuda 11.8
- 来自AutoDL的4090D实例

> 实际运行中显存在16G左右，所以想复现程序可能需要显存在16G以上的显卡。
>
> 由于个别库在Windows安装失败，所以需要Ubuntu环境。
>
> CPU和mps执行会在加载`ChatGLM2-6B`模型时报错。
>
> 4090D按速度估计可能需要半个月可以执行完Producer模块

### Producer

```shell
cd ./Producer/inference
python producer.py
```



这部分代码主要执行下面三行：

```python
# 加载大语言模型
model = LLM(args)
# 读数据集
arxiv_data, sample_neighbor_df = read_arxiv_dataset()
# 用LLM推理得到高质量对齐数据
model.inference_chatglm_arxiv(arxiv_data, sample_neighbor_df)
```

其中LLM类模型代码如下：

```python
class LLM(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self._args = args
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True)
        # model
        torch.cuda.empty_cache() # 这里显存溢出了
        try:
            self.llm = AutoModel.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True).half().to(device)
        except RuntimeError:
            self.llm = AutoModel.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True).float().to(torch.device('cpu'))

    def inference_chatglm_arxiv(self, arxiv_data, sample_neighbor_df):
        """
        遍历arxiv_data的每一个节点
            对于每一个节点使用LLM生成他的五个关键词和对所有邻居节点主题和内容的总结
        将这些数据和id等信息存入summary_embeddings.csv

        Args:
            arxiv_data:
            sample_neighbor_df:

        Returns:

        """
        self.llm.eval()

        # {0: "Title: evasion... Abstract: In security-sensitive...",1: "Title: ..."}
        node_title_and_abs = arxiv_data.set_index('node_id')['title_abstract'].to_dict()

        # value是key的邻居节点列表
        # src_to_dst_dict = {0: [74997, 81796, 86748, 122809, 163274], 1: ...}
        src_to_dst_dict = sample_neighbor_df.groupby('src_node')['dst_node'].apply(list).to_dict()

        # node2title = {0: 'evasion attacks against machine learning at test time',1: '...'}
        node2title = arxiv_data.set_index('node_id')['title'].to_dict()

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} total paper count: {arxiv_data.shape[0]}")
        summary = []
        for data in arxiv_data.iterrows():
            """
            对于每个节点，生成一句‘The title and abstract of this paper are as follows: Title:{论文标题}, Abstract: {论文摘要}\n please summarize this paper and list five key words of this paper. All answers are in English and No Chinese in your answer'
            用这句话询问LLM，可以得到五个关键词
            """
            node_id = data[1]['node_id']
            title = data[1]['title']
            src_prompt_pre = "The title and abstract of this paper are as follows: "
            src_prompt = '\n please summarize this paper and list five key words of this paper. All answers are in English and No Chinese in your answer'
            src_title_abstract = data[1]['title_abstract']
            node_word_input = src_prompt_pre + src_title_abstract
            if len(node_word_input[0]) > 3000- len(src_prompt):
                node_word_input = node_word_input[:3000-len(src_prompt)]
            node_word_input += src_prompt

            """
            找到节点的邻居节点，生成一句‘\n The title and abstract of this paper are as follows: Title:{论文标题}, Abstract: {论文摘要}\nTitle:... \n Please summarize the topic and content of these papers. All answers are in English and No Chinese in your answer'
            用这句话询问LLM，可以得到对所有邻居节点主题和内容的总结
            """
            dst_prompt_pre = '\n The paper title and abstract are provided as follows: '
            dst_prompt = "\n Please summarize the topic and content of these papers. All answers are in English and No Chinese in your answer"
            dst_title_abstract = ""
            for neighbor_id in src_to_dst_dict[node_id]:
                dst_title_abstract = dst_title_abstract + node_title_and_abs[neighbor_id] + '\n'

            neighbor_word_input  = dst_prompt_pre + dst_title_abstract
            if len(neighbor_word_input[0]) > 3000-len(dst_prompt):
                neighbor_word_input = neighbor_word_input[:3000-len(dst_prompt)]
            neighbor_word_input += dst_prompt

            try:
                # 从标题和摘要生成5个关键词
                response_node, _ = self.llm.chat(self.tokenizer,
                                                        node_word_input ,
                                                        history=[])
                response_neighbor, _ = self.llm.chat(self.tokenizer,
                                                            neighbor_word_input,
                                                            history=[])
                summary.append({
                    'node_id': node_id,
                    'title': title,
                    'response_node': response_node,
                    'response_neighbor': response_neighbor
                })
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} paper {node_id+1} title: \"{title}\"")
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("CUDA out of memory error detected, skipping this batch")
                    continue
                else:
                    continue

        """
        将前面所获得的所有回答整理格式存入summary_embeddings.csv
        """
        summary_df = pd.DataFrame(summary)
        embeddings = torch.load("../../data/arxiv/graphsage_node_embeddings.pt").to('cpu')
        new_data = []
        for _, row in summary_df.iterrows():
            node_id = int(row['node_id'])
            embedding = np.array(embeddings[node_id].detach())
            str_array = [str(num) for num in embedding]
            str_representation = ", ".join(str_array)
            title = node2title[row['node_id']]

            new_data.append({
                'node_id': node_id,
                'embedding':str_representation,
                'paper_summary':row['response_node'],
                'citepapers_summary':row['response_neighbor'],
                'title':title
                })
        summary_embeddings = pd.DataFrame(new_data)
        summary_embeddings.to_csv('../../data/arxiv/summary_embeddings.csv',index=False)
```

生成的结果如下图![image-20240529105119919](https://cdn.jsdelivr.net/gh/tippye/PicCloud@master/uPic/2024/05/29/image-20240529105119919.png)

## Translator-Train

训练共需要两个阶段

#### stage1：训练Translator进行GM和文本的对齐。

```shell
cd ./Translator/train
python train.py --cfg-path ./pretrain_arxiv_stage1.yaml
```

```python
def main(job_id):
    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)

    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg,
        job_id=job_id,
        task=task,
        model=model,
        datasets=datasets
    )
    runner.train()
```

执行时，这里的`task`主要用于多线程，我们需要关注的主要是`runner.train()`过程

- 第一阶段使用的数据集为`producer`生成的`summary_embeddings.csv`文件[<img src="https://github.githubassets.com/favicons/favicon.svg" style="zoom:50%;" />](https://github.com/alibaba/GraphTranslator/blob/main/data/arxiv/summary_embeddings.csv)

- 训练使用的优化器为`AdamW`[<img src="https://github.githubassets.com/favicons/favicon.svg" style="zoom:50%;" />](https://github.com/alibaba/GraphTranslator/blob/main/Translator/runners/runner_base.py#L91)
  - 这一步会将参数分为有权重衰减和无权重衰减的两种参数
    - 有权重衰减的参数包括维度大于等于2的和名称中包含`bias`或`ln`或`bn`的参数
    - 衰减权重配置为`0.05`
    - 无衰减的参数衰减权重配置为`0`
  - 学习率配置为`1e-4`
  - Adam算法中的第一阶矩估计和第二阶矩估计的指数衰减率设置为固定值`0.9`和`0.999`

- 这里的`runner`追溯代码可以发现用的是默认的类`runner_base`[<img src="https://github.githubassets.com/favicons/favicon.svg" style="zoom:50%;" />](https://github.com/alibaba/GraphTranslator/blob/main/Translator/runners/runner_base.py)。

- 最终追溯到训练每个epoch的代码为`base_task._train_inner_loop()`[<img src="https://github.githubassets.com/favicons/favicon.svg" style="zoom:50%;" />](https://github.com/alibaba/GraphTranslator/blob/main/Translator/tasks/base_task.py#L149)

  - 使用的预训练模型为`TranslatorQformerArxiv`，基于`Q-Former`,使用`bert-base-uncased`与训练参数进行初始化

    - 特征数为768
    - 最大文本长度512
    - Qformer查询令牌数量为32
    - Qformer交叉注意力频率为2

  - 模型的前向传播过程[<img src="https://github.githubassets.com/favicons/favicon.svg" style="zoom:50%;" />](https://github.com/alibaba/GraphTranslator/blob/main/Translator/models/translator_models/translator_qformer_arxiv.py#L90)

    - 对比目标(图文对比)[<img src="https://github.githubassets.com/favicons/favicon.svg" style="zoom:50%;" />](https://github.com/alibaba/GraphTranslator/blob/main/Translator/models/translator_models/translator_qformer_arxiv.py#L127)

      - > 通过计算$H_v$与$T_v$​之间的相似度，并选择最高得分来最大化二者的互信息

      - 分别将图像文本特征合并，使用`torch.matmul`分别计算得到图像与文本之间的特征相似度`sim_q2t`和文本与图像之间的特征相似度`sim_t2q`

      - 分别取最大值然后进行归一化得到`sim_i2t`和`sim_t2i`

    - 生成目标

      - > 根据节点嵌入$z_v$生成描述文本，并优化生成文本与实际节点描述之间的交叉熵损失，使得query tokens $Q$能捕获更多与$t_v$相关的细节。

      - 最后使用交叉熵损失函数分别计算`sim_i2t`和`sim_t2i`与`targets`之间的损失，取二者均值作为损失

    - 图文匹配[<img src="https://github.githubassets.com/favicons/favicon.svg" style="zoom:50%;" />](https://github.com/alibaba/GraphTranslator/blob/main/Translator/models/translator_models/translator_qformer_arxiv.py#L162)

      - 使用`softmax`分别计算图像到文本的相似度`weights_i2t`和文本到图像的相似度`weights_t2i`
      - 对于每个文本，取一个负样本图像放在`behavior_embeds_neg`；对于每个图像，取一个负样本文本放在`text_atts_neg`
      - 将正样本和负样本的文本ID、注意力掩码以及行为嵌入进行堆叠和拼接，形成`text_ids_all`, `text_atts_all`, `behavior_embeds_all`
      - 扩展查询令牌`query_tokens_itm`并创建对应的注意力掩码`query_atts_itm`
      - 使用`bert`模型进行处理得到`vl_embeddings`
      - 使用交叉熵函数计算损失`loss_itm`，这表示模型在预测匹配度时的错误率，是训练中需要最小化的量。

    - 图像描述[<img src="https://github.githubassets.com/favicons/favicon.svg" style="zoom:50%;" />](https://github.com/alibaba/GraphTranslator/blob/main/Translator/models/translator_models/translator_qformer_arxiv.py#L230)

      - 使用`bert`模型计算语言模型的损失

    - 返回前三步计算得到的损失值，将三种损失值的和作为损失

  - 每次训练中

  - 使用AMP(自动混合精度)进行前向传播和计算损失

  - 每隔32次迭代更新一次参数

  - 在所有进程之间通过`MetricLogger`同步数据，输出平均值

  - 返回平均后的损失和学习率

  - > 细粒度对齐通过将每个节点嵌入$h_{v,i}$与[CLS]标记$\overline t_v$​拼接并输入二分类器来学习，计算所有查询的logits平均值作为匹配得分。

- 第二阶段
