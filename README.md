## Must-read papers on NRL/NE.
NRL: network representation learning. NE: network embedding.

Contributed by [Cunchao Tu](http://thunlp.org/~tcc/), Yuan Yao, Zhengyan Zhang, GanquCui, Hao Wang (BUPT), Changxin Tian (BUPT), Jie Zhou and Cheng Yang (BUPT).

We release [OpenNE](https://github.com/thunlp/openne), an open source toolkit for NE/NRL. This repository provides a standard NE/NRL(Network Representation Learning）training and testing framework. Currently, the implemented models in OpenNE include DeepWalk, LINE, node2vec, GraRep, TADW and GCN.

### Content
1. [Survey Papers](#survey-papers)
2. [Models](#models)
    1. [Bacis Models](#bacis-models)
    1. [Attributed Network](#attributed-network)
    1. [Dynamic Network](#dynamic-network)
    1. [Heterogeneous Information Network](#heterogeneous-information-network)
    1. [Bipartite Network](#bipartite-network)
    1. [Directed Network](#directed-network)
    1. [Other Models](#other-models)
3. [Applications](#applications)
    1. [Natural Language Processing](#natural-language-processing)
    1. [Knowledge Graph](#knowledge-graph)
    1. [Social Network](#social-network)
    1. [Graph Clustering](#graph-clustering)
    1. [Community Detection](#community-detection)
    1. [Recommendation](#recommendation)
    1. [Other Applications](#other-applications)

### [Survey Papers](#content)

1. **Representation Learning on Graphs: Methods and Applications.**
*William L. Hamilton, Rex Ying, Jure Leskovec.* IEEE Data(base) Engineering Bulletin 2017. [paper](https://arxiv.org/pdf/1709.05584.pdf)

1. **Graph Embedding Techniques, Applications, and Performance: A Survey.**
*Palash Goyal, Emilio Ferrara.* Knowledge Based Systems 2017. [paper](https://arxiv.org/pdf/1705.02801.pdf)

1. **A Comprehensive Survey of Graph Embedding: Problems, Techniques and Applications.**
*Hongyun Cai, Vincent W. Zheng, Kevin Chen-Chuan Chang.* TKDE 2017. [paper](https://arxiv.org/pdf/1709.07604.pdf)

1. **Network Representation Learning: A Survey.**
*Daokun Zhang, Jie Yin, Xingquan Zhu, Chengqi Zhang.* IEEE Transactions on Big Data 2018. [paper](https://arxiv.org/pdf/1801.05852.pdf)

1. **A Tutorial on Network Embeddings.**
*Haochen Chen, Bryan Perozzi, Rami Al-Rfou, Steven Skiena.* arxiv 2018. [paper](https://arxiv.org/pdf/1808.02590.pdf)

1. **Network Representation Learning: An Overview.(In Chinese)**
*Cunchao Tu, Cheng Yang, Zhiyuan Liu, Maosong Sun.* 2017. [paper](http://engine.scichina.com/publisher/scp/journal/SSI/47/8/10.1360/N112017-00145)

1. **Relational inductive biases, deep learning, and graph networks.**
*Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulcehre, Francis Song, Andrew Ballard, Justin Gilmer, George Dahl, Ashish Vaswani, Kelsey Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matt Botvinick, Oriol Vinyals, Yujia Li, Razvan Pascanu.* arxiv 2018. [paper](https://arxiv.org/pdf/1806.01261.pdf)

### [Models](#content)

#### [Bacis Models](#content)

1. **SepNE: Bringing Separability to Network Embedding.**
*Ziyao Li, Liang Zhang, Guojie Song.* AAAI 2019. [paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4333)

1. **Robust Negative Sampling for Network Embedding.**
*Mohammadreza Armandpour, Patrick Ding, Jianhua Huang, Xia Hu.* AAAI 2019. [paper](https://www.stat.tamu.edu/~armand/R-NS.pdf)

1. **Network Structure and Transfer Behaviors Embedding via Deep Prediction Model.**
*Xin Sun, Zenghui Song, Junyu Dong, Yongbo Yu, Claudia Plant, Christian Böhm.* AAAI 2019. [paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/4436)

1. **Simplifying Graph Convolutional Networks.**
*Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, Kilian Weinberger.* ICML 2019. [paper](http://proceedings.mlr.press/v97/wu19e/wu19e.pdf)

1. **GMNN: Graph Markov Neural Networks.**
*Meng Qu, Yoshua Bengio, Jian Tang.* ICML 2019. [paper](http://proceedings.mlr.press/v97/qu19a/qu19a.pdf)

1. **Stochastic Blockmodels meet Graph Neural Networks.**
*Nikhil Mehta, Lawrence Carin Duke, Piyush Rai.* ICML 2019. [paper](http://proceedings.mlr.press/v97/mehta19a/mehta19a.pdf)

1. **Disentangled Graph Convolutional Networks.**
*Jianxin Ma, Peng Cui, Kun Kuang, Xin Wang, Wenwu Zhu.* ICML 2019. [paper](http://proceedings.mlr.press/v97/ma19a/ma19a.pdf)

1. **Position-aware Graph Neural Networks.**
*Jiaxuan You, Rex Ying, Jure Leskovec.* ICML 2019. [paper](http://proceedings.mlr.press/v97/you19b/you19b.pdf)

1. **MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.**
*Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Kristina Lerman, Hrayr Harutyunyan, Greg Ver Steeg, Aram Galstyan.* ICML 2019. [paper](http://proceedings.mlr.press/v97/abu-el-haija19a/abu-el-haija19a.pdf)

1. **Graph U-Nets.**
*Hongyang Gao, Shuiwang Ji.* ICML 2019. [paper](http://proceedings.mlr.press/v97/gao19a/gao19a.pdf)

1. **Self-Attention Graph Pooling.**
*Junhyun Lee, Inyeop Lee, Jaewoo Kang.* ICML 2019. [paper](http://proceedings.mlr.press/v97/lee19c/lee19c.pdf)

1. **Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via Ranking.**
*Aleksandar Bojchevski, Stephan Günnemann.* ICLR 2018. [paper](https://arxiv.org/pdf/1707.03815.pdf)

1. **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling.**
*Jie Chen, Tengfei Ma, Cao Xiao.* ICLR 2018. [paper](https://arxiv.org/pdf/1801.10247.pdf)

1. **Graph Attention Networks.**
*Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio.* ICLR 2018. [paper](https://arxiv.org/pdf/1710.10903.pdf)

1. **Stochastic Training of Graph Convolutional Networks with Variance Reduction.**
*Jianfei Chen, Jun Zhu, Le Song.* ICML 2018. [paper](https://arxiv.org/pdf/1710.10568.pdf)

1. **Adversarially Regularized Graph Autoencoder for Graph Embedding.**
*Shirui Pan, Ruiqi Hu, Guodong Long, Jing Jiang, Lina Yao, Chengqi Zhang.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0362.pdf)

1. **Discrete Network Embedding.**
*Xiaobo Shen, Shirui Pan, Weiwei Liu, Yew-Soon Ong, Quan-Sen Sun.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0493.pdf)

1. **Feature Hashing for Network Representation Learning.**
*Qixiang Wang, Shanfeng Wang, Maoguo Gong, Yue Wu.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0390.pdf)

1. **Deep Inductive Network Representation Learning.**
*Ryan A. Rossi, Rong Zhou, Nesreen K. Ahmed.* WWW 2018. [paper](http://ryanrossi.com/pubs/rossi-et-al-WWW18-BigNet.pdf)

1. **Active Discriminative Network Representation Learning.**
*Li Gao, Hong Yang, Chuan Zhou, Jia Wu, Shirui Pan, Yue Hu.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0296.pdf)

1. **MILE: A Multi-Level Framework for Scalable Graph Embedding.**
*Jiongqian Liang, Saket Gurukar, Srinivasan Parthasarathy.* arxiv 2018. [paper](https://arxiv.org/pdf/1802.09612.pdf)

1. **Out-of-sample extension of graph adjacency spectral embedding.**
*Keith Levin, Farbod Roosta-Khorasani, Michael W. Mahoney, Carey E. Priebe.* ICML 2018. [paper](https://arxiv.org/pdf/1802.06307.pdf)

1. **DeepWalk: Online Learning of Social Representations.**
*Bryan Perozzi, Rami Al-Rfou, Steven Skiena.* KDD 2014. [paper](https://arxiv.org/pdf/1403.6652) [code](https://github.com/phanein/deepwalk)

1. **Non-transitive Hashing with Latent Similarity Componets.**
*Mingdong Ou, Peng Cui, Fei Wang, Jun Wang, Wenwu Zhu.* KDD 2015. [paper](http://cuip.thumedialab.com/papers/KDD-NonTransitiveHashing.pdf)

1. **GraRep: Learning Graph Representations with Global Structural Information.**
*Shaosheng Cao, Wei Lu, Qiongkai Xu.* CIKM 2015. [paper](https://www.researchgate.net/profile/Qiongkai_Xu/publication/301417811_GraRep/links/5847ecdb08ae8e63e633b5f2/GraRep.pdf) [code](https://github.com/ShelsonCao/GraRep)

1. **LINE: Large-scale Information Network Embedding.**
*Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Qiaozhu Me.* WWW 2015. [paper](https://arxiv.org/pdf/1503.03578.pdf) [code](https://github.com/tangjianpku/LINE)

1. **Deep Neural Networks for Learning Graph Representations.**
*Shaosheng Cao, Wei Lu, Xiongkai Xu.* AAAI 2016. [paper](https://pdfs.semanticscholar.org/1a37/f07606d60df365d74752857e8ce909f700b3.pdf) [code](https://github.com/ShelsonCao/DNGR)

1. **Revisiting Semi-supervised Learning with Graph Embeddings.**
*Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov.* ICML 2016. [paper](http://www.jmlr.org/proceedings/papers/v48/yanga16.pdf)

1. **Max-Margin DeepWalk: Discriminative Learning of Network Representation.**
*Cunchao Tu, Weicheng Zhang, Zhiyuan Liu, Maosong Sun.* IJCAI 2016. [paper](http://thunlp.org/~tcc/publications/ijcai2016_mmdw.pdf) [code](https://github.com/thunlp/mmdw)

1. **Discriminative Deep RandomWalk for Network Classification.**
*Juzheng Li, Jun Zhu, Bo Zhang.* ACL 2016. [paper](http://www.aclweb.org/anthology/P16-1095)

1. **Structural Deep Network Embedding.**
*Daixin Wang, Peng Cui, Wenwu Zhu.* KDD 2016. [paper](http://cuip.thumedialab.com/papers/SDNE.pdf)

1. **Structural Neighborhood Based Classification of Nodes in a Network.**
*Sharad Nandanwar, M. N. Murty.* KDD 2016. [paper](http://www.kdd.org/kdd2016/papers/files/Paper_679.pdf)

1. **Community Preserving Network Embedding.**
*Xiao Wang, Peng Cui, Jing Wang, Jian Pei, Wenwu Zhu, Shiqiang Yang.* AAAI 2017. [paper](http://cuip.thumedialab.com/papers/NE-Community.pdf)

1. **Semi-supervised Classification with Graph Convolutional Networks.**
*Thomas N. Kipf, Max Welling.* ICLR 2017. [paper](https://arxiv.org/pdf/1609.02907.pdf) [code](https://github.com/tkipf/gcn)

1. **Fast Network Embedding Enhancement via High Order Proximity Approximation.**
*Cheng Yang, Maosong Sun, Zhiyuan Liu, Cunchao Tu.* IJCAI 2017. [paper](http://thunlp.org/~tcc/publications/ijcai2017_neu.pdf) [code](https://github.com/thunlp/neu)

1. **CANE: Context-Aware Network Embedding for Relation Modeling.**
*Cunchao Tu, Han Liu, Zhiyuan Liu, Maosong Sun.* ACL 2017. [paper](http://thunlp.org/~tcc/publications/acl2017_cane.pdf) [code](https://github.com/thunlp/cane)

1. **A General View for Network Embedding as Matrix Factorization.**
*Xin Liu, Tsuyoshi Murata, Kyoung-Sook Kim, Chatchawan Kotarasu, Chenyi Zhuang.* WSDM 2019. [paper](https://dl.acm.org/citation.cfm?doid=3289600.3291029)

1. **Co-Embedding Attributed Networks.**
*Zaiqiao Meng, Shangsong Liang, Xiangliang Zhang, Hongyan Bao.* WSDM 2019. [paper](https://mine.kaust.edu.sa/Documents/papers/WSDM19attribute.pdf)

1. **Enhanced Network Embeddings via Exploiting Edge Labels.**
*Haochen Chen, Xiaofei Sun, Yingtao Tian, Bryan Perozzi, Muhao Chen, Steven Skiena.* CIKM 2018. [paper](https://arxiv.org/pdf/1809.05124.pdf)

1. **Improve Network Embeddings with Regularization.**
*Yi Zhang, Jianguo Lu, Ofer Shai.* CIKM 2018. [paper](https://jlu.myweb.cs.uwindsor.ca/n2v.pdf)

1. **Modeling Multi-way Relations with Hypergraph Embedding.**
*Chia-An Yu, Ching-Lun Tai, Tak-Shing Chan, Yi-Hsuan Yang.* CIKM 2018. [paper](https://dl.acm.org/citation.cfm?id=3269274) [code](https://github.com/chia-an/HGE)

1. **REGAL: Representation Learning-based Graph Alignment.**
*Mark Heimann, Haoming Shen, Tara Safavi, Danai Koutra.* CIKM 2018. [paper](https://arxiv.org/pdf/1802.06257.pdf)

1. **Adversarial Network Embedding.**
*Quanyu Dai, Qiang Li, Jian Tang, Dan Wang.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.07838.pdf) [code](https://github.com/sachinbiradar9/Adverserial-Inductive-Deep-Walk)

1. **Bernoulli Embeddings for Graphs.**
*Vinith Misra, Sumit Bhatia.* AAAI 2018. [paper](http://sumitbhatia.net/papers/aaai18.pdf)

1. **GraphGAN: Graph Representation Learning with Generative Adversarial Nets.**
*Hongwei Wang, jia Wang, jialin Wang, MIAO ZHAO, Weinan Zhang, Fuzheng Zhang, Xie Xing, Minyi Guo.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.08267.pdf)

1. **HARP: Hierarchical Representation Learning for Networks.**
*Haochen Chen, Bryan Perozzi, Yifan Hu, Steven Skiena.* AAAI 2018. [paper](https://arxiv.org/pdf/1706.07845.pdf) [code](https://github.com/GTmac/HARP)

1. **Social Rank Regulated Large-scale Network Embedding.**
*Yupeng Gu, Yizhou Sun, Yanen Li, Yang Yang.* WWW 2018. [paper](http://yangy.org/works/ge/rare.pdf)

1. **Latent Network Summarization: Bridging Network Embedding and Summarization.**
*Di Jin,Ryan Rossi,Danai Koutra,Eunyee Koh,Sungchul Kim,Anup Rao* KDD 2019. [paper](https://arxiv.org/pdf/1811.04461.pdf)

1. **NodeSketch: Highly-Efficient Graph Embeddings via Recursive Sketching.**
*Dingqi Yang,Paolo Rosso,Bin Li,Philippe Cudre-Mauroux.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330951/p1162-yang.pdf)

1. **ProGAN: Network Embedding via Proximity Generative Adversarial Network.**
*Hongchang Gao,Jian Pei,Heng Huang.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330866/p1308-gao.pdf)

1. **Scalable Global Alignment Graph Kernel Using Random Features: From Node Embedding to Graph Embedding.**
*Lingfei Wu,Ian En-Hsu Yen,Zhen Zhang,Kun Xu,Liang Zhao,Xi Peng,Yinglong Xia,Charu Aggarwal.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330918/p1418-wu.pdf)

1. **Scalable Graph Embeddings via Sparse Transpose Proximities.**
*Yuan Yin,Zhewei Wei.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330860/p1429-yin.pdf)

1. **AutoNRL: Hyperparameter Optimization for Massive Network Representation Learning.**
*Ke Tu,Jianxin Ma,Peng Cui,Jian Pei,Wenwu Zhu.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330848/p216-tu.pdf)

1. **Graph Representation Learning via Hard and Channel-Wise Attention Networks.**
*Hongyang Gao,Shuiwang Ji.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330897/p741-gao.pdf)

1. **Adversarial Training Methods for Network Embedding.**
*Quanyu Dai,Xiao Shen,Liang Zhang,Qiang Li,Dan Wang.* WWW 2019. [paper](https://arxiv.org/pdf/1908.11514.pdf)

1. **Multi-relational Network Embeddings with Relational Proximity and Node Attributes.**
*Ming-Han Feng,Chin-Chi Hsu,Cheng-Te Li,Mi-Yen Yeh,Shou-De Lin.* WWW 2019. [paper](https://pdfs.semanticscholar.org/6274/3cbebc142897c6c005f3c12c00b9202ca43f.pdf?_ga=2.108748866.1527570260.1569422306-1231101604.1568798295)

1. **Sampled in Pairs and Driven by Text: A New Graph Embedding Framework.**
*Liheng Chen,Yanru Qu,Zhenghui Wang,Weinan Zhang,Ken Chen,Shaodian Zhang,Yong Yu.* WWW 2019. [paper](https://sci-hub.tw/10.1145/3308558.3313520#)

1. **DDGK: Learning Graph Representations via Deep Divergence Graph Kernels.**
*Rami Al-Rfou,Dustin Zelle,Bryan Perozzi.* WWW 2019. [paper](https://arxiv.org/pdf/1904.09671.pdf)

1. **Tag2Vec: Learning Tag Representations in Tag Networks.**
*Junshan Wang,Zhicong Lu,Guojia Song,Yue Fan,Lun Du,Wei Lin.* WWW 2019. [paper](https://arxiv.org/pdf/1905.03041.pdf)

1. **struc2vec: Learning Node Representations from Structural Identity.**
*Leonardo F. R. Ribeiro, Pedro H. P. Saverese, Daniel R. Figueiredo.* KDD 2017. [paper](https://arxiv.org/pdf/1704.03165.pdf)

1. **Inductive Representation Learning on Large Graphs.**
*William L. Hamilton, Rex Ying, Jure Leskovec.* NIPS 2017. [paper](https://arxiv.org/pdf/1706.02216.pdf)

1. **Learning Graph Embeddings with Embedding Propagation.**
*Alberto Garcia Duran, Mathias Niepert.* NIPS 2017. [paper](https://arxiv.org/pdf/1710.03059.pdf)

1. **Enhancing the Network Embedding Quality with Structural Similarity.**
*Tianshu Lyu, Yuan Zhang, Yan Zhang.* CIKM 2017. [paper](https://pdfs.semanticscholar.org/e54a/374d7e24260450e2081b93005a491d1b9116.pdf)

1. **An Attention-based Collaboration Framework for Multi-View Network Representation Learning.**
*Meng Qu, Jian Tang, Jingbo Shang, Xiang Ren, Ming Zhang, Jiawei Han.* CIKM 2017. [paper](https://arxiv.org/pdf/1709.06636.pdf)

1. **On Embedding Uncertain Graphs.**
*Jiafeng Hu, Reynold Cheng, Zhipeng Huang, Yixang Fang, Siqiang Luo.* CIKM 2017. [paper](https://i.cs.hku.hk/~zphuang/pub/CIKM17.pdf)

1. **Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec.**
*Jiezhong Qiu, Yuxiao Dong, Hao Ma, Jian Li, Kuansan Wang, Jie Tang.* WSDM 2018. [paper](https://arxiv.org/pdf/1710.02971.pdf)

1. **Conditional Network Embeddings.**
*Bo Kang, Jefrey Lijffijt, Tijl De Bie.* ICLR 2019. [paper](https://arxiv.org/abs/1805.07544)

1. **Deep Graph Infomax.**
*Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm.* ICLR 2019. [paper](https://arxiv.org/abs/1809.10341)

1. **Anonymous Walk Embeddings.**
*Sergey Ivanov, Evgeny Burnaev.* ICML 2018. [paper](https://arxiv.org/pdf/1805.11921.pdf)

1. **Fairwalk: Towards Fair Graph Embedding.**
*Tahleen Rahman, Bartlomiej Surma, Michael Backes, Yang Zhang.* IJCAI 2019. [paper](https://yangzhangalmo.github.io/papers/IJCAI19.pdf)

1. **Graph and Autoencoder Based Feature Extraction for Zero-shot Learning.**
*Yang Liu, Deyan Xie, Quanxue Gao, Jungong Han, Shujian Wang, Xinbo Gao.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0421.pdf)

1. **Graph Space Embedding.**
*João Pereira, Evgeni Levin, Erik Stroes, Albert Groen.* IJCAI 2019. [paper](https://arxiv.org/abs/1907.13443)

1. **Arbitrary-Order Proximity Preserved Network Embedding.**
*Ziwei Zhang, Peng Cui, Xiao Wang, Jian Pei, Xuanrong Yao, Wenwu Zhu.* KDD 2018. [paper](http://cuip.thumedialab.com/papers/NE-ArbitraryProximity.pdf)

1. **Deep Variational Network Embedding in Wasserstein Space.**
*Dingyuan Zhu, Peng Cui, Daixin Wang, Wenwu Zhu.* KDD 2018. [paper](http://cuip.thumedialab.com/papers/NE-DeepVariational.pdf)

1. **MEGAN: A Generative Adversarial Network for Multi-View Network Embedding.**
*Yiwei Sun, Suhang Wang, Tsung-Yu Hsieh, Xianfeng Tang, Vasant Honavar.* IJCAI 2019. [paper](https://arxiv.org/abs/1909.01084)

1. **Network Embedding under Partial Monitoring for Evolving Networks**
*Yu Han, Jie Tang, Qian Chen.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0342.pdf)

1. **Network Embedding with Dual Generation Tasks.**
*Jie Liu, Na Li, Zhicheng He.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0709.pdf)

1. **Triplet Enhanced AutoEncoder: Model-free Discriminative Network Embedding.**
*Yao Yang, Haoran Chen, Junming Shao.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0745.pdf)

1. **Deep Recursive Network Embedding with Regular Equivalence.**
*Ke Tu, Peng Cui, Xiao Wang, Philip S. Yu, Wenwu Zhu.* KDD 2018. [paper](http://cuip.thumedialab.com/papers/NE-RegularEquivalence.pdf)

1. **Learning Structural Node Embeddings via Diffusion Wavelets.**
*Claire Donnat, Marinka Zitnik, David Hallac, Jure Leskovec.* KDD 2018. [paper](https://arxiv.org/pdf/1710.10321.pdf)

1. **Self-Paced Network Embedding.**
*Hongchang Gao, Heng Huang.* KDD 2018. [paper](https://par.nsf.gov/servlets/purl/10074506)

1. **Learning Deep Network Representations with Adversarially Regularized Autoencoders.**
*Wenchao Yu, Cheng Zheng, Wei Cheng, Charu Aggarwal, Dongjin Song, Bo Zong, Haifeng Chen, Wei Wang.* KDD 2018. [paper](https://sites.cs.ucsb.edu/~bzong/doc/kdd-18.pdf)

1. **Large-Scale Learnable Graph Convolutional Networks.**
*Hongyang Gao, Zhengyang Wang, Shuiwang Ji.* KDD 2018. [paper](https://arxiv.org/pdf/1808.03965)

1. **DFNets: Spectral CNNs for Graphs with Feedback-Looped Filters.**
*W. O. K. Asiri Suranga Wijesinghe, Qing Wang.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-3235)

1. **vGraph: A Generative Model for Joint Community Detection and Node Representation Learning.**
*Fan-Yun Sun, Meng Qu, Jordan Hoffmann, Chin-Wei Huang, Jian Tang.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-288)

1. **GraphZoom: A Multi-level Spectral Approach for Accurate and Scalable Graph Embedding.**
*Chenhui Deng, Zhiqiang Zhao, Yongyu Wang, Zhiru Zhang, Zhuo Feng.* ICLR 2020. [paper](https://openreview.net/pdf?id=r1lGO0EKDH)

1. **Inductive and Unsupervised Representation Learning on Graph Structured Objects.**
*Lichen Wang, Bo Zong, Qianqian Ma, Wei Cheng, Jingchao Ni, Wenchao Yu, Yanchi Liu, Dongjin Song, Haifeng Chen, Yun Fu.* ICLR 2020. [paper](https://openreview.net/pdf?id=rkem91rtDB)

#### [Attributed Network](#content)

1. **Outlier Aware Network Embedding for Attributed Networks.**
*Sambaran Bandyopadhyay, N. Lokesh, M. N. Murty.* AAAI 2019. [paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/3763)

1. **Large-Scale Heterogeneous Feature Embedding.**
*Xiao Huang, Qingquan Song, Fan Yang, Xia Hu.* AAAI 2019. [paper](https://aaai.org/ojs/index.php/AAAI/article/view/4276)

1. **Deep Bayesian Optimization on Attributed Graphs.**
*Jiaxu Cui, Bo Yang, Xia Hu.* AAAI 2019. [paper](https://arxiv.org/pdf/1905.13403.pdf)

1. **Efficient Attributed Network Embedding via Recursive Randomized Hashing.**
*Wei Wu, Bin Li, Ling Chen, Chengqi Zhang.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0397.pdf)

1. **Deep Attributed Network Embedding.**
*Hongchang Gao, Heng Huang.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0467.pdf)

1. **ANRL: Attributed Network Representation Learning via Deep Neural Networks.**
*Zhen Zhang, Hongxia Yang, Jiajun Bu, Sheng Zhou, Pinggang Yu, Jianwei Zhang, Martin Ester, Can Wang.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0438.pdf)

1. **Integrative Network Embedding via Deep Joint Reconstruction.**
*Di Jin, Meng Ge, Liang Yang, Dongxiao He, Longbiao Wang, Weixiong Zhang.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0473.pdf)

1. **node2vec: Scalable Feature Learning for Networks.**
*Aditya Grover, Jure Leskovec.* KDD 2016. [paper](http://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf) [code](https://github.com/aditya-grover/node2vec)

1. **Network Representation Learning with Rich Text Information.**
*Cheng Yang, Zhiyuan Liu, Deli Zhao, Maosong Sun, Edward Y. Chang.* IJCAI 2015. [paper](http://thunlp.org/~yangcheng/publications/ijcai15.pdf) [code](https://github.com/thunlp/tadw)

1. **Tri-Party Deep Network Representation.**
*Shirui Pan, Jia Wu, Xingquan Zhu, Chengqi Zhang, Yang Wang.* IJCAI 2016. [paper](https://www.ijcai.org/Proceedings/16/Papers/271.pdf)

1. **TransNet: Translation-Based Network Representation Learning for Social Relation Extraction.**
*Cunchao Tu, Zhengyan Zhang, Zhiyuan Liu, Maosong Sun.* IJCAI 2017. [paper](http://thunlp.org/~tcc/publications/ijcai2017_transnet.pdf) [code](https://github.com/thunlp/transnet)

1. **PRRE: Personalized Relation Ranking Embedding for Attributed Networks.**
*Sheng Zhou, Hongxia Yang, Xin Wang, Jiajun Bu, Martin Ester, Pinggang Yu, Jianwei Zhang, Can Wang.* CIKM 2018. [paper](https://dl.acm.org/citation.cfm?id=3271741) [code](https://github.com/zhoushengisnoob/PRRE)

1. **RSDNE: Exploring Relaxed Similarity and Dissimilarity from Completely-imbalanced Labels for Network Embedding.**
*Zheng Wang, Xiaojun Ye, Chaokun Wang, YueXin Wu, Changping Wang, Kaiwen Liang.* AAAI 2018. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16062/15722) [code](https://github.com/zhengwang100/RSDNE)

1. **Semi-supervised embedding in attributed networks with outliers.**
*Jiongqian Liang, Peter Jacobs, Jiankai Sun, and Srinivasan Parthasarathy.* SDM 2018. [paper](https://arxiv.org/pdf/1703.08100.pdf) [code](http://jiongqianliang.com/SEANO/)

1. **A Representation Learning Framework for Property Graphs.**
*Yifan Hou,Hongzhi Chen,Changji Li,James Cheng,Ming-Chang Yang.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330948/p65-hou.pdf)

1. **Learning from Labeled and Unlabeled Vertices in Networks.**
*Wei Ye, Linfei Zhou, Dominik Mautz, Claudia Plant, Christian B?hm.* KDD 2017. [paper](https://dl.acm.org/citation.cfm?id=3098142)

1. **Label Informed Attributed Network Embedding.**
*Xiao Huang, Jundong Li, Xia Hu.* WSDM 2017. [paper](http://www.public.asu.edu/~jundongl/paper/WSDM17_LANE.pdf)

1. **Accelerated Attributed Network Embedding.**
*Xiao Huang, Jundong Li, Xia Hu.* SDM 2017. [paper](http://www.public.asu.edu/~jundongl/paper/SDM17_AANE.pdf)

1. **Variation Autoencoder Based Network Representation Learning for Classification.**
*Hang Li, Haozheng Wang, Zhenglu Yang, Masato Odagaki.* ACL 2017. [paper](https://aclweb.org/anthology/P17-3010)

1. **Attributed Signed Network Embedding.**
*Suhang Wang, Charu Aggarwal, Jiliang Tang, Huan Liu.* CIKM 2017. [paper](https://suhangwang.ist.psu.edu/publications/SNEA.pdf)

1. **From Properties to Links: Deep Network Embedding on Incomplete Graphs.**
*Dejian Yang, Senzhang Wang, Chaozhuo Li, Xiaoming Zhang, Zhoujun Li.* CIKM 2017. [paper](https://dl.acm.org/citation.cfm?id=3132847.3132975)

1. **Exploring Expert Cognition for Attributed Network Embedding.**
*Xiao Huang, Qingquan Song, Jundong Li, Xia Ben Hu.* WSDM 2018. [paper](http://www.public.asu.edu/~jundongl/paper/WSDM18_NEEC.pdf)

1. **Hierarchical Taxonomy Aware Network Embedding.**
*Jianxin Ma, Peng Cui, Xiao Wang, Wenwu Zhu.* KDD 2018. [paper](https://jianxinma.github.io/assets/NetHiex.pdf)

1. **Network-Specific Variational Auto-Encoder for Embedding in Attribute Networks.**
*Di Jin, Bingyi Li, Pengfei Jiao, Dongxiao He, Weixiong Zhang.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0370.pdf)

1. **SPINE: Structural Identity Preserved Inductive Network Embedding.**
*Junliang Guo, Linli Xu, Jingchang Liu.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0333.pdf)

1. **Content to Node: Self-translation Network Embedding.**
*Jie Liu, Zhicheng He, Lai Wei, Yalou Huang.* KDD 2018. [paper](https://dl.acm.org/citation.cfm?id=3219988)

1. **Wasserstein Weisfeiler-Lehman Graph Kernels.**
*Matteo Togninalli, Elisabetta Ghisu, Felipe Llinares-López, Bastian Rieck, Karsten Borgwardt.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-3470)

1. **Rethinking Kernel Methods for Node Representation Learning on Graphs.**
*Yu Tian, Long Zhao, Xi Peng, Dimitris Metaxas.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-6235)

#### [Dynamic Network](#content)

1. **Dynamic Network Embedding : An Extended Approach for Skip-gram based Network Embedding.**
*Lun Du, Yun Wang, Guojie Song, Zhicong Lu, Junshan Wang.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0288.pdf)

1. **Dynamic Network Embedding by Modeling Triadic Closure Process.**
*Lekui Zhou, Yang Yang, Xiang Ren, Fei Wu, Yueting Zhuang.* AAAI 2018. [paper](http://yangy.org/works/dynamictriad/dynamic_triad.pdf)

1. **DepthLGP: Learning Embeddings of Out-of-Sample Nodes in Dynamic Networks.**
*Jianxin Ma, Peng Cui, Wenwu Zhu.* AAAI 2018. [paper](https://jianxinma.github.io/assets/DepthLGP.pdf)

1. **TIMERS: Error-Bounded SVD Restart on Dynamic Networks.**
*Ziwei Zhang, Peng Cui, Jian Pei, Xiao Wang, Wenwu Zhu.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.09541.pdf)

1. **Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks.**
*Srijan Kumar,Xikun Zhang,Jure Leskovec.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330895/p1269-kumar.pdf)

1. **Attributed Network Embedding for Learning in a Dynamic Environment.**
*Jundong Li, Harsh Dani, Xia Hu, Jiliang Tang, Yi Chang, Huan Liu.* CIKM 2017. [paper](https://arxiv.org/pdf/1706.01860.pdf)

1. **DyRep: Learning Representations over Dynamic Graphs.**
*Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, Hongyuan Zha.* ICLR 2019. [paper](https://openreview.net/forum?id=HyePrhR5KX)

1. **Embedding Temporal Network via Neighborhood Formation.**
*Yuan Zuo, Guannan Liu, Hao Lin, Jia Guo, Xiaoqian Hu, Junjie Wu.* KDD 2018. [paper](https://zuoyuan.github.io/files/htne_kdd18.pdf)

1. **Node Embedding over Temporal Graphs.**
*Uriel Singer, Ido Guy, Kira Radinsky.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0640.pdf)

1. **Dynamic Embeddings for User Profiling in Twitter.**
*Shangsong Liang, Xiangliang Zhang, Zhaochun Ren, Evangelos Kanoulas.* KDD 2018. [paper](https://repository.kaust.edu.sa/bitstream/handle/10754/628781/p1764-liang.pdf?sequence=1&isAllowed=y)

1. **NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks.**
*Wenchao Yu, Wei Cheng, Charu Aggarwal, Kai Zhang, Haifeng Chen, Wei Wang.* KDD 2018. [paper](http://www.shichuan.org/hin/topic/Embedding/2018.KDD%202018%20NetWalk_A%20Flexible%20Deep%20Embedding%20Approach%20for%20Anomaly%20Detection%20in%20Dynamic%20Networks.pdf)

1. **Scalable Optimization for Embedding Highly-Dynamic and Recency-Sensitive Data.**
*Xumin Chen, Peng Cui, Shiqiang Yang.* KDD 2018. [paper](http://pengcui.thumedialab.com/papers/NE-ScalableOptimization.pdf)

1. **Variational Graph Recurrent Neural Networks.**
*Ehsan Hajiramezanali, Arman Hasanzadeh, Krishna Narayanan, Nick Duffield, Mingyuan Zhou, Xiaoning Qian.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-5712)

#### [Heterogeneous Information Network](#content)

1. **Relation Structure-Aware Heterogeneous Information Network Embedding.**
*Yuanfu Lu, Chuan Shi, Linmei Hu, Zhiyuan Liu.* AAAI 2019. [paper](https://arxiv.org/abs/1905.08027)

1. **Hyperbolic Heterogeneous Information Network Embedding.**
*Xiao Wang, Yiding Zhang, Chuan Shi.* AAAI 2019. [paper](http://shichuan.org/doc/65.pdf)

1. **Learning Latent Representations of Nodes for Classifying in Heterogeneous Social Networks.**
*Yann Jacob, Ludovic Denoyer, Patrick Gallinar.* WSDM 2014. [paper](http://webia.lip6.fr/~gallinar/gallinari/uploads/Teaching/WSDM2014-jacob.pdf)

1. **Heterogeneous Network Embedding via Deep Architectures.**
*Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, Charu C. Aggarwal, Thomas S. Huang.* KDD 2015. [paper](http://www.ifp.illinois.edu/~chang87/papers/kdd_2015.pdf)

1. **metapath2vec: Scalable Representation Learning for Heterogeneous Networks.**
*Yuxiao Dong, Nitesh V. Chawla, Ananthram Swami.* KDD 2017. [paper](https://www3.nd.edu/~dial/publications/dong2017metapath2vec.pdf) [code](https://ericdongyx.github.io/metapath2vec/m2v.html)

1. **SHNE: Representation Learning for Semantic-Associated Heterogeneous Networks.**
*Chuxu Zhang, Ananthram Swami, Nitesh V. Chawla.* WSDM 2019. [paper](https://dl.acm.org/citation.cfm?id=3291001) [code](https://github.com/chuxuzhang/WSDM2019_SHNE)

1. **Are Meta-Paths Necessary?: Revisiting Heterogeneous Graph Embeddings.**
*Rana Hussein, Dingqi Yang, Philippe Cudré-Mauroux.* CIKM 2018. [paper](https://exascale.info/assets/pdf/hussein2018cikm.pdf)

1. **Abnormal Event Detection via Heterogeneous Information Network Embedding.**
*Shaohua Fan, Chuan Shi, Xiao Wang.* CIKM 2018. [paper](http://shichuan.org/doc/62.pdf)

1. **Multidimensional Network Embedding with Hierarchical Structures.**
*Yao Ma, Zhaochun Ren, Ziheng Jiang, Jiliang Tang, Dawei Yin.* WSDM 2018. [paper](http://cse.msu.edu/~mayao4/downloads/Multidimensional_Network_Embedding_with_Hierarchical_Structure.pdf)

1. **Curriculum Learning for Heterogeneous Star Network Embedding via Deep Reinforcement Learning.**
*Meng Qu, Jian Tang, Jiawei Han.* WSDM 2018. [paper](http://delivery.acm.org/10.1145/3160000/3159711/p468-qu.pdf?ip=203.205.141.123&id=3159711&acc=ACTIVE%20SERVICE&key=39FCDE838982416F%2E39FCDE838982416F%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1519788484_7383495a5c522cbe124e62e4d768f8cc)

1. **Generative Adversarial Network based Heterogeneous Bibliographic Network Representation for Personalized Citation Recommendation.**
*J. Han, Xiaoyan Cai, Libin Yang.* AAAI 2018. [paper](https://pdfs.semanticscholar.org/1596/d6487012696ba400fb69904a2c372a08a2be.pdf)

1. **Distance-aware DAG Embedding for Proximity Search on Heterogeneous Graphs.**
*Zemin Liu, Vincent W. Zheng, Zhou Zhao, Fanwei Zhu, Kevin Chen-Chuan Chang, Minghui Wu, Jing Ying.* AAAI 2018. [paper](https://pdfs.semanticscholar.org/b1cc/127a65c40e71121106d0c663f9b5baf9d6f9.pdf)

1. **Representation Learning for Attributed Multiplex Heterogeneous Network.**
*Yukuo Cen,Xu Zou,Jianwei Zhang,Hongxia Yang,Jingren Zhou,Jie Tang.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330964/p1358-cen.pdf)

1. **Adversarial Learning on Heterogeneous Information Networks.**
*Binbin Hu,Yuan Fang,Chuan Shi* KDD 2019 [paper](http://delivery.acm.org/10.1145/3340000/3330970/p120-hu.pdf)

1. **HetGNN: Heterogeneous Graph Neural Network.**
*Chuxu Zhang,Dongjin Song,Chao Huang,Ananthram Swami,Nitesh V. Chawla.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330961/p793-zhang.pdf)

1. **IntentGC: a Scalable Graph Convolution Framework Fusing Heterogeneous Information for Recommendation**
*Jun Zhao, Zhou Zhou, Ziyu Guan, Wei Zhao, Ning Wei, Guang Qiu and Xiaofei He.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330686/p2347-zhao.pdf)

1. **Metapath-guided Heterogeneous Graph Neural Network for Intent Recommendation.**
*Shaohua Fan, Junxiong Zhu, Xiaotian Han, Chuan Shi, Linmei Hu, Biyu Ma and Yongliang Li.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330673/p2478-fan.pdf)

1. **Your Style Your Identity: Leveraging Writing and Photography Styles for Drug Trafficker Identification in Darknet Markets over Attributed Heterogeneous Information Network.**
*Yiming Zhang, Yujie Fan,Wei Song, Shifu HouYanfang Ye, Xin Li,Liang Zhao,Chuan Shi,Jiabin Wang, Qi Xiong.* WWW 2019. [paper](https://www.gwern.net/docs/sr/2019-zhang.pdf)

1. **HIN2Vec: Explore Meta-paths in Heterogeneous Information Networks for Representation Learning.**
*Tao-yang Fu, Wang-Chien Lee, Zhen Lei.* CIKM 2017. [paper](http://shichuan.org/hin/topic/Embedding/2017.%20CIKM%20HIN2Vec.pdf)

1. **SHINE: Signed Heterogeneous Information Network Embedding for Sentiment Link Prediction.**
*Hongwei Wang, Fuzheng Zhang, Min Hou, Xing Xie, Minyi Guo, Qi Liu.* WSDM 2018. [paper](https://arxiv.org/pdf/1712.00732.pdf)

1. **ActiveHNE: Active Heterogeneous Network Embedding.**
*Xia Chen, Guoxian Yu, Jun Wang, Carlotta Domeniconi, Zhao Li, Xiangliang Zhang.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0294.pdf)

1. **Unified Embedding Model over Heterogeneous Information Network for Personalized Recommendation.**
*Zekai Wang, Hongzhi Liu, Yingpeng Du, Zhonghai Wu, Xing Zhang.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0529.pdf)

1. **Easing Embedding Learning by Comprehensive Transcription of Heterogeneous Information Networks.**
*Yu Shi, Qi Zhu, Fang Guo, Chao Zhang, Jiawei Han.* KDD 2018. [paper](https://yu-shi-homepage.github.io/kdd18.pdf)

1. **PME: Projected Metric Embedding on Heterogeneous Networks for Link Prediction.**
*Hongxu Chen, Hongzhi Yin, Weiqing Wang, Hao Wang, Quoc Viet Hung Nguyen, Xue Li.* KDD 2018. [paper](http://net.pku.edu.cn/daim/hongzhi.yin/papers/KDD18-Hongxu.pdf)

#### [Bipartite Network](#content)

1. **Collaborative Similarity Embedding for Recommender Systems.**
*Chih-Ming Chen,Chuan-Ju Wang,Ming-Feng Tsai,Yi-Hsuan Yang.* WWW 2019. [paper](https://arxiv.org/pdf/1902.06188.pdf)

1. **Learning Node Embeddings in Interaction Graphs.**
*Yao Zhang, Yun Xiong, Xiangnan Kong, Yangyong Zhu.* CIKM 2017. [paper](https://web.cs.wpi.edu/~xkong/publications/papers/cikm17.pdf)

1. **Hierarchical Representation Learning for Bipartite Graphs.**
*Chong Li, Kunyang Jia, Dan Shen, C.J. Richard Shi, Hongxia Yang.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0398.pdf)

#### [Directed Network](#content)

1. **ATP: Directed Graph Embedding with Asymmetric Transitivity Preservation.**
*Jiankai Sun, Bortik Bandyopadhyay, Armin Bashizade, Jiongqian Liang, P. Sadayappan, Srinivasan Parthasarathy.* AAAI 2019. [paper](https://www.aaai.org/ojs/index.php/AAAI/article/view/3794)

1. **Asymmetric Transitivity Preserving Graph Embedding.**
*Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, Wenwu Zhu.* KDD 2016. [paper](http://cuip.thumedialab.com/papers/hoppe.pdf)

1. **"Bridge": Enhanced Signed Directed Network Embedding.**
*Yiqi Chen, Tieyun Qian, Huan Liu, Ke Sun.* CIKM 2018. [paper](https://dl.acm.org/citation.cfm?id=3271738)

1. **SIDE: Representation Learning in Signed Directed Networks.**
*Junghwan Kim, Haekyu Park, Ji-Eun Lee, U Kang.* WWW 2018. [paper](https://datalab.snu.ac.kr/side/resources/side.pdf)

1. **Low-dimensional statistical manifold embedding of directed graphs.**
*Thorben Funke, Tian Guo, Alen Lancic, Nino Antulov-Fantulin.* ICLR 2020. [paper](https://openreview.net/pdf?id=SkxQp1StDH)

#### [Other Models](#content)

1. **Scalable Multiplex Network Embedding. （Multiplex Network)**
*Hongming Zhang, Liwei Qiu, Lingling Yi, Yangqiu Song.* IJCAI 2018. [paper](http://www.cse.ust.hk/~yqsong/papers/2018-IJCAI-MultiplexNetworkEmbedding.pdf)

1. **Structural Deep Embedding for Hyper-Networks. (Hyper-Network)**
*Ke Tu, Peng Cui, Xiao Wang, Fei Wang, Wenwu Zhu.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.10146.pdf)

1. **Representation Learning for Scale-free Networks. (Scale-free Network)**
*Rui Feng, Yang Yang, Wenjie Hu, Fei Wu, Yueting Zhuang.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.10755.pdf)

1. **Co-Regularized Deep Multi-Network Embedding. (Multi-Network)**
*Jingchao Ni, Shiyu Chang, Xiao Liu, Wei Cheng, Haifeng Chen, Dongkuan Xu, Xiang Zhang.* WWW 2018. [paper](https://nijingchao.github.io/paper/www18_dmne.pdf)

1. **Joint Link Prediction and Network Alignment via Cross-graph Embedding. (Multi-Network)**
*Xingbo Du, Junchi Yan, Hongyuan Zha.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0312.pdf)

1. **DANE: Domain Adaptive Network Embedding. (Multi-Network)**
*Yizhou Zhang, Guojie Song, Lun Du, Shuwen Yang, Yilun Jin.* IJCAI 2019. [paper](https://arxiv.org/abs/1906.00684)

1. **SPARC: Self-Paced Network Representation for Few-Shot Rare Category Characterization. (Few-Shot Learning)**
*Dawei Zhou, Jingrui He, Hongxia Yang, Wei Fan.* KDD 2018. [paper](https://dl.acm.org/authorize?N665885)

### [Applications](#content)

#### [Natural Language Processing](#content)

1. **Personalized Question Routing via Heterogeneous Network Embedding.**
*Zeyu Li, Jyun-Yu Jiang, Yizhou Sun, Wei Wang.* AAAI 2019. [paper](http://web.cs.ucla.edu/~yzsun/papers/2019_AAAI_QR.pdf)

1. **PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks.**
*Jian Tang, Meng Qu, Qiaozhu Mei.* KDD 2015. [paper](https://arxiv.org/pdf/1508.00200.pdf) [code](https://github.com/mnqu/PTE)

#### [Knowledge Graph](#content)

1. **Interaction Embeddings for Prediction and Explanation in Knowledge Graphs.**
*Wen Zhang, Bibek Paudel, Wei Zhang, Abraham Bernstein, Huajun Chen.* WSDM 2019. [paper](https://arxiv.org/pdf/1903.04750.pdf)

1. **Shared Embedding Based Neural Networks for Knowledge Graph Completion.**
*Saiping Guan, Xiaolong Jin, Yuanzhuo Wang, Xueqi Cheng.* CIKM 2018 [paper](https://dl.acm.org/citation.cfm?id=3271705)

1. **Re-evaluating Embedding-Based Knowledge Graph Completion Methods.**
*Farahnaz Akrami, Lingbing Guo, Wei Hu, Chengkai Li.* CIKM 2018. [paper](http://ranger.uta.edu/~cli/pubs/2018/kgcompletion-cikm18short-akrami.pdf)

1. **Quaternion Knowledge Graph Embeddings.**
*SHUAI ZHANG, Yi Tay, Lina Yao, Qi Liu.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-1565)

1. **Neural Recurrent Structure Search for Knowledge Graph Embedding.**
*Yongqi Zhang, Quanming Yao, Lei Chen.* AAAI 2020. [paper](https://arxiv.org/abs/1911.07132)

1. **Knowledge Graph Alignment Network with Gated Multi-­‐hop Neighborhood Aggregation.**
*Zequn Sun, Chengming Wang, Wei Hu, Muhao Chen, Jian Dai, Wei Zhang, Yuzhong Qu.* AAAI 2020. [paper](https://arxiv.org/abs/1911.08936)

#### [Social Network](#content)

1. **Adversarial Learning for Weakly-Supervised Social Network Alignment.**
*Chaozhuo Li, Senzhang Wang, Yukun Wang, Philip Yu, Yanbo Liang, Yun Liu, Zhoujun Li.* AAAI 2019. [paper](https://aaai.org/ojs/index.php/AAAI/article/view/3889)

1. **TransConv: Relationship Embedding in Social Networks.**
*Yi-Yu Lai, Jennifer Neville, Dan Goldwasser.* AAAI 2019. [paper](https://aaai.org/ojs/index.php/AAAI/article/view/4314)

1. **Semi-supervised User Geolocation via Graph Convolutional Networks.**
*Afshin Rahimi, Trevor Cohn, Timothy Baldwin.* ACL 2018. [paper](https://arxiv.org/pdf/1804.08049.pdf)

1. **MASTER: across Multiple social networks, integrate Attribute and STructure Embedding for Reconciliation.**
*Sen Su, Li Sun, Zhongbao Zhang, Gen Li, Jielun Qu.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0537.pdf)

1. **MEgo2Vec: Embedding Matched Ego Networks for User Alignment Across Social Networks.**
*Jing Zhang, Bo Chen, Xianming Wang, Hong Chen, Cuiping Li, Fengmei Jin, Guojie Song, Yutao Zhang.* CIKM 2018. [paper](https://dl.acm.org/citation.cfm?id=3271705)

1. **Link Prediction via Subgraph Embedding-Based Convex Matrix Completion.**
*Zhu Cao, Linlin Wang, Gerard De melo.* AAAI 2018. [paper](http://iiis.tsinghua.edu.cn/~weblt/papers/link-prediction-subgraphembeddings.pdf)

1. **On Exploring Semantic Meanings of Links for Embedding Social Networks.**
*Linchuan Xu, Xiaokai Wei, Jiannong Cao, Philip S Yu.* WWW 2018. [paper](https://pdfs.semanticscholar.org/ccd3/ede78393628b5f0256ebfccbb4ac293394de.pdf)

1. **MCNE: An End-to-End Framework for Learning Multiple Conditional Network Representations of Social Network.**
*Hao Wang,Tong Xu,Qi Liu,Defu Lian,Enhong Chen,Dongfang Du,Han Wu,Wen Su.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330931/p1064-wang.pdf)

1. **Unsupervised Feature Selection in Signed Social Networks.**
*Kewei Cheng, Jundong Li, Huan Liu.* KDD 2017. [paper](http://www.public.asu.edu/~jundongl/paper/KDD17_SignedFS.pdf)

1. **Learning Network Embedding with Community Structural Information.**
*Yu Li, Ying Wang, Tingting Zhang, Jiawei Zhang, Yi Chang.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0407.pdf)

#### [Graph Clustering](#content)

1. **Spectral Clustering in Heterogeneous Information Networks.**
*Xiang Li , Ben Kao, Zhaochun Ren, Dawei Yin.* AAAI 2019. [paper](https://www.researchgate.net/profile/Xiang_Li238/publication/332606853_Spectral_Clustering_in_Heterogeneous_Information_Networks/links/5cc035e892851c8d2200aa29/Spectral-Clustering-in-Heterogeneous-Information-Networks.pdf)

1. **Multi-view Clustering with Graph Embedding for Connectome Analysis.**
*Guixiang Ma, Lifang He, Chun-Ta Lu, Weixiang Shao, Philip S Yu, Alex D Leow, Ann B Ragin.* CIKM 2017. [paper](https://www.cs.uic.edu/~clu/doc/cikm17_mcge.pdf)

1. **Adversarial Graph Embedding for Ensemble Clustering.**
*Zhiqiang Tao, Hongfu Liu, Jun Li, Zhaowen Wang, Yun Fu.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0494.pdf)

1. **Variational Graph Embedding and Clustering with Laplacian Eigenmaps.**
*Zitai Chen, Chuan Chen, Zong Zhang, Zibin Zheng, Qingsong Zou.* IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0297.pdf)

#### [Community Detection](#content)

1. **Incorporating Network Embedding into Markov Random Field for Better Community Detection.**
*Di Jin, Xinxin You, Weihao Li, Dongxiao He, Peng Cui, Francoise Fogelman-Soulie, Tanmoy Chakraborty.* AAAI 2019. [paper](http://pengcui.thumedialab.com/papers/NE-MRF.pdf)

1. **A Unified Framework for Community Detection and Network Representation Learning.**
*Cunchao Tu, Xiangkai Zeng, Hao Wang, Zhengyan Zhang, Zhiyuan Liu, Maosong Sun, Bo Zhang, Leyu Lin.* TKDE 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8403293/)

1. **COSINE: Community-Preserving Social Network Embedding from Information Diffusion Cascades.**
*Yuan Zhang, Tianshu Lyu, Yan Zhang.* AAAI 2018. [paper](https://pdfs.semanticscholar.org/fec8/24c51b59063ba92b66bb7404010954ced5ac.pdf)

1. **Multi-facet Network Embedding: Beyond the General Solution of Detection and Representation.**
*Liang Yang, Xiaochun Cao, Yuanfang Guo.* AAAI 2018. [paper](https://yangliang.github.io/pdf/aaai18.pdf)

1. **Community Detection in Attributed Graphs: An Embedding Approach.**
*Ye Li, Chaofeng Sha, Xin Huang, Yanchun Zhang.* AAAI 2018. [paper](https://www.comp.hkbu.edu.hk/~xinhuang/publications/pdfs/AAAI18.pdf)

1. **Preserving Proximity and Global Ranking for Node Embedding.**
*Yi-An Lai, Chin-Chi Hsu, Wenhao Chen, Mi-Yen Yeh, Shou-De Lin.* NIPS 2017. [paper](https://pdfs.semanticscholar.org/b692/c82115889115ef3e63fb7e6b23c8eb9c85b3.pdf)

1. **Learning Community Embedding with Community Detection and Node Embedding on Graphs.**
*Sandro Cavallari, Vincent W. Zheng, Hongyun Cai, Kevin ChenChuan Chang, Erik Cambria.* CIKM 2017. [paper](https://sentic.net/community-embedding.pdf)

1. **End to end learning and optimization on graphs.**
*Bryan Wilder, Eric Ewing, Bistra Dilkina, Milind Tambe.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-2620)

#### [Recommendation](#content)

1. **Graph Convolutional Neural Networks for Web-Scale Recommender Systems.**
*Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, Jure Leskovec.* KDD 2018. [paper](https://arxiv.org/pdf/1806.01973.pdf)

1. **Is a Single Vector Enough? Exploring Node Polysemy for Network Embedding.**
*Ninghao Liu,Qiaoyu Tan,Yuening Li,Hongxia Yang,Jingren Zhou,Xia Hu.* KDD 2019. [paper](https://arxiv.org/pdf/1905.10668.pdf)

1. **Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender System.**
*Qitian Wu,Hengrui Zhang,Xiaofeng Gao,Peng He,Paul Weng,Han Gao,Guihai Chen.* WWW 2019. [paper](https://arxiv.org/pdf/1903.10433.pdf)

#### [Other Applications](#content)

1. **Cash-out User Detection based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism. (Finance)**
*Binbin Hu, Zhiqiang Zhang, Chuan Shi, Jun Zhou, Xiaolong Li, Yuan Qi.* AAAI 2019. [paper](http://shichuan.org/doc/64.pdf)

1. **Building Causal Graphs from Medical Literature and Electronic Medical Records. (Medicine)**
*Galia Nordon, Gideon Koren, Varda Shalev, Benny Kimelfeld, Uri Shalit, Kira Radinsky.* AAAI 2019. [paper](http://www.kiraradinsky.com/files/aaai-building-causal.pdf)

1. **Adversarial Attacks on Node Embeddings via Graph Poisoning. (Adversarial)**
*Aleksandar Bojchevski, Stephan Günnemann.* ICML 2019. [paper](http://proceedings.mlr.press/v97/bojchevski19a/bojchevski19a.pdf)

1. **Compositional Fairness Constraints for Graph Embeddings. (Adversarial)**
*Avishek Bose, William Hamilton.* ICML 2019. [paper](http://proceedings.mlr.press/v97/bose19a/bose19a.pdf)

1. **Gromov-Wasserstein Learning for Graph Matching and Node Embedding. (Graph Matching)**
*Hongteng Xu, Dixin Luo, Hongyuan Zha, Lawrence Carin Duke.* ICML 2019. [paper](http://proceedings.mlr.press/v97/xu19b/xu19b.pdf)

1. **Graph Matching Networks for Learning the Similarity of Graph Structured Objects. (Graph Matching)**
*Yujia Li, Chenjie Gu, Thomas Dullien, Oriol Vinyals, Pushmeet Kohli.* ICML 2019. [paper](http://proceedings.mlr.press/v97/li19d/li19d.pdf)

1. **MolGAN: An implicit generative model for small molecular graphs. (Molecular Generation)**
*Nicola De Cao, Thomas Kipf.* ICML Workshop 2018. [paper](https://arxiv.org/pdf/1805.11973.pdf)

1. **Relational recurrent neural networks. (Relational Reasoning)**
*Adam Santoro, Ryan Faulkner, David Raposo, Jack Rae, Mike Chrzanowski, Theophane Weber, Daan Wierstra, Oriol Vinyals, Razvan Pascanu, Timothy Lillicrap.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1806.01822.pdf)

1. **Constructing Narrative Event Evolutionary Graph for Script Event Prediction. (Script Event Prediction)**
*Zhongyang Li, Xiao Ding, Ting Liu.* IJCAI 2018. [paper](https://arxiv.org/abs/1805.05081) [code](https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018)

1. **A Network-embedding Based Method for Author Disambiguation. (Author Disambiguation)**
*Jun Xu, Siqi Shen, Dongsheng Li, Yongquan Fu.* CIKM 2018. [paper](https://dl.acm.org/citation.cfm?id=3269272)

1. **Deep Graph Embedding for Ranking Optimization in E-commerce.(E-commerce)**
*Chen Chu, Zhao Li, Beibei Xin, Fengchao Peng, Chuanren Liu, Remo Rohs, Qiong Luo, Jingren Zhou.* CIKM 2018. [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6330176/)

1. **Learning Network-to-Network Model for Content-rich Network Embedding.**
*Zhicheng He,Jie Liu,Na Li,Yalou Huang.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330924/p1037-he.pdf)

1. **Unifying Inter-region Autocorrelation and Intra-region Structures for Spatial Embedding via Collective Adversarial Learning.**
*Yunchao Zhang,Pengyang Wang,Xiaolin Li,Yu Zheng,Yanjie Fu.* KDD 2019. [paper](http://delivery.acm.org/10.1145/3340000/3330972/p1700-zhang.pdf)

1. **Neural IR Meets Graph Embedding: A Ranking Model for Product Search.**
*Yuan Zhang,Dong Wang,Yan Zhang.* WWW 2019. [paper](https://arxiv.org/pdf/1901.08286.pdf)

1. **Cross-Network Embedding for Multi-Network Alignment.**
*Xiaokai Chu,Xinxin Fan,Di Yao,Zhihua Zhu,Jianhui Huang,Jingping Bi.* WWW 2019. [paper](https://sci-hub.tw/10.1145/3308558.3313499#)

1. **Name Disambiguation in Anonymized Graphs using Network Embedding. (Name Disambiguation)**
*Baichuan Zhang, Mohammad Al Hasan.* CIKM 2017. [paper](https://arxiv.org/pdf/1702.02287.pdf)

1. **NetGAN: Generating Graphs via Random Walks. (Graph Generation)**
*Aleksandar Bojchevski, Oleksandr Shchur, Daniel Zügner, Stephan Günnemann.* ICML 2018. [paper](https://arxiv.org/pdf/1803.00816)

1. **Graph Networks as Learnable Physics Engines for Inference and Control. (Physics)**
*Alvaro Sanchez-Gonzalez, Nicolas Heess, Jost Tobias Springenberg, Josh Merel, Martin Riedmiller, Raia Hadsell, Peter Battaglia.* ICML 2018. [paper](https://arxiv.org/pdf/1806.01242.pdf)

1. **Relational inductive bias for physical construction in humans and machines. (Human physical reasoning)**
*Jessica B. Hamrick, Kelsey R. Allen, Victor Bapst, Tina Zhu, Kevin R. McKee, Joshua B. Tenenbaum, Peter W. Battaglia.* CogSci 2018. [paper](https://arxiv.org/pdf/1806.01203.pdf)

1. **N-Gram Graph: Simple Unsupervised Representation for Graphs, with Applications to Molecules. (Chemistry)**
*Shengchao Liu, Mehmet F Demirel, Yingyu Liang.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/9054-n-gram-graph-simple-unsupervised-representation-for-graphs-with-applications-to-molecules.pdf)

1. **Learning metrics for persistence-based summaries and applications for graph classification. (Graph classification)**
*Qi Zhao, Yusu Wang.* NeurIPS 2019. [paper](http://papers.nips.cc/paper/by-source-2019-5218)

1. **Cross-­‐Modality Attention with Semantic Graph Embedding for Multi-­‐Label Classification. (Multi-label classification)**
*Renchun You, Zhiyao Guo, Cui Lei, Xiang Long, Yingze Bao, Shilei Wen.* AAAI 2020 [paper](https://arxiv.org/abs/1912.07872)
