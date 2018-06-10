## Must-read papers on NRL/NE.
NRL: network representation learning. NE: network embedding.

Contributed by [Cunchao Tu](http://thunlp.org/~tcc/), Yuan Yao and Zhengyan Zhang.

We release [OpenNE](https://github.com/thunlp/openne), an open source toolkit for NE/NRL. This repository provides a standard NE/NRL(Network Representation Learning）training and testing framework. Currently, the implemented models in OpenNE include DeepWalk, LINE, node2vec, GraRep, TADW and GCN.

### Survey papers:

1. **Representation Learning on Graphs: Methods and Applications.**
*William L. Hamilton, Rex Ying, Jure Leskovec.* 2017. [paper](https://arxiv.org/pdf/1709.05584.pdf)

1. **Graph Embedding Techniques, Applications, and Performance: A Survey.**
*Palash Goyal, Emilio Ferrara.* 2017. [paper](https://arxiv.org/pdf/1705.02801.pdf)

1. **A Comprehensive Survey of Graph Embedding: Problems, Techniques and Applications.**
*Hongyun Cai, Vincent W. Zheng, Kevin Chen-Chuan Chang.* 2017. [paper](https://arxiv.org/pdf/1709.07604.pdf)

1. **Network Representation Learning: A Survey.**
*Daokun Zhang, Jie Yin, Xingquan Zhu, Chengqi Zhang.* 2018. [paper](https://arxiv.org/pdf/1801.05852.pdf)

1. **Network Representation Learning: An Overview.(In Chinese)**
*Cunchao Tu, Cheng Yang, Zhiyuan Liu, Maosong Sun.* 2017. [paper](http://engine.scichina.com/publisher/scp/journal/SSI/47/8/10.1360/N112017-00145)

1. ***Relational inductive biases, deep learning, and graph networks.***
*Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulcehre, Francis Song, Andrew Ballard, Justin Gilmer, George Dahl, Ashish Vaswani, Kelsey Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matt Botvinick, Oriol Vinyals, Yujia Li, Razvan Pascanu.* 2018. [paper](https://arxiv.org/pdf/1806.01261.pdf)

### Journal and Conference papers:

1. **DeepWalk: Online Learning of Social Representations.**
*Bryan Perozzi, Rami Al-Rfou, Steven Skiena.*  KDD 2014. [paper](https://arxiv.org/pdf/1403.6652) [code](https://github.com/phanein/deepwalk)

1. **Learning Latent Representations of Nodes for Classifying in Heterogeneous Social Networks.**
*Yann Jacob, Ludovic Denoyer, Patrick Gallinar.* WSDM 2014. [paper](http://webia.lip6.fr/~gallinar/gallinari/uploads/Teaching/WSDM2014-jacob.pdf)

1. **Non-transitive Hashing with Latent Similarity Componets.**
*Mingdong Ou, Peng Cui, Fei Wang, Jun Wang, Wenwu Zhu.*  KDD 2015. [paper](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/KDD-NonTransitiveHashing.pdf)

1. **GraRep: Learning Graph Representations with Global Structural Information.**
*Shaosheng Cao, Wei Lu, Qiongkai Xu.*  CIKM 2015. [paper](https://www.researchgate.net/profile/Qiongkai_Xu/publication/301417811_GraRep/links/5847ecdb08ae8e63e633b5f2/GraRep.pdf) [code](https://github.com/ShelsonCao/GraRep)

1. **LINE: Large-scale Information Network Embedding.**
*Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Qiaozhu Me.*  WWW 2015. [paper](https://arxiv.org/pdf/1503.03578.pdf) [code](https://github.com/tangjianpku/LINE)

1. **Network Representation Learning with Rich Text Information.**
*Cheng Yang, Zhiyuan Liu, Deli Zhao, Maosong Sun, Edward Y. Chang.*  IJCAI 2015. [paper](http://thunlp.org/~yangcheng/publications/ijcai15.pdf) [code](https://github.com/thunlp/tadw)

1. **PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks.**
*Jian Tang, Meng Qu, Qiaozhu Mei.*  KDD 2015. [paper](https://arxiv.org/pdf/1508.00200.pdf) [code](https://github.com/mnqu/PTE)

1. **Heterogeneous Network Embedding via Deep Architectures.**
*Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, Charu C. Aggarwal, Thomas S. Huang.* KDD 2015. [paper](http://www.ifp.illinois.edu/~chang87/papers/kdd_2015.pdf)

1. **Deep Neural Networks for Learning Graph Representations.**
*Shaosheng Cao, Wei Lu, Xiongkai Xu.* AAAI 2016. [paper](https://pdfs.semanticscholar.org/1a37/f07606d60df365d74752857e8ce909f700b3.pdf) [code](https://github.com/ShelsonCao/DNGR)

1. **Asymmetric Transitivity Preserving Graph Embedding.**
*Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, Wenwu Zhu.*  KDD 2016. [paper](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/hoppe.pdf)

1. **Revisiting Semi-supervised Learning with Graph Embeddings.**
*Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov.* ICML 2016. [paper](http://www.jmlr.org/proceedings/papers/v48/yanga16.pdf)

1. **node2vec: Scalable Feature Learning for Networks.**
*Aditya Grover, Jure Leskovec.*  KDD 2016. [paper](http://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf) [code](https://github.com/aditya-grover/node2vec)

1. **Max-Margin DeepWalk: Discriminative Learning of Network Representation.**
*Cunchao Tu, Weicheng Zhang, Zhiyuan Liu, Maosong Sun.*  IJCAI 2016. [paper](http://thunlp.org/~tcc/publications/ijcai2016_mmdw.pdf) [code](https://github.com/thunlp/mmdw)

1. **Tri-Party Deep Network Representation.**
*Shirui Pan, Jia Wu, Xingquan Zhu, Chengqi Zhang, Yang Wang.* IJCAI 2016. [paper](https://www.ijcai.org/Proceedings/16/Papers/271.pdf)

1. **Discriminative Deep RandomWalk for Network Classification.**
*Juzheng Li, Jun Zhu, Bo Zhang.* ACL 2016. [paper](http://www.aclweb.org/anthology/P16-1095)

1. **Structural Deep Network Embedding.**
*Daixin Wang, Peng Cui, Wenwu Zhu.*  KDD 2016. [paper](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/SDNE.pdf)

1. **Structural Neighborhood Based Classification of Nodes in a Network.**
*Sharad Nandanwar, M. N. Murty.* KDD 2016. [paper](http://www.kdd.org/kdd2016/papers/files/Paper_679.pdf)

1. **Community Preserving Network Embedding.**
*Xiao Wang, Peng Cui, Jing Wang, Jian Pei, Wenwu Zhu, Shiqiang Yang.* AAAI 2017. [paper](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/NE-Community.pdf)

1. **Semi-supervised Classification with Graph Convolutional Networks.**
*Thomas N. Kipf, Max Welling.* ICLR 2017. [paper](https://arxiv.org/pdf/1609.02907.pdf) [code](https://github.com/tkipf/gcn)

1. **CANE: Context-Aware Network Embedding for Relation Modeling.**
*Cunchao Tu, Han Liu, Zhiyuan Liu, Maosong Sun.* ACL 2017. [paper](http://thunlp.org/~tcc/publications/acl2017_cane.pdf) [code](https://github.com/thunlp/cane)

1. **Fast Network Embedding Enhancement via High Order Proximity Approximation.**
*Cheng Yang, Maosong Sun, Zhiyuan Liu, Cunchao Tu.* IJCAI 2017. [paper](http://thunlp.org/~tcc/publications/ijcai2017_neu.pdf) [code](https://github.com/thunlp/neu)

1. **TransNet: Translation-Based Network Representation Learning for Social Relation Extraction.**
*Cunchao Tu, Zhengyan Zhang, Zhiyuan Liu, Maosong Sun.* IJCAI 2017. [paper](http://thunlp.org/~tcc/publications/ijcai2017_transnet.pdf) [code](https://github.com/thunlp/transnet)

1. **metapath2vec: Scalable Representation Learning for Heterogeneous Networks.**
*Yuxiao Dong, Nitesh V. Chawla, Ananthram Swami.* KDD 2017. [paper](https://www3.nd.edu/~dial/publications/dong2017metapath2vec.pdf) [code](https://ericdongyx.github.io/metapath2vec/m2v.html)

1. **Learning from Labeled and Unlabeled Vertices in Networks.**
*Wei Ye, Linfei Zhou, Dominik Mautz, Claudia Plant, Christian Böhm.* KDD 2017. [paper](http://dl.acm.org/citation.cfm?id=3098142)

1. **Unsupervised Feature Selection in Signed Social Networks.**
*Kewei Cheng, Jundong Li, Huan Liu.* KDD 2017. [paper](http://www.public.asu.edu/~jundongl/paper/KDD17_SignedFS.pdf)

1. **struc2vec: Learning Node Representations from Structural Identity.**
*Leonardo F. R. Ribeiro, Pedro H. P. Saverese, Daniel R. Figueiredo.* KDD 2017. [paper](https://arxiv.org/pdf/1704.03165.pdf) [code](https://github.com/leoribeiro/struc2vec)

1. **Label Informed Attributed Network Embedding.**
*Xiao Huang, Jundong Li, Xia Hu.* WSDM 2017. [paper](http://people.tamu.edu/~xhuang/Xiao_WSDM17.pdf) [code](https://github.com/xhuang31/LANE)

1. **Accelerated Attributed Network Embedding.**
*Xiao Huang, Jundong Li, Xia Hu.* SDM 2017. [paper](http://www.public.asu.edu/~jundongl/paper/SDM17_AANE.pdf) [code](https://github.com/xhuang31/AANE_Python)

1. **Inductive Representation Learning on Large Graphs.**
*William L. Hamilton, Rex Ying, Jure Leskovec.* NIPS 2017. [paper](https://arxiv.org/pdf/1706.02216.pdf) [code](https://github.com/williamleif/GraphSAGE)

1. **Variation Autoencoder Based Network Representation Learning for Classification.**
*Hang Li, Haozheng Wang, Zhenglu Yang, Masato Odagaki.* ACL 2017. [paper](http://aclweb.org/anthology/P17-3010)

1. **Preserving Proximity and Global Ranking for Node Embedding.**
*Yi-An Lai, Chin-Chi Hsu, Wenhao Chen, Mi-Yen Yeh, Shou-De Lin.* NIPS 2017. 

1. **Learning Graph Embeddings with Embedding Propagation.**
*Alberto Garcia Duran, Mathias Niepert.* NIPS 2017. [paper](https://arxiv.org/pdf/1710.03059.pdf)

1. **Name Disambiguation in Anonymized Graphs using Network Embedding.**
*Baichuan Zhang, Mohammad Al Hasan.* CIKM 2017. [paper](https://arxiv.org/pdf/1702.02287.pdf)

1. **Enhancing the Network Embedding Quality with Structural Similarity.**
*Tianshu Lyu, Yuan Zhang, Yan Zhang.* CIKM 2017. [paper](http://www.cis.pku.edu.cn/faculty/system/zhangyan/papers/CIKM2017-lts.pdf)

1. **Attributed Signed Network Embedding.**
*Suhang Wang, Charu Aggarwal, Jiliang Tang, Huan Liu.* CIKM 2017. [paper](http://www.public.asu.edu/~swang187/publications/SNEA.pdf)

1. **Attributed Network Embedding for Learning in a Dynamic Environment.**
*Jundong Li, Harsh Dani, Xia Hu, Jiliang Tang, Yi Chang, Huan Liu.* CIKM 2017. [paper](https://arxiv.org/pdf/1706.01860.pdf)

1. **HIN2Vec: Explore Meta-paths in Heterogeneous Information Networks for Representation Learning.**
*Tao-yang Fu, Wang-Chien Lee, Zhen Lei.* CIKM 2017. [paper](http://shichuan.org/hin/topic/Embedding/2017.%20CIKM%20HIN2Vec.pdf) [code](https://github.com/csiesheep/hin2vec)

1. **From Properties to Links: Deep Network Embedding on Incomplete Graphs.**
*Dejian Yang, Senzhang Wang, Chaozhuo Li, Xiaoming Zhang, Zhoujun Li.* CIKM 2017. [paper](https://www.researchgate.net/profile/Senzhang_Wang2/publication/320885185_From_Properties_to_Links_Deep_Network_Embedding_on_Incomplete_Graphs/links/5a0a36b345851551b78d3164/From-Properties-to-Links-Deep-Network-Embedding-on-Incomplete-Graphs.pdf)

1. **An Attention-based Collaboration Framework for Multi-View Network Representation Learning.**
*Meng Qu, Jian Tang, Jingbo Shang, Xiang Ren, Ming Zhang, Jiawei Han.* CIKM 2017. [paper](https://arxiv.org/pdf/1709.06636.pdf)

1. **On Embedding Uncertain Graphs.**
*Jiafeng Hu, Reynold Cheng, Zhipeng Huang, Yixang Fang, Siqiang Luo.* CIKM 2017. [paper](http://i.cs.hku.hk/~zphuang/pub/CIKM17.pdf)

1. **Multi-view Clustering with Graph Embedding for Connectome Analysis.**
*Guixiang Ma, Lifang He, Chun-Ta Lu, Weixiang Shao, Philip S Yu, Alex D Leow, Ann B Ragin.* CIKM 2017. [paper](https://www.cs.uic.edu/~clu/doc/cikm17_mcge.pdf)

1. **Learning Node Embeddings in Interaction Graphs.**
*Yao Zhang, Yun Xiong, Xiangnan Kong, Yangyong Zhu.* CIKM 2017. [paper](https://web.cs.wpi.edu/~xkong/publications/papers/cikm17.pdf)

1. **Learning Community Embedding with Community Detection and Node Embedding on Graphs.**
*Sandro Cavallari, Vincent W. Zheng, Hongyun Cai, Kevin ChenChuan Chang, Erik Cambria.* CIKM 2017. [paper](http://sentic.net/community-embedding.pdf) [code](https://github.com/andompesta/ComE)

1. **Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec.**
*Jiezhong Qiu, Yuxiao Dong, Hao Ma, Jian Li, Kuansan Wang, Jie Tang.* WSDM 2018. [paper](https://arxiv.org/pdf/1710.02971.pdf)

1. **Exploring Expert Cognition for Attributed Network Embedding.**
*Xiao Huang, Qingquan Song, Jundong Li, Xia Ben Hu.* WSDM 2018. [paper](http://people.tamu.edu/~xhuang/Xiao_WSDM18.pdf)

1. **SHINE: Signed Heterogeneous Information Network Embedding for Sentiment Link Prediction.**
*Hongwei Wang, Fuzheng Zhang, Min Hou, Xing Xie, Minyi Guo, Qi Liu.* WSDM 2018. [paper](https://arxiv.org/pdf/1712.00732.pdf)

1. **Multidimensional Network Embedding with Hierarchical Structures.**
*Yao Ma, Zhaochun Ren, Ziheng Jiang, Jiliang Tang, Dawei Yin.* WSDM 2018. [paper](http://cse.msu.edu/~mayao4/downloads/Multidimensional_Network_Embedding_with_Hierarchical_Structure.pdf)

1. **Curriculum Learning for Heterogeneous Star Network Embedding via Deep Reinforcement Learning.**
*Meng Qu, Jian Tang, Jiawei Han.* WSDM 2018. [paper](http://delivery.acm.org/10.1145/3160000/3159711/p468-qu.pdf?ip=203.205.141.123&id=3159711&acc=ACTIVE%20SERVICE&key=39FCDE838982416F%2E39FCDE838982416F%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1519788484_7383495a5c522cbe124e62e4d768f8cc)

1. **Adversarial Network Embedding.**
*Quanyu Dai, Qiang Li, Jian Tang, Dan Wang.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.07838.pdf)

1. **COSINE: Community-Preserving Social Network Embedding from Information Diffusion Cascades.**
*Yuan Zhang, Tianshu Lyu, Yan Zhang.* AAAI 2018.

1. **Dynamic Network Embedding by Modeling Triadic Closure Process.**
*Lekui Zhou, Yang Yang, Xiang Ren, Fei Wu, Yueting Zhuang.* AAAI 2018. [paper](http://yangy.org/works/dynamictriad/dynamic_triad.pdf)

1. **Multi-facet Network Embedding: Beyond the General Solution of Detection and Representation.**
*Liang Yang, Xiaochun Cao, Yuanfang Guo.* AAAI 2018. 

1. **RSDNE: Exploring Relaxed Similarity and Dissimilarity from Completely-imbalanced Labels for Network Embedding.**
*Zheng Wang, Xiaojun Ye, Chaokun Wang, YueXin Wu, Changping Wang, Kaiwen Liang.* AAAI 2018. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16062/15722) [code](https://github.com/zhengwang100/RSDNE)

1. **Link Prediction via Subgraph Embedding-Based Convex Matrix Completion.**
*Zhu Cao, Linlin Wang, Gerard De melo.* AAAI 2018.

1. **Generative Adversarial Network based Heterogeneous Bibliographic Network Representation for Personalized Citation Recommendation.**
*J. Han, Xiaoyan Cai, Libin Yang.* AAAI 2018.

1. **DepthLGP: Learning Embeddings of Out-of-Sample Nodes in Dynamic Networks.**
*Jianxin Ma, Peng Cui, Wenwu Zhu.* AAAI 2018. [paper](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/DepthLGP.pdf)

1. **Structural Deep Embedding for Hyper-Networks.**
*Ke Tu, Peng Cui, Xiao Wang, fei Wang, Wenwu Zhu.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.10146.pdf)

1. **TIMERS: Error-Bounded SVD Restart on Dynamic Networks.**
*Ziwei Zhang, Peng Cui, Jian Pei, Xiao Wang, Wenwu Zhu.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.09541.pdf)

1. **Community Detection in Attributed Graphs: An Embedding Approach.**
*Ye Li, Chaofeng Sha, Xin Huang, Yanchun Zhang.* AAAI 2018.

1. **Bernoulli Embeddings for Graphs.**
*Vinith Misra, Sumit Bhatia.* AAAI 2018. [paper](http://sumitbhatia.net/papers/aaai18.pdf)

1. **Distance-aware DAG Embedding for Proximity Search on Heterogeneous Graphs.**
*Zemin Liu, Vincent W. Zheng, Zhou Zhao, Fanwei Zhu, Kevin Chen-Chuan Chang, Minghui Wu, Jing Ying.* AAAI 2018.

1. **GraphGAN: Graph Representation Learning with Generative Adversarial Nets.**
*Hongwei Wang, jia Wang, jialin Wang, MIAO ZHAO, Weinan Zhang, Fuzheng Zhang, Xie Xing, Minyi Guo.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.08267.pdf)

1. **HARP: Hierarchical Representation Learning for Networks.**
*Haochen Chen, Bryan Perozzi, Yifan Hu, Steven Skiena.* AAAI 2018. [paper](https://arxiv.org/pdf/1706.07845.pdf) [code](https://github.com/GTmac/HARP)

1. **Representation Learning for Scale-free Networks.**
*Rui Feng, Yang Yang, Wenjie Hu, Fei Wu, Yueting Zhuang.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.10755.pdf)

1. **Social Rank Regulated Large-scale Network Embedding.**
*Yupeng Gu, Yizhou Sun, Yanen Li, Yang Yang.* WWW 2018. [paper](http://yangy.org/works/ge/rare.pdf)

1. **Co-Regularized Deep Multi-Network Embedding.**
*Jingchao Ni, Shiyu Chang, Xiao Liu, Wei Cheng, Haifeng Chen, Dongkuan Xu, Xiang Zhang.* WWW 2018.

1. **On Exploring Semantic Meanings of Links for Embedding Social Networks.**
*Linchuan Xu, Xiaokai Wei, Jiannong Cao, Philip S Yu.* WWW 2018. [paper](http://www2.comp.polyu.edu.hk/~cslcxu/papers/IIRL.pdf)

1. **SIDE: Representation Learning in Signed Directed Networks.**
*Junghwan Kim, Haekyu Park, Ji-Eun Lee, U Kang.* WWW 2018. [paper](https://datalab.snu.ac.kr/side/resources/side.pdf)

1. **NetGAN: Generating Graphs via Random Walks.**
*Aleksandar Bojchevski, Oleksandr Shchur, Daniel Zügner, Stephan Günnemann.* ICML 2018. [paper](https://arxiv.org/abs/1803.00816)

1. **Anonymous Walk Embeddings.**
*Sergey Ivanov, Evgeny Burnaev.* ICML 2018. [paper](https://arxiv.org/pdf/1805.11921.pdf)

1. **Relational inductive bias for physical construction in humans and machines.**
*Jessica B. Hamrick, Kelsey R. Allen, Victor Bapst, Tina Zhu, Kevin R. McKee, Joshua B. Tenenbaum, Peter W. Battaglia.*  CogSci 2018. [paper](https://arxiv.org/pdf/1806.01203.pdf)

1. **Graph Networks as Learnable Physics Engines for Inference and Control.**
*Alvaro Sanchez-Gonzalez, Nicolas Heess, Jost Tobias Springenberg, Josh Merel, Martin Riedmiller, Raia Hadsell, Peter Battaglia
.* ICML 2018. [paper](https://arxiv.org/pdf/1806.01242.pdf)

1. **Arbitrary-Order Proximity Preserved Network Embedding.**
*Ziwei Zhang, Peng Cui, Xiao Wang, Jian Pei, Xuanrong Yao, Wenwu Zhu.* KDD 2018.

1. **NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks.**
*Wenchao Yu, Wei Cheng, Charu Aggarwal, Kai Zhang, Haifeng Chen, Wei Wang.* KDD 2018.

1. **Dynamic Embeddings for User Profiling in Twitter.**
*Shangsong Liang, Xiangliang Zhang, Zhaochun Ren, Evangelos Kanoulas.* KDD 2018.

1. **Deep Variational Network Embedding in Wasserstein Space.**
*Dingyuan Zhu, Peng Cui, Daixin Wang, Wenwu Zhu.* KDD 2018.

1. **Embedding Temporal Network via Neighborhood Formation.**
*Yuan Zuo, Guannan Liu, Hao Lin, Jia Guo, Xiaoqian Hu, Junjie Wu.* KDD 2018. [paper](https://zuoyuan.github.io/files/htne_kdd18.pdf)

1. **Hierarchical Taxonomy Aware Network Embedding.**
*Jianxin Ma, Peng Cui, Xiao Wang, Wenwu Zhu.* KDD 2018.

1. **Deep Recursive Network Embedding with Regular Equivalence.**
*Ke Tu, Peng Cui, Xiao Wang, Philip S. Yu, Wenwu Zhu.* KDD 2018.

1. **PME: Projected Metric Embedding on Heterogeneous Networks for Link Prediction.**
*Hongxu Chen, Hongzhi Yin, Weiqing Wang, Hao Wang, Quoc Viet Hung Nguyen, Xue Li.* KDD 2018.

1. **Content to Node: Self-translation Network Embedding.**
*Jie Liu, Zhicheng He, Lai Wei, Yalou Huang.* KDD 2018.

1. **On Interpretation of Network Embedding via Taxonomy Induction.**
*Ninghao Liu, Xiao Huang, Jundong Li, Xia Hu.* KDD 2018. [paper](http://people.tamu.edu/~xhuang/Ninghao_KDD18.pdf)

1. **Easing Embedding Learning by Comprehensive Transcription of Heterogeneous Information Networks.**
*Yu Shi, Qi Zhu, Fang Guo, Chao Zhang, Jiawei Han.* KDD 2018. [paper](http://yushi2.web.engr.illinois.edu/kdd18.pdf)

1. **Learning Structural Node Embeddings via Diffusion Wavelets.**
*Claire Donnat, Marinka Zitnik, David Hallac, Jure Leskovec.* KDD 2018. [paper](https://arxiv.org/pdf/1710.10321.pdf)

1. **Self-Paced Network Embedding.**
*Hongchang Gao, Heng Huang.* KDD 2018.

1. **Scalable Optimization for Embedding Highly-Dynamic and Recency-Sensitive Data.**
*Xumin Chen, Peng Cui, Shiqiang Yang.* KDD 2018.

1. **Expressive Graph Comparison via Multi-Scale Representations.**
*Anton Tsitsulin, Davide Mottin, Panagiotis Karras, Alexander Bronstein, Emmanuel M.* KDD 2018.

1. **Learning Deep Network Representations with Adversarially Regularized Autoencoders.**
*Wenchao Yu, Cheng Zheng, Wei Cheng, Charu Aggarwal, Dongjin Song, Bo Zong, Haifeng Chen, Wei Wang.* KDD 2018.

1. **SPARC: Self-Paced Network Representation for Few-Shot Rare Category Characterization.**
*Dawei Zhou, Jingrui He, Hongxia Yang, Wei Fan.* KDD 2018.

1. **Large-Scale Learnable Graph Convolutional Networks.**
*Hongyang Gao, Zhengyang Wang, Shuiwang Ji.* KDD 2018.

1. **Graph Convolutional Neural Networks for Web-Scale Recommender Systems.**
*Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, Jure Leskovec.* KDD 2018. [paper](https://arxiv.org/pdf/1806.01973.pdf)

1. **Semi-supervised User Geolocation via Graph Convolutional Networks.**
*Afshin Rahimi, Trevor Cohn, Timothy Baldwin.* ACL 2018. [paper](https://arxiv.org/pdf/1804.08049.pdf)

### Preprints

This section contains promising recent preprints.

1. **MolGAN: An implicit generative model for small molecular graphs.**
*Nicola De Cao, Thomas Kipf.* [paper](https://arxiv.org/pdf/1805.11973.pdf)

2. **Relational recurrent neural networks.**
*Adam Santoro, Ryan Faulkner, David Raposo, Jack Rae, Mike Chrzanowski, Theophane Weber, Daan Wierstra, Oriol Vinyals, Razvan Pascanu, Timothy Lillicrap.* [paper](https://arxiv.org/pdf/1806.01822.pdf)
