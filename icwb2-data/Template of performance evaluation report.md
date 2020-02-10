## 与代表性分词软件的性能对比

我们选择[LTP-3.2.0 ](https://github.com/HIT-SCIR/ltp)、[ICTCLAS(2015版) ](https://github.com/NLPIR-team/NLPIR/tree/master/NLPIR-ICTCLAS)、[jieba(C++版)](https://github.com/yanyiwu/cppjieba)等国内具代表性的分词软件与THULAC做性能比较。我们选择Windows作为测试环境，根据第二届国际汉语分词测评（[The Second International Chinese Word Segmentation Bakeoff](http://sighan.cs.uchicago.edu/bakeoff2005/))发布的国际中文分词测评标准，对不同软件进行了速度和准确率测试。

在第二届国际汉语分词测评中，共有四家单位提供的测试语料（Academia Sinica、 City University 、Peking University 、Microsoft Research）, 在评测提供的资源[icwb2-data](http://sighan.cs.uchicago.edu/bakeoff2005/)中包含了来自这四家单位的训练集（training）、测试集（testing）, 以及根据各自分词标准而提供的相应测试集的标准答案（icwb2-data/scripts/gold）．在icwb2-data/scripts目录下含有对分词进行自动评分的perl脚本score。

我们在统一测试环境下，对上述流行分词软件和THULAC进行了测试，使用的模型为各分词软件自带模型。THULAC使用的是随软件提供的简单模型Model_1。评测环境为 Intel Core i5 2.4 GHz 评测结果如下：

msr_test（560KB）

| Algorithm       | Time  | Precision | Recall | F-Measure |
| --------------- | ----- | --------- | ------ | --------- |
| LTP-3.2.0       | 3.21s | 0.867     | 0.896  | 0.881     |
| ICTCLAS(2015版) | 0.55s | 0.869     | 0.914  | 0.891     |
| jieba(C++版)    | 0.26s | 0.814     | 0.809  | 0.811     |
| THULAC_lite     | 0.62s | 0.877     | 0.899  | 0.888     |

pku_test（510KB）

| Algorithm       | Time  | Precision | Recall | F-Measure |
| --------------- | ----- | --------- | ------ | --------- |
| LTP-3.2.0       | 3.83s | 0.960     | 0.947  | 0.953     |
| ICTCLAS(2015版) | 0.53s | 0.939     | 0.944  | 0.941     |
| jieba(C++版)    | 0.23s | 0.850     | 0.784  | 0.816     |
| THULAC_lite     | 0.51s | 0.944     | 0.908  | 0.926     |

除了以上在标准测试集上的评测，我们也对各个分词工具在大数据上的速度进行了评测，结果如下：

CNKI_journal.txt（51 MB）

| Algorithm       | Time     | Speed       |
| --------------- | -------- | ----------- |
| LTP-3.2.0       | 348.624s | 149.80KB/s  |
| ICTCLAS(2015版) | 106.461s | 490.59KB/s  |
| jieba(C++版)    | 22.558s  | 2314.89KB/s |
| THULAC_lite     | 42.625s  | 1221.05KB/s |