## 利用icwb2-data工具包对中文分词预测结果进行评估

### 工具包下载地址

http://sighan.cs.uchicago.edu/bakeoff2005/

- 安装ActivePerl，先能识别后缀名为.pl的文件

- 将ActivePerl下的bin目录加到环境变量中。

- 安装diffUtil工具包。
  http://gnuwin32.sourceforge.net/packages/diffutils.htm
  直接下载setup格式的安装包（注意尽量安装在没有空格和中文的路径下）

- 修改icwb2-data/scripts/score脚本
  把46行的代码修改成：

  ```
  $diff = “E:/GnuWin32/bin/diff”;(该目录为安装目录)
  ```

  把52,53行的代码修改成：(注意E:/GnuWin32/tmp目录要存在)

  ```
  $tmp1=“E:/GnuWin32/tmp/comp01$$”;$tmp2=“E:/GnuWin32/tmp/comp02$$”;
  ```

- 将diffUtil下的bin目录加到环境变量中。

### 运行

- 以下利用其自带的中文分词工具进行说明。在scripts目录里包含一个基于最大匹配法的中文分词器mwseg.pl，以北京大学提供的人民日报语料库为例，用法如下：

  - 分词：

    ```powershell
    # 打开cmd.exe 切换到 scripts 目录下（win下切换目录需要先切换盘符，再cd到相应目录下，dir可以查看目录下文件列表，如切换至D盘：）
    C:\Users\Administrator>D:   ->   D:\>
    # 执行分词
    perl ./mwseg.pl ../gold/pku_training_words.txt < ../testing/pku_test.txt > pku_test_seg.txt
    ```

    其中第一个参数需提供一个词表文件pku_training_word.txt，输入为pku\_test.txt，输出为pku\_test\_seg.txt。

  - 利用score评分：

    ```powershell
    perl score ..\gold\pku_training_words.txt ..\gold\pku_test_gold.txt pku_test_setg.txt > score.txt 2> error.txt
    ```

    评分脚本“score”是用来比较两个分词文件的，需要三个参数：

    1. 训练集词表（The training set word list）
    2. “黄金”标准分词文件（The gold standard segmentation）
    3. 测试集的切分文件（The segmented test file）

    而score.txt则包含了详细的评分结果，不仅有总的评分结果，还包括每一句的对比结果。这里只看最后的总评结果（别的分词器的）：

    ```
    = SUMMARY:
    === TOTAL INSERTIONS:	9274
    === TOTAL DELETIONS:	1365
    === TOTAL SUBSTITUTIONS:	8377
    === TOTAL NCHANGE:	19016
    === TOTAL TRUE WORD COUNT:	104372
    === TOTAL TEST WORD COUNT:	112281
    === TOTAL TRUE WORDS RECALL:	0.907
    === TOTAL TEST WORDS PRECISION:	0.843
    === F MEASURE:	0.874
    === OOV Rate:	0.058
    === OOV Recall Rate:	0.069
    === IV Recall Rate:	0.958
    ###	pku_test_seg.txt	9274	1365	8377	19016	104372	112281	0.907	0.843 0.874	0.058	0.069	0.958
    ```

    说明这个中文分词器在北大提供的语料库上的测试结果是：召回率为90.7%，准确率为84.3%，F值为87.4%等。

  

  





# 以下内容暂未做验证（个人笔记）

```
//进入前面预测、测试的生成文件目录下perl score 训练文件 测试文件名 输出结果名 >输出到的文件名//例如:perl score maxtrain11.txt maxtest11.txt maxoutput.txt > o.txt
```

在输出的文件“o.txt”下即可查看正确率、召回率等。

