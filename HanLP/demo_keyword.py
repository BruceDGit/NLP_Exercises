# # -*- coding:utf-8 -*-
from pyhanlp import *


def demo_keyword(content):
    """ 关键词提取

    >>> content = (
    ...    "程序员(英文Programmer)是从事程序开发、维护的专业人员。"
    ...    "一般将程序员分为程序设计人员和程序编码人员，"
    ...    "但两者的界限并不非常清楚，特别是在中国。"
    ...    "软件从业人员分为初级程序员、高级程序员、系统"
    ...    "分析员和项目经理四大类。")
    >>> demo_keyword(content)
    [程序员, 程序, 分为, 人员, 软件]
    [从业人员分为, 特别是在中国, 程序员系统分析员, 系统分析员项目经理, 软件从业人员]
    [程序员=4.0, 人员=3.0, 程序=3.0, 分为=2.0, 编码=1.0]
    """
    # TextRankKeyword = JClass("com.hankcs.hanlp.summary.TextRankKeyword")
    keyword_list = HanLP.extractKeyword(content, 5)
    print(keyword_list)

    phrase_list = HanLP.extractPhrase(content, 5)
    print(phrase_list)

    tf_idf_counter = JClass('com.hankcs.hanlp.mining.word.TfIdfCounter')
    counter = tf_idf_counter()
    counter.add(content, content)
    counter.compute()
    print(counter.getKeywordsOf(content, 5).toString())


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    # content = (
    # "程序员(英文Programmer)是从事程序开发、维护的专业人员。"
    # "一般将程序员分为程序设计人员和程序编码人员，"
    # "但两者的界限并不非常清楚，特别是在中国。"
    # "软件从业人员分为初级程序员、高级程序员、系统"
    # "分析员和项目经理四大类。")
    # demo_keyword(content)
