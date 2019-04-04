# 基于tf-idf的中文问答机器人

环境配置
    
    Python版本为3.6
    gensim
    jieba
    NLTK
    
目录说明  
  
     QAdemo_base1文件夹包含的是完成整个问答demo流程所需要的脚本。
         stopwordList文件夹是停用词的数据
              stopword.txt 扩展的停用词表
         userdict文件夹是外部词的数据
              userdict.txt 自定义的外部词
         jiebaSegment.py 
              封装好的结巴分词，支持多种切分模式
         sentence.py
              封装好的读取句子的类
         sentenceSimilarity.py
              支持tf-idf，lda，lsa等多个模型
         tmodel.py
              直接利用模型的问答
         tmode2.py  
              加入倒排索引后的问答
    
结果展示：  
![chat]( https://github.com/WenRichard/QAmodel-for-Retrievalchatbot/raw/master/QAdemo_base1/image/chat.png "百度AnyQ Framework")
