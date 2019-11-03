import pandas as pd
import jieba
import pkuseg

data = pd.read_csv('paper1.csv', encoding='utf-8', header=0)

data['cut'] = data['text'].apply(lambda x : list(set(jieba.cut(x))))

stopwords = pd.read_csv('CNStopWords.csv', encoding='utf-8', header=0)

stop_list = stopwords['StopWords'].tolist()

# 使用结巴分词
data['cut'] = data['text'].apply(lambda x : [i for i in jieba.cut(x) if i not in stop_list])

#print(data.head())

# 使用北京大学分词库
seg = pkuseg.pkuseg()
data_pku = pd.read_csv('paper1.csv', encoding='utf-8', header=0)
data_pku['cut'] = data_pku['text'].apply(lambda x : [i for i in seg.cut(x) if i not in stop_list])

#print(data_pku.head())

#合并分词
words = []
for content in data['cut']:
    words.extend(content)

#方式1： 创建分词框
corpus = pd.DataFrame(words, columns=['word'])
corpus['cnt'] = 1

#统计分词
g = corpus.groupby(['word']).agg({'cnt':'count'}).sort_values('cnt', ascending=False)

print(g.head(10))

#方式2：使用相关库
from collections import Counter
from pprint import pprint

counter = Counter(words)
pprint(counter.most_common(10))