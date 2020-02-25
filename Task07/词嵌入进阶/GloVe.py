#GloVe模型
#对word2vec改进
#子词嵌入（subword embedding）：FastText 以固定大小的 n-gram 形式将单词更细致地表示为了子词的集合，而 BPE (byte pair encoding) 算法则能根据语料库的统计信息，自动且动态地生成高频子词的集合；
#GloVe 全局向量的词嵌入: 通过等价转换 Word2Vec 模型的条件概率公式，我们可以得到一个全局的损失函数表达，并在此基础上进一步优化模型。
import torch
import torchtext.vocab as vocab

#载入已训练好的词向量，可直接在网上获取
print([key for key in vocab.pretrained_aliases.keys() if "glove" in key])
cache_dir = "/home/kesci/input/GloVe6B5429"
glove = vocab.GloVe(name='6B', dim=50, cache=cache_dir)
print("一共包含%d个词。" % len(glove.stoi))
print(glove.stoi['beautiful'], glove.itos[3366])

#近义词
#向量空间的余弦相似性主要由于对于不同方向的标准向量代表不同的含义，而长度代表学习出的含义强度，由此可以使用k近邻方法寻找近义词
def knn(W, x, k):
    '''
    @params:
        W: 所有向量的集合
        x: 给定向量
        k: 查询的数量
    @outputs:
        topk: 余弦相似性最大k个的下标
        [...]: 余弦相似度
    '''
    cos = torch.matmul(W, x.view((-1,))) / (
        (torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())
    _, topk = torch.topk(cos, k=k)
    topk = topk.cpu().numpy()
    return topk, [cos[i].item() for i in topk]

def get_similar_tokens(query_token, k, embed):
    '''
    @params:
        query_token: 给定的单词
        k: 所需近义词的个数
        embed: 预训练词向量
    '''
    topk, cos = knn(embed.vectors,
                    embed.vectors[embed.stoi[query_token]], k+1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
        print('cosine sim=%.3f: %s' % (c, (embed.itos[i])))

get_similar_tokens('chip', 3, glove)
get_similar_tokens('baby', 3, glove)
get_similar_tokens('beautiful', 3, glove)

#类比词
#通过叠加向量使原本输入类比的关系类比到新的需要类比的词上
def get_analogy(token_a, token_b, token_c, embed):
    '''
    @params:
        token_a: 词a
        token_b: 词b
        token_c: 词c
        embed: 预训练词向量
    @outputs:
        res: 类比词d
    '''
    vecs = [embed.vectors[embed.stoi[t]] 
                for t in [token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.vectors, x, 1)
    res = embed.itos[topk[0]]
    return res

get_analogy('man', 'woman', 'son', glove)#叠加的向量可理解为性别转换

get_analogy('beijing', 'china', 'tokyo', glove)#叠加的向量可理解为首都->国家的映射

get_analogy('bad', 'worst', 'big', glove)

get_analogy('do', 'did', 'go', glove)