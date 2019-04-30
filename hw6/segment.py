#!/usr/bin/env python3
import os,sys
import emoji
import jieba
data=open(sys.argv[1],'r')
jieba.set_dictionary(sys.argv[2])
output=open(sys.argv[3],'w')
punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
        ﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
        々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
        ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…⋯''')
print(punct)
filterpunt = lambda s: ''.join(x if x not in punct else ' ' for x in s)

stopwords=(' ')
for line in data:
    line=emoji.demojize(line).strip(' \n')
    line=line[line.find(',')+1:]
    line=filterpunt(line)
    words=jieba.cut(line)
    for word in words:
        if word not in stopwords:
            output.write(word+' ')
    output.write('\n')
