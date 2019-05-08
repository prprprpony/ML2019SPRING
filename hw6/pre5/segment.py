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
        ︽︿﹁﹃﹙﹛﹝（｛“‘-—_…⋯!"#$%&()*+,-./:;<=>?@[\]^_`{|}~
        \u3000\xa0\u2006 qwertyuiopasdfghjklzxcvbnm1234567890''')
print(punct)
filterpunt = lambda s: ''.join(x if x not in punct else ' ' for x in s)

stopwords=(' ')
first=True
for line in data:
    if first:
        first=False
        continue
    line=emoji.demojize(line).strip(' \n')
    line=line[line.find(',')+1:]
    line=line.lower()
    line=filterpunt(line)
    words=jieba.cut(line)
    for word in words:
        if word.isalnum():
            output.write(word+' ')
    output.write('\n')
