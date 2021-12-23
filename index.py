#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene, threading, time,jieba
from datetime import datetime

from java.io import File
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
from bs4 import BeautifulSoup

"""
This class is loosely based on the Lucene (java implementation) demo class 
org.apache.lucene.demo.IndexFiles.  It will take a directory as an argument
and will index all of the files in that directory and downward recursively.
It will index on the file path, the file name and the file contents.  The
resulting Lucene index will be placed in the current directory and called
'index'.
"""


class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)


class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, root, storeDir):

        if not os.path.exists(storeDir):
            os.mkdir(storeDir)

        store = SimpleFSDirectory(File(storeDir))
        analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
        analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        config = IndexWriterConfig(Version.LUCENE_CURRENT, analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)

        self.indexDocs(root, writer)
        ticker = Ticker()
        print 'commit index',
        threading.Thread(target=ticker.run).start()
        writer.commit()
        writer.close()
        ticker.tick = False
        print 'done'

    def indexDocs(self, root, writer):

        t1 = FieldType()
        t1.setIndexed(False)
        t1.setStored(True)
        t1.setTokenized(False)

        t2 = FieldType()
        t2.setIndexed(True)
        t2.setStored(True)
        t2.setTokenized(True)
        t2.setIndexOptions(FieldInfo.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
        path='/home/tanshuo/Downloads/dazuoye/data/suning'
        path02='/home/tanshuo/Downloads/dazuoye/data/jingdong'
        l=os.listdir(path)
        for file in l:
            path1=path+"/"+file
            sets=os.listdir(path1)
            for set in sets:
                path2=path1+"/"+set
                data=os.listdir(path2)
                for x in data:
                    if(x=="1.txt"):
                        with open(path2+"/"+x) as f:
                            head=f.readlines()
                        head=head[0].strip("\n")
                #lujing=path2
                lujing="/suning/"+file+"/"+set
                print "adding suning",head
                head=' '.join(jieba.cut(head))
                doc=Document()
                doc.add(Field("head",head,t2))
                doc.add(Field("lujing",lujing,t1))
                writer.addDocument(doc)
        l2=os.listdir(path02)
        for file2 in l2:
            wenjianjia=os.listdir(path02+"/"+file2)
            lujing2=path02+"/"+file2+"/"+wenjianjia[0]
            lujing3="/jingdong/"+file2+"/"+wenjianjia[0]
            with open(lujing2) as f:
                result=f.readlines()
            head=result[2].strip('\n').decode('gbk','ignore')
            print "adding jingdong",head
            head=' '.join(jieba.cut(head))
            doc=Document()
            doc.add(Field("head",head,t2))
            doc.add(Field("lujing",lujing3,t1))
            writer.addDocument(doc)
            


if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print 'lucene', lucene.VERSION
    start = datetime.now()
    try:
        IndexFiles('html', "index-n")
        end = datetime.now()
        print end - start
    except Exception, e:
        print "Failed: ", e
        raise e
