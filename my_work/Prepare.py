# -*- coding: utf-8 -*-
import numpy as np
import codecs
import re
import nltk
from nltk.tokenize import TweetTokenizer
from math import log

class prepare_tweets:
    def __init__(self,filename,IsBD=False):
        self.filename=filename
        self.IsBD=IsBD
        self.readfile()
        self.tknzr = TweetTokenizer()
        self.tknzr.reduce_len=True
        #infer spaces###################################################
        self.words = open("data/dictionary.txt").read().split()
        self.wordcost = dict((k, log((i+1)*log(len(self.words)))) for i,k in enumerate(self.words))
        self.maxword = max(len(x) for x in self.words)
        #infer spaces###################################################

    def get_train_target(self):
        return self.tag,self.tweets

    def readfile(self):
        lines = codecs.open(self.filename, 'r', 'utf-8').readlines()
        self.tweets = []
        self.tag=[]
        self.entity=[]
        for line in lines:
            line=line.strip()
            tokens=line.split('\t')
            if(self.IsBD):
                self.tweets.append(tokens[3])
                self.tag.append(tokens[2])
                self.entity.append(tokens[1])
            else:
                self.tweets.append(tokens[2])
                self.tag.append(tokens[1])
        #convert to numpy array
        self.tag=np.array(self.tag)
        self.tweets=np.array(self.tweets)

    def prepare(self):

        not_delete=(self.tweets!="Not Available")
        self.tweets=self.tweets[not_delete]
        self.tag=self.tag[not_delete]
        for i,t in enumerate(self.tweets):
            temp=self.tknzr.tokenize(self.clean_tweet(self.tweets[i]))

            if(self.IsBD):
                if(re.search("[^A-Za-z ]",self.entity[i])):
                    re.sub(re.escape(self.entity[i].lower()),"_Entity_",self.tweets[i],re.IGNORECASE)
                else:
                    for j,token in enumerate(temp):
                            if(token.lower()==self.entity[i].lower()):
                                temp[j]="_Entity_"

            temp=" ".join(temp)
            ###################################################################
            hashtags=re.findall(u"#([A-Za-z0-9_]+)",temp)
            temp+=" ."
            for hashtag in hashtags:
                if(len(hashtag)>2):
                    hashtag=hashtag.lower()
                    hashtag=re.sub('\d+','',hashtag)
                    hashtag=re.sub('_','',hashtag)
                    words= self.infer_spaces(hashtag)
                    temp+=" "+words+" ."
            ##################################################################
            # print len(temp.split())
            self.tweets[i]=temp

        self.tag[self.tag=='neutral']=2
        self.tag[self.tag=='positive']=1
        self.tag[self.tag=='negative']=0

#infer spaces###################################################
    def infer_spaces(self,s):
        """Uses dynamic programming to infer the location of spaces in a string
        without spaces."""

        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i-self.maxword):i]))
            return min((c + self.wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

        # Build the cost array.
        cost = [0]
        for i in range(1,len(s)+1):
            c,k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i>0:
            c,k = best_match(i)
            assert c == cost[i]
            if(len(s[i-k:i])>1):
                out.append(s[i-k:i])
            i -= k

        return " ".join(reversed(out))
#infer spaces###################################################

    def clean_tweet(self,tweet):
            # patterns to remove first
        pat = [\
            (u'^(RT )@([A-Za-z0-9_]+):',u''), #retweet part
            (u'http[s]?://[a-zA-Z0-9_\-./~\?=%&]+', u' _URL_ '),  # remove links
            (u'www[a-zA-Z0-9_\-?=%&/.~]+', u' _URL_ '),
#            u'\n+': u' ',                     # remove newlines
            (u'<br />', u' '),  # remove html line breaks
            (u'</?[^>]+>', u' '),  # remove html markup
#            u'http': u'',
            (u'[a-zA-Z]+\.org', u' _URL_ '),
            (u'[a-zA-Z]+\.com', u' _URL_ '),

            (u'@([A-Za-z0-9_]+)', u' _USERNAME_ '),
            # (u"#([A-Za-z0-9_]+)", u' _HASHTAG_ '),
            #&amp;
            (u'&amp;#\d+;', u' '),
            (u'&amp;', u' '),
            (u'&lt;', u' '),
            (u'&gt;', u' '),
            (u'&amp;', u' '),
	        (u'&quot;', u' '),


	    #char normallization
    	    (u"\'s", u" \'s"),
            (u"\'ve", u" \'ve"),
            (u"n\'t", u" n\'t"),
            (u"\'re", u" \'re"),
            (u"\'d", u" \'d"),
            (u"\'ll", u" \'ll"),
            #internet emotions
            (u':-\*', u' _KISS_ '), #kiss
            (u':\*', u' _KISS_ '), #kiss
            (u':-\)\*', u' _KISS_ '), #kiss
            (u'[\U0001f618\U0001f61a\U0001f617\U0001f619]',' _KISS_ '), #kiss

            (u':D', u' :D '),  #laugh
            (u':-D', u' :D '),  #laugh
            (u'[\U0001f604\U0001f603\U0001f600\U0001f602\U0001f605]',' :D '), #laugh

            (u'=\)', u' :) '),  #happy
            (u':\)', u' :) '), #happy
            (u':-\)', u' :) '), #happy
            (u'[\U0001f60a\u263a\U0001f60d]',' :) '), #happy

            (u':\(', u' :( '), #sad
            (u'=\(', u' :( '), #sad
            (u':-\(', u' :( '), #sad
            (u'[\U0001f614\U0001f612\U0001f61e\U0001f622\U0001f62d]', u' :( '), #sad

            (u';\)', u' ;) '), #winking
            (u';-\)', u' ;) '), #winking

            (u':P', u' :P '), #tongue
            (u':p', u' :P '), #tongue
            (u'[\U0001f61c\U0001f61d\U0001f61b\U0001f60b]', u' :P '), #tongue

            (u'<3', u' <3 '), #Heart
            (u':o', u' :o '), #surprise
            (u':-o', u' :o '), #surprise

            #numbers
            (u'([ \\/])', u' \\1 '), #for numbers more than two digits
            (u'([^A-Za-z0-9_])[0-9]{2,200}(\.[0-9]{1,200})?([^A-Za-z0-9_])', u"\\1_NUM_\\3"),
            (u'^[0-9]{2,200}(\.[0-9]{1,200})?([^A-Za-z0-9_])', u"_NUM_\\2"),
            (u'([^A-Za-z0-9_])[0-9]{2,200}(\.[0-9]{1,200})?$', u"\\1_NUM_"),
            # (u'[^A-Za-z0-9_][0-9]\.[0-9]+[^A-Za-z0-9_]?', u' _NUM_ '),

            #spaces
            (u'\s+', u' '),  # remove spaces
            (u'^\s+', u''),  # remove spaces start
            (u'\s+$', u''),  # remove spaces end
            (u'\.+', u'.'),  # multiple dots




        ]

        # do some subsitutions
        for k, v in pat:
            tweet = re.sub(k, v, tweet)

        # if empty string, skip
        if tweet == u'' or tweet == u' ':
            return u""
        else:
            return ("_StartToken_ "+ tweet +" _EndToken_")



    def tokenize(self,tweet):
        tokens=self.tknzr.tokenize(tweet)
        # for i in range(0,len(tokens)):
            # if(tokens[i].isupper()):
            #     tokens[i]=tokens[i].lower()

        return tokens


    def write_hashtags(self,outfile):
        hash_tags=[]

        for tweet in self.tweets:
            tokens=self.tokenize(tweet)
            for token in tokens:
                if(token.startswith('#') and len(token)>3):
                    hash_tags.append(token)

        open(outfile, 'w').write(u'\n'.join(list(set(hash_tags))).encode('utf-8'))

    def writefile(self,outfile):
        lines=[]
        for i in xrange(len(self.tweets)):
            line = u"%s\t%s\n" % (self.tweets[i],self.tag[i])
            lines.append(line)

        open(outfile, 'w').write(u''.join(lines).encode('utf-8'))


# tknzr =TweetTokenizer()
# print(tknzr.tokenize('hi my name?'))


# p.writefile("data/A_All_Prepared.tsv")

# lines=codecs.open('data/emo.txt','r').readlines()
# # print lines
#
# for l in lines:
#     l=l.decode('utf-8')
#     print l.encode('raw_unicode_escape')


