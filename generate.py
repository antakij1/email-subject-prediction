import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet

enron = pd.read_pickle("./enron_cleaned.pkl")

train, test = train_test_split(enron, test_size=0.05)

cvec = CountVectorizer(analyzer='word',
                      ngram_range=(1,3),
                      max_features=None,
                      stop_words='english',
                      min_df=2,
                      max_df=0.95)

count_matrix = cvec.fit_transform(train.Complete)
km_count = KMeans(n_clusters=4)
km_count.fit(count_matrix)
order_centroids = km_count.cluster_centers_.argsort()[:, ::-1]
terms = cvec.get_feature_names()
for i in range(4):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print()
vec = cvec.fit(train.Complete)
new_data = vec.transform(test.Complete)
predictions = km_count.predict(new_data)

chunkGram = r"Chunk: {(<JJ>|<JJR>|<JJS>)*(<NN>|<NNS>)+}"
chunkParser = nltk.RegexpParser(chunkGram)

i=0
temp=[]
for string in test['content']:
    max_score = 0
    best_scoring = 0
    max_length = 0
    best_lengthwise = 0
    for st in sent_tokenize(string):
        tagged_words = nltk.pos_tag(word_tokenize(st))
        chunked = chunkParser.parse(tagged_words)
        for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
            leaves = subtree.leaves()
            head, _ = leaves[len(leaves) - 1]
            if (len(head) < 3 or len(wordnet.synsets(head)) == 0):
                continue
            if(len(leaves) > max_length):
                best_lengthwise = leaves
                max_length = len(leaves)
            word1 = wordnet.synsets(head)[0]
            for ind in order_centroids[predictions[i], :10]:
                if(terms[ind] == 'enron' ):
                    word2 = wordnet.synsets('company')[0]
                elif(len(wordnet.synsets(terms[ind])) == 0):
                    continue
                else:
                    word2 = wordnet.synsets(terms[ind])[0]
                if(word1.wup_similarity(word2) == None):
                    continue
                elif(word1.wup_similarity(word2) > max_score):
                    best_scoring = leaves
                    max_score = word1.wup_similarity(word2)
    subject = []
    if type(best_lengthwise) is not int:
        for tup in best_lengthwise:
            subject.append(tup[0])
        if(best_lengthwise != best_scoring):
            subject.append("and")
            for wd in best_scoring:
                subject.append(wd[0])
        sub_line = ' '.join(subject)
        temp.append([string,sub_line,test.iloc[i, 4]])
    i += 1

generated = pd.DataFrame(data=temp, columns = ['message', 'generated_subject', 'real_subject'])
pickle.dump(generated, open('C:/Users/Joe/Desktop/email-subject-prediction/generated.pkl','wb'))
print(generated.head())