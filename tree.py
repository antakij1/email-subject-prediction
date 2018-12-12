import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
import re
from random import *
from sklearn.tree import DecisionTreeClassifier

good=[]

enron = pd.read_pickle("./enron_cleaned.pkl")


pd.set_option('display.max_colwidth', 2000)

def lazy_clean(data):
    for string in data['content']:
        if(type(string) == str):
            if("Original" not in string and "Forwarded" not in string and "@" not in string):
                string = re.sub(r'<Embedded Picture .Device Independent Bitmap.>', ' ', string)
                string = re.sub(' *\r|\n * ', '', string)
                string = re.sub(r'/', '', string)
                string = re.sub(r'\\', '', string)
                string = re.sub(r'_', ' ', string)
                string = re.sub(r'=', ' ', string)
                good.append(string)
    return pd.DataFrame(data = good, columns = ['content'])
enron = lazy_clean(enron)
enron_train_half, enron_test_half = train_test_split(enron, test_size=0.25)

enron_train_half = enron_train_half.iloc[:100]
enron_test_half = enron_test_half.iloc[:30]
print(enron_train_half.shape)
print(enron_test_half.shape)
pickle.dump(enron_train_half, open("C:\\Users\\Joe\\Desktop\\email-subject-prediction\\enron_train_half.pkl", "wb"))
pickle.dump(enron_test_half, open("C:\\Users\\Joe\\Desktop\\email-subject-prediction\\enron_test_half.pkl", "wb"))


tfvec = TfidfVectorizer(ngram_range=(1, 3),
                       max_features=None,
                       stop_words='english',
                       min_df=2,
                       max_df=0.95)

tf_matrix = tfvec.fit_transform(enron_train_half.content)
km_tf = KMeans(n_clusters=4)
km_tf.fit(tf_matrix)
order_centroids = km_tf.cluster_centers_.argsort()[:, ::-1]
terms = tfvec.get_feature_names()
for i in range(4):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print()

another_vec = tfvec.fit(enron_train_half.content)
more_new_data = another_vec.transform(enron_test_half.content)
predictions_tf = km_tf.predict(more_new_data)

def grade(pred_arr, d):
    data = [] #NP, first occurence, sentence position, document position, length in words,length in characters, maximum affinity to cluster, salience(Y/N)

    chunkGram = r"Chunk: {(<JJ>|<JJR>|<JJS>|<CD>)*(<NN>|<NNS>|<NNP>|<NNPS>)+}"
    chunkParser = nltk.RegexpParser(chunkGram)
    string_counter = 0
    for string in d['content']:
        sent_counter = 0
        for st in sent_tokenize(string):
            word_counter = 0
            tagged_words = nltk.pos_tag(word_tokenize(st))
            tagged_words = [x + (word_counter,) for x in tagged_words]
            chunked = chunkParser.parse(tagged_words)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                temp=[]
                leaves = subtree.leaves()
                head, _, sent_pos = leaves[len(leaves) - 1]
                temp.append(string.find(head))
                temp.append(sent_pos)
                temp.append(sent_counter)
                temp.append(len(leaves))
                char_counter = 0
                for part in leaves:
                    char_counter += len(part[0])
                temp.append(char_counter)
                if(len(head) < 3):
                    temp.append(0)
                    continue
                elif(head == 'enron' or head == 'dynegy'):
                    word1 = wordnet.synsets('company')[0]
                elif(head == 'bush'):
                    word1 = wordnet.synsets('president')[0]
                elif(len(wordnet.synsets(head)) == 0):
                    temp.append(0)
                    continue
                else:
                    word1 = wordnet.synsets(head)[0]
                max_score = 0
                for ind in order_centroids[pred_arr[i], :10]:
                    if(type(terms[ind]) is not str):
                        continue
                    else:
                        last = terms[ind].split()
                        last = last[len(last)-1]
                    if(last == 'enron' or last == 'dynegy'):
                        word2 = wordnet.synsets('company')[0]
                    elif(last == 'bush'):
                        word2 = wordnet.synsets('president')[0]
                    elif(len(wordnet.synsets(last)) == 0):
                        continue
                    else:
                        word2 = wordnet.synsets(last)[0]
                    if (word1.wup_similarity(word2) == None):
                        continue
                    elif (word1.wup_similarity(word2) > max_score):
                        max_score = word1.wup_similarity(word2)
                temp.append(max_score)
                if randrange(30) <= 2:
                    temp.append(1)
                temp.append(0)
                data.append(temp)
            sent_counter += 1
        string_counter += 1
    return pd.DataFrame(data = data, columns=['NP', 'first_occurence', 'sentence_position', 'document position', 'length_in_words', 'length_in_characters', 'maximum_affinity_to_cluster', 'salience'])

tree_data = grade(predictions_tf, enron_test_half)
pickle.dump(tree_data, open("C:\\Users\\Joe\\Desktop\\email-subject-prediction\\tree_data.pkl", "wb"))

tree_data = pd.read_pickle("./tree_data.pkl")

x = tree_data.drop(["salient"], axis=1)
y = tree_data["salient"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

tree_class = DecisionTreeClassifier()
tree_class.fit(list(x_train),y_train) #error is here

