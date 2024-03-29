from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd



class Similarity:
    def __init__(self, emails):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        self.emails = emails
        self.emails.drop(self.emails.query("To == '' | From == '' | Body == ''").index, inplace=True)
        self.vect = TfidfVectorizer(stop_words='english', max_df=0.50, min_df=2)
        self.X = self.vect.fit_transform(self.emails.Body.astype('U'))
        self.X_dense = self.X.todense()
        self.coords = PCA(n_components = 2).fit_transform(np.asarray(self.X_dense))
        self.data_frame = pd.DataFrame(self.coords, columns =['x', 'y'])
        self.data_frame = self.data_frame.dropna()
        self.features = self.vect.get_feature_names_out()
        self.n_clusters = 3
        self.clf = KMeans(n_clusters=self.n_clusters, max_iter=100, init='k-means++', n_init=1)
        self.labels = self.clf.fit_predict(self.X)
        self.stopwords = list(ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient']))
        self.vec = TfidfVectorizer(analyzer='word', stop_words=self.stopwords, max_df=0.3, min_df=2)
        self.vec_train = self.vec.fit_transform(self.emails.Body.astype('U'))
        self.cosine_sim = linear_kernel(self.vec_train[0:1], self.vec_train).flatten()


    def rec_email_process(self,rec_emailFrom,rec_emailSub,rec_emailbody):
        self.emails_after_rec = self.emails
        self.emails = self.emails_after_rec
        self.rec_email_from,self.rec_email_subject,self.rec_email_body = rec_emailFrom,rec_emailSub,rec_emailbody
        self.emails_after_rec.loc[len(self.emails_after_rec.index)] = {"Body":self.rec_email_body,"Subject":self.rec_email_subject,"From":self.rec_email_from,"To":["Myself"]}
        self.vec_rec = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)
        self.vec_train_rec = self.vec.fit_transform(self.emails_after_rec.Body.astype('U'))
        self.final_labels = self.clf.fit_predict(self.vec_train_rec)
        self.final_labels_list = self.final_labels.tolist()
        self.cosine_sim_rec = linear_kernel(self.vec_train_rec[-1], self.vec_train_rec).flatten()
        similarity_value = self.cosine_sim_rec
        similarity_value = similarity_value.tolist()
        similarity_value.remove(max(similarity_value))
        self.related_rec_email_indices = self.cosine_sim_rec.argsort()[:-10:-1]
        print("Similarity: ", max(similarity_value))
        first_rec_email_index = self.related_rec_email_indices[1]
        return self.emails_after_rec.Body[first_rec_email_index], max(similarity_value)