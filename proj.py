from flask import Flask,render_template,request
#from flask_restful import reqparse,abort,Api,Resource
import pickle
import numpy as np
#from model import NLPModel
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

#prediction function
def ValuePredictor(reviews):
    model = pickle.load(open('model.pkl','rb'))
    #query = model.vectorizer_transform(np.array([statement]))
    
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    english_stop_words = stopwords.words('english')
    #review = re.sub('[^a-zA-Z]',' ', to_predict)
    #review= review.lower()
    #review= review.split()
    #ps = PorterStemmer()
    #review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #review = ' '.join(review)
    #corpus.append(review)
    removed_stop_words = []
    for review in reviews:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    '''
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=1802180)
    X_test = hv.transform(corpus)
    print(X_test)
    '''
    
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemm_review = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in removed_stop_words]
    '''
    print(lemm_review)
    
    from sklearn.feature_extraction.text import CountVectorizer
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    ngram_vectorizer.fit(lemm_review)
    X_test = ngram_vectorizer.transform(lemm_review)
    '''
    #reviews = reviews.reshape(1,-1)
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=1802180)
    X_test = hv.transform(lemm_review)
    
    print(X_test)
    print(reviews)
    result = model.predict(X_test)
    print(result)
    return result

@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict=request.form['statement']
        mname = request.form['mname']
        test = np.array([to_predict])
        result=ValuePredictor([to_predict])
        print(result)
        prediction=""
        
        if result[len(result)-1]==0:
            prediction="Negative"
        else:
            prediction="Posititve"
        
        del result
    return render_template('result.html',mname=mname,prediction=prediction)

if __name__ =="__main__":
    app.run()
    
