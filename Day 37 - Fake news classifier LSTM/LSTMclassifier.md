## Fake News Classifier Using LSTM

In this project, I would try to implement LSTM (Long short-term memory) to classify fake news based on the `title` of the news. LSTM is an artificial neural network used in the field of deep learning and is heavily used in NLP based tasks.

The dataset has been taken from kaggle: https://www.kaggle.com/c/fake-news/data#


```python
# Importing Libraries
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dropout
```


```python
# Checking tensorflow version
tf.__version__
```




    '2.4.1'




```python
# Read the dataset
df=pd.read_csv('train.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>author</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>Darrell Lucus</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>
      <td>Daniel J. Flynn</td>
      <td>Ever get the feeling your life circles the rou...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Why the Truth Might Get You Fired</td>
      <td>Consortiumnews.com</td>
      <td>Why the Truth Might Get You Fired October 29, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>15 Civilians Killed In Single US Airstrike Hav...</td>
      <td>Jessica Purkiss</td>
      <td>Videos 15 Civilians Killed In Single US Airstr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Iranian woman jailed for fictional unpublished...</td>
      <td>Howard Portnoy</td>
      <td>Print \nAn Iranian woman has been sentenced to...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop Nan Values
df=df.dropna()
df.reset_index(inplace=True, drop = True)
```


```python
# Get the Independent Features
X=df.drop('label',axis=1)
```


```python
# Get the Dependent features
y=df['label']
```


```python
X.shape
```




    (18285, 4)




```python
y.shape
```




    (18285,)




```python
# Vocabulary size (size of the vocabulary dictionary)
voc_size=5000
```

### One-hot Representation

The first step here would be to convert the collection of words into a one-hot representation.
But before that, we need to make sure the data is pre-processed.


```python
# creating a copy of X
messages=X.copy()
```


```python
messages['title'][0]
```




    'House Dem Aide: We Didn’t Even See Comey’s Letter Until Jason Chaffetz Tweeted It'




```python
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/badvendetta/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```python
# Dataset Preprocessing

ps = PorterStemmer()

corpus = []

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i]) # removing unnecessary characters
    review = review.lower() # converting to lower case
    review = review.split() # removing whitespaces
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # stemming the words
    review = ' '.join(review)
    corpus.append(review)
```


```python
corpus[0]
```




    'hous dem aid even see comey letter jason chaffetz tweet'




```python
# Finally, converting to one-hot representation
onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr[0]
```




    [159, 1465, 2227, 692, 4245, 3604, 1480, 4713, 4721, 4983]



### Embedding Representation


```python
sent_length=20 # max length of sentence we are allowing
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length) # pre-padding used here
print(embedded_docs)
```

    [[   0    0    0 ... 4713 4721 4983]
     [   0    0    0 ...  506 4107 4484]
     [   0    0    0 ... 2066 1239  865]
     ...
     [   0    0    0 ... 2747 1218 1903]
     [   0    0    0 ... 2131  977 3240]
     [   0    0    0 ... 4104 3760 4027]]



```python
embedded_docs[0]
```




    array([   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  159,
           1465, 2227,  692, 4245, 3604, 1480, 4713, 4721, 4983], dtype=int32)



### Building the model


```python
embedding_vector_features=40 # Number of features in the Feature representation
model=Sequential() # Building a sequential model
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length)) # Adding an embedding layer
model.add(LSTM(100)) # Adding an LSTM layer with 100 neurons
model.add(Dense(1,activation='sigmoid')) # Adding a dense layer with sigmoid activation function
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) # Compiling the model
print(model.summary())
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 20, 40)            200000    
    _________________________________________________________________
    lstm (LSTM)                  (None, 100)               56400     
    _________________________________________________________________
    dense (Dense)                (None, 1)                 101       
    =================================================================
    Total params: 256,501
    Trainable params: 256,501
    Non-trainable params: 0
    _________________________________________________________________
    None



```python
len(embedded_docs),y.shape
```




    (18285, (18285,))




```python
X_final=np.array(embedded_docs)
y_final=np.array(y)
```


```python
X_final.shape,y_final.shape
```




    ((18285, 20), (18285,))




```python
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.30, random_state=42)
```

### Model Training


```python
# Model training with 10 epochs (Will take some time!)
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
```

    Epoch 1/10
    200/200 [==============================] - 10s 36ms/step - loss: 0.4795 - accuracy: 0.7414 - val_loss: 0.2099 - val_accuracy: 0.9120
    Epoch 2/10
    200/200 [==============================] - 7s 36ms/step - loss: 0.1406 - accuracy: 0.9469 - val_loss: 0.1854 - val_accuracy: 0.9223
    Epoch 3/10
    200/200 [==============================] - 8s 39ms/step - loss: 0.0903 - accuracy: 0.9675 - val_loss: 0.1973 - val_accuracy: 0.9231
    Epoch 4/10
    200/200 [==============================] - 7s 33ms/step - loss: 0.0636 - accuracy: 0.9805 - val_loss: 0.2289 - val_accuracy: 0.9185
    Epoch 5/10
    200/200 [==============================] - 6s 32ms/step - loss: 0.0401 - accuracy: 0.9891 - val_loss: 0.3322 - val_accuracy: 0.9185
    Epoch 6/10
    200/200 [==============================] - 7s 35ms/step - loss: 0.0236 - accuracy: 0.9940 - val_loss: 0.3528 - val_accuracy: 0.9172
    Epoch 7/10
    200/200 [==============================] - 8s 38ms/step - loss: 0.0167 - accuracy: 0.9954 - val_loss: 0.3183 - val_accuracy: 0.9158
    Epoch 8/10
    200/200 [==============================] - 6s 32ms/step - loss: 0.0132 - accuracy: 0.9972 - val_loss: 0.4026 - val_accuracy: 0.9109
    Epoch 9/10
    200/200 [==============================] - 6s 31ms/step - loss: 0.0072 - accuracy: 0.9984 - val_loss: 0.4734 - val_accuracy: 0.9079
    Epoch 10/10
    200/200 [==============================] - 6s 32ms/step - loss: 0.0070 - accuracy: 0.9981 - val_loss: 0.4943 - val_accuracy: 0.9156





    <tensorflow.python.keras.callbacks.History at 0x7f857eeb1280>



### Performance Metrics And Accuracy


```python
y_pred=model.predict_classes(X_test)

accuracy_score(y_test,y_pred)
```

    /Users/badvendetta/anaconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/sequential.py:450: UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01. Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
      warnings.warn('`model.predict_classes()` is deprecated and '





    0.9156033539919796




```python
confusion_matrix(y_test,y_pred)
```




    array([[2807,  300],
           [ 163, 2216]])



So we got a pretty good accuracy score of 0.91 which is much better than the Logistic regression score of 0.81 which I previously trained. To further improve the score, I tried adding two dropout layers to the sequential model as shown below.

## Summary

We were able to classify the fake news titles with a pretty good accuracy of 0.91 on the test data by using LSTM. We could also try playing around with some hyperparameters to improve the model accuracy such as vocab size, embedding vector features, etc. We could also try adding dropout layers to our sequential model to imporove the model performance further.
