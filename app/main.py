import re
from flask import Flask, request, json
from keras.preprocessing import text, sequence
from pathlib import Path
import tensorflow as tf
import os

# app = Flask(__name__)

# model = tf.keras.models.load_model(str(Path().absolute())+'\\api-v1')

# model = tf.keras.models.load_model('D:\Project\Python\Data\\verifIDNews-api\\api-v1')

# @app.route('/')
# def home():
#     return 'Wellcome to verifIDNews'

# @app.route('/predict' ,methods=['POST'])
def predict(user_input):
    # user_input = [request.form.get('input')]
    # model = keras.models.load_model(str(os.path.dirname(__file__))+ '/api_model.h5')
    # model = pickle.load(open(str(os.path.dirname(__file__))+ '\\api-v1', 'rb'))
    path = str(os.path.dirname(__file__)) + '\\..\\api-v1'
    model= tf.keras.models.load_model(path )
    max_features = 1000
    maxlen = 300
        
    tokenizer = text.Tokenizer(num_words = max_features)
    tokenized_test = tokenizer.texts_to_sequences(user_input)
    user_input_tokenized = sequence.pad_sequences(tokenized_test, maxlen = maxlen)
    output = model.predict(user_input_tokenized)
        
    label = ''
        
    # if output > 0.75 : 
    #     label = 'Berita Terindikasi sebagai Fakta'
    # elif output <= 0.75 and output > 0.60 : 
    #     label = 'Berita Terindikasi lemah Fakta'
    # elif output <= 0.6 and output > 0.5 :
    #     label = 'Berita perlu ditinjau ulang antara Fakta atau Hoax'
    # elif output <= 0.5 and output > 0.35 : 
    #     label = 'Berita Terindikasi Lemah sebagai Hoax'
    # elif output <= 0.35 and output > 0.15 : 
    #     label = 'Berita Terindikasi sebagai Hoax'
    # elif output <= 0.15 :
    #     label = 'Berita Terindikasi Kuat sebagai Hoax'
        
    # return bytes(output)
    # response = app.response_class(
    #     response= json.dumps({'indicator': str(output),'msg' : user_input }),
    #     status=200,
    #     mimetype='application/json'
    # )

    return output



print(predict('jij'))