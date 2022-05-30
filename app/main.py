from copyreg import pickle
from flask import Flask, request, json
from keras.preprocessing import text, sequence
from keras.models import load_model
from pathlib import Path
# import tensorflow as tf
import pickle

app = Flask(__name__)

# model = tf.keras.models.load_model(str(Path().absolute())+'\\api-v1')

# model = tf.keras.models.load_model('D:\Project\Python\Data\\verifIDNews-api\\api-v1')

@app.route('/')
def home():
    return 'Wellcome to verifIDNews'

@app.route('/predict' ,methods=['POST'])
def predict():
    user_input = [request.form.get('input')]
    # model = keras.models.load_model(str(os.path.dirname(__file__))+ '/api_model.h5')
    # model = pickle.load(open(str(os.path.dirname(__file__))+ '\\..\\CNN_model', 'rb'))
    
    # model = pickle.load(open('api_model.h5', 'rb'))
    # path = str(os.path.dirname(__file__)) + '\\..\\CNN_model'
    path= str(Path().cwd()) +'\\CNN_model'
    model= load_model(path)
    
    # with open('app/api_model.h5', 'rb') as model_file:
    #     model = pickle.load(model_file)
        

    
    max_features = 10000
    maxlen = 171
    label = ''
    output = []

        
    tokenizer = text.Tokenizer(num_words = max_features)
    tokenized_test = tokenizer.texts_to_sequences(user_input)
    user_input_tokenized = sequence.pad_sequences(tokenized_test, maxlen = maxlen)
    output = model.predict(user_input_tokenized)
    
    output = output[0][0]
        
    label = ''
        
    if output > 0.75 : 
        label = 'Berita Terindikasi sebagai Fakta'
    elif output <= 0.75 and output > 0.60 : 
        label = 'Berita Terindikasi lemah Fakta'
    elif output <= 0.6 and output > 0.5 :
        label = 'Berita perlu ditinjau ulang antara Fakta atau Hoax'
    elif output <= 0.5 and output > 0.35 : 
        label = 'Berita Terindikasi Lemah sebagai Hoax'
    elif output <= 0.35 and output > 0.15 : 
        label = 'Berita Terindikasi sebagai Hoax'
    elif output <= 0.15 :
        label = 'Berita Terindikasi Kuat sebagai Hoax'
    
    response = app.response_class(
        response= json.dumps({'indicator': str(output),'msg' : label }),
        status=200,
        mimetype='application/json'
    )
    
    return response


if __name__ == "__main__":
        app.run(debug=True)
        