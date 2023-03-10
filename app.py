from flask import Flask, render_template, request, flash
import pandas as pd
import pickle

app = Flask(__name__)
app.secret_key = 'abcdesf'

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = request.form.to_dict()
        print(data)
        df = pd.DataFrame(data, index= [0])
        df  = df.astype('int64')
        with open('normalizer_pkl' , 'rb') as f:
            norm = pickle.load(f)
        with open('model_pkl' , 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(norm.transform(df))[0]
        if y_pred == 0:
            message = 'This client will not subscribe term deposit' 
        else:
            message = 'This client will subscribe term deposit'
        flash(message)
        return render_template('index.html')
    else:
        return render_template('index.html') 


if __name__ == '__main__':
    app.run(debug = True)
    
