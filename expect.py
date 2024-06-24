from flask import Flask,request,render_template
import pickle
import numpy as np  

app = Flask(__name__)

def load_model():
    with open('expectancy.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
scaler = data['scaler']

@app.route('/')
def homepage():
    return render_template('expect.html')

@app.route('/submit',methods = ['POST'])
def predict():
    a = request.form.get('adult mortality')
    b = request.form.get('hiv/aids')
    c = request.form.get('thinness 1-19 years')
    d = request.form.get('thinness 5-9 years')
    e = request.form.get('income composition of resources')
    f = request.form.get('schooling')
    
    x = np.array([[a,b,c,d,e,f]])
    x = scaler.transform(x)
    prediction = model.predict(x)
    
    msg = f"Predicted life expectancy is: {np.round(prediction,2)}"
    
    return render_template('expect.html',text=msg)

if __name__ == '__main__':
    app.run(host="0.0.0.0")