from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('Random.pkl')
onehot = joblib.load('one_joblib')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	print("$"*30)
	print(int_features)
	c = ['Species','Length1', 'Length2', 'Length3', 'Height','Width']
	df = pd.DataFrame(int_features,columns=c)
	l = onehot.transform(df.iloc[:,:1])
	c = onehot.get_feature_names_out()
	t = pd.DataFrame(l,columns=c)
	l2 = df.iloc[:,1:]
	final =pd.concat([t,l2],axis=1)
	result = model.predict(final)
	print("The Result is :",result)


	print(int_features)

	return render_template("main.html",prediction_text=" The Estimated Fish Weight is {} in gms.".format(result))


if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=8000, debug =True)
