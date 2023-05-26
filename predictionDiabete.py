from distutils.log import debug # va aidé à 
from flask import Flask,request,jsonify, render_template,redirect,url_for
import sklearn
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app=Flask(__name__) # pour preparer l'environnement
models=pickle.load(open('ModelDiab.pkl','rb'))# pour reconnaitre les elements du target

@app.route('/') # pour avoir la route
def home():
	return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
	models=pickle.load(open('ModelDiab.pkl','rb')) #chergement du model
	int_futures=[float(i)for i in request.form.values()] #converssion des donnees
	dernier_futures=[np.array(int_futures)] #donner la forme aux donnees de mm maniere pytho
	dernier_futures=np.array([dernier_futures]).reshape(1,8) # on prend tout 
	predire=models.predict(dernier_futures)
	if(models.predict(dernier_futures)==0):
		predire="positif"
	else:
		predire="negatif" 
	return render_template('index.html',prediction_text_="votre type du diabete est:{}".format(predire))
if __name__=='__main__':
	app.run(debug=True) # debug permet de detecter l'erreur


