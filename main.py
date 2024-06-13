from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np
app=Flask(__name__)

predict_model = pickle.load(open("LinearRegressionModel.pkl","rb"))
car=pd.read_csv("CleanedCar.csv")

@app.route('/')
def index():
   companies = sorted(car['company'].unique())
   car_model = sorted(car['name'].unique())
   year  =sorted(car['year'].unique(),reverse=True)
   fuel_type = car['fuel_type'].unique()
   return render_template('index.html',companies=companies,car_model=car_model,year=year,fuel_type=fuel_type)

@app.route('/final_data',methods=['POST'])
def final_data():
   company = request.form.get('company')
   car_model = request.form.get('car')
   year = int(request.form.get('year'))
   fuel = request.form.get('fuel')
   kms = int(request.form.get('kms'))

   predict_output = predict_model.predict(pd.DataFrame([[car_model,company,year,kms,fuel]],columns=['name','company','year','kms_driven','fuel_type']))

   return str(np.round(predict_output[0],2))
if __name__ =="__main__":
     app.run(debug=False)