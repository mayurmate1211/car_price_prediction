from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)
model=pickle.load(open('car.pkl','rb'))
car=pd.read_csv('clean_car_data.csv')

@app.route('/')
def my():
    car_model=sorted(car['Name'].unique())
    year=sorted(car['Year'].unique(),reverse=True)
    fuel_type=car['Fuel_Type'].unique()
    
    return render_template('index.html',car_model=car_model,years=year,fuel_type=fuel_type)


@app.route('/predict',methods=['POST'])
def predict():

    car_model=request.form.get('car_model')
    year=request.form.get('Year')
    fuel_type=request.form.get('Fuel_Type')
    driven=int(request.form.get('Kilometers_Driven'))
    mileage=float(request.form.get('Mileage'))
    engine=int(request.form.get('Engine'))
    power=float(request.form.get('Power'))


    prediction=model.predict(pd.DataFrame([[car_model,year,driven,fuel_type,mileage,engine,power]],columns=['Name', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Mileage', 'Engine','Power']))
                              


    

    return  str(np.round(prediction[0],2))

    


    



if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=8080)
