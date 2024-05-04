from keras.models import load_model
from flask import Flask, render_template, request
import numpy as np
from main import preprocess_input, predict_churn, x
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = load_model('mymodel.h5')  # Load your model
sc = StandardScaler()  # Initialize StandardScaler
sc.fit(x)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("result.html")
    
    else:
        
        France = float(request.form.get('France'))
        Spain = float(request.form.get('Spain'))
        Germany = float(request.form.get('Germany'))
        CreditScore = float(request.form.get('CreditScore'))
        Gender = float(request.form.get('Gender'))
        Age = float(request.form.get('Age'))
        Tenure = float(request.form.get('Tenure'))
        Balance = float(request.form.get('Balance'))
        NumOfProducts = float(request.form.get('NumOfProducts'))
        HasCrCard = float(request.form.get('HasCrCard'))
        IsActiveMember = float(request.form.get('IsActiveMember'))
        EstimatedSalary = float(request.form.get('EstimatedSalary'))
    
    #transforming data      
    input_data = np.array([[France, Spain, Germany, CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])
    input_data_scaled = sc.transform(input_data)

    # Perform prediction
    churn_prediction = model.predict(input_data_scaled)

    # Format the prediction as Churn (1) or Not Churn (0)
    churn_result = "Churn" if churn_prediction > 0.5 else "Not Churn"

    return render_template('result.html', churn_result=churn_result)


if __name__ == '__main__':
    app.run(debug=True)
