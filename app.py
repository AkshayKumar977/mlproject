from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import pickle
import os
app = Flask(__name__)
model = None
scaler = None
def load_model():
    global model,scaler
    try:
        with open("model.pkl","rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        with open("scaler.pkl","rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully")
        return True
    except FileNotFoundError as e:
        print("Error: Required files not found")
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        return False
if not load_model():
    print("\n WARNING: Model is not loaded. Prediciton will not work until the model files are available.\n")
@app.route('/')
def home():
    return render_template("index.html")
@app.route("/predict",methods = ['POST'])
def predict():
    try:

    
        if model is None or scaler is None:
             return jsonify({
                "error":"Model not loaded." 
            }),500
        data = request.form
        features = {
        "LIMIT_BAL": float(data['LIMIT_BAL']),
        'SEX': int(data['SEX']),
        'EDUCATION': int(data['EDUCATION']),
        'MARRIAGE': int(data['MARRIAGE']),
        'AGE': int(data['AGE']),
        'PAY_0': int(data['PAY_0']),
        'PAY_2': int(data['PAY_2']),
        'PAY_3': int(data['PAY_3']),
        'PAY_4': int(data['PAY_4']),
        'PAY_5': int(data['PAY_5']),
        'PAY_6': int(data['PAY_6']),
        'BILL_AMT1': float(data['BILL_AMT1']),
        'BILL_AMT2': float(data['BILL_AMT2']),
        'BILL_AMT3': float(data['BILL_AMT3']),
        'BILL_AMT4': float(data['BILL_AMT4']),
        'BILL_AMT5': float(data['BILL_AMT5']),
        'BILL_AMT6': float(data['BILL_AMT6']),
        'PAY_AMT1': float(data['PAY_AMT1']),
        'PAY_AMT2':float(data['PAY_AMT2']),
        'PAY_AMT3': float(data['PAY_AMT3']),
        'PAY_AMT4': float(data['PAY_AMT4']),
        'PAY_AMT5': float(data['PAY_AMT5']),
        'PAY_AMT6': float(data['PAY_AMT6'])
           }
    except (ValueError, TypeError) as e:
        return jsonify({'error': f"Error converting input values: {str(e)}"}),400
        input_df = pd.DataFrame([features])
        numerical_col = ['LIMIT_BAL','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
        input_df[numerical_col] = scaler.transform(input_df[numerical_col])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        default_prob = float(probability[1] * 100)
        no_default_prob = float(probability[0] * 100)
        if default_prob<30:
            risk_category = "Low Risk"
            risk_color = "success"
            recommendation = "Approved - Low default probability. Standard temrs recommends."
        elif default_prob < 60:
            risk_category = "Medium Risk"
            risk_color = "warning"
            recommendation = "Review Required - Moderate default risk. Consider higher interest rate or additional colletral."
        else:
            risk_category = "High Risk"
            risk_color = "danger"
            recommendation = "Caution - High default probability. Recommend rejection or require significant colletral."
        result = {
            "prediction": int(prediction),
            "prediction_text": 'Default' if prediction == 1 else 'Not default',
            'probability_default': float(probability[1]*100),
            'probability_no_default': float(probability[0]*100),
            'risk_category': risk_category,
            'risk_color': risk_color,
            'recommendation': recommendation
        }
        return jsonify(result)
    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}),400
    except ValueError as e:
        return jsonify({'error': f"Invalid input values: {str(e)}"}),400,
    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}),500
@app.route('/health',methods = ['GET'])
def health():
    status = {
        'status':'healthy' if model is not None and scaler is not None else 'Unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }
    return jsonify(status)
if __name__ == '__main__':
    app.run(debug = True)



        
