from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(request.url)

    file = request.files['file']
    privileged_gender = request.form['privileged_gender']
    unprivileged_gender = request.form['unprivileged_gender']
    metric = request.form['metric']

    if file and file.filename == 'loan_data.csv':
        df = pd.read_csv(file)

        df['gender'] = df['gender'].map({privileged_gender: 1, unprivileged_gender: 0})

        
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

        
        df = pd.get_dummies(df, columns=['Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)

        df.drop(columns=['Loan_ID'], inplace=True)

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

        X = df.drop(columns=['Loan_Status'])
        y = df['Loan_Status']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(X, y, X_train, X_test, y_train, y_test, y_pred, y_prob, model, metric)
        metrics_json = json.dumps(metrics)

        return redirect(url_for('show_result', metric=metric, result=metrics_json))

def calculate_metrics(X, y, X_train, X_test, y_train, y_test, y_pred, y_prob, model, metric):
    if metric == 'classification':
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
        }
    
    if metric == 'binary_label_dataset':
        conf_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        return {
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn,
            'True Positives': tp
        }
    
    if metric == 'dataset':
        class_balance = np.bincount(y)
        class_balance_train = np.bincount(y_train)
        class_balance_test = np.bincount(y_test)
        return {
            'Overall Class Balance': class_balance.tolist(),
            'Training Set Class Balance': class_balance_train.tolist(),
            'Test Set Class Balance': class_balance_test.tolist()
        }
    
    if metric == 'sample_distortion':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        distortion_train = np.mean(np.abs(X_train - X_train_scaled))
        distortion_test = np.mean(np.abs(X_test - X_test_scaled))
        return {
            'Training Set Distortion': distortion_train,
            'Test Set Distortion': distortion_test
        }

    if metric == 'mdss':
        subgroup_threshold = np.median(X[:, 0])
        subgroup_mask = X_test[:, 0] > subgroup_threshold
        subgroup_y_test = y_test[subgroup_mask]
        subgroup_y_pred = y_pred[subgroup_mask]
        return {
            'Subgroup Accuracy': accuracy_score(subgroup_y_test, subgroup_y_pred),
            'Subgroup Precision': precision_score(subgroup_y_test, subgroup_y_pred),
            'Subgroup Recall': recall_score(subgroup_y_test, subgroup_y_pred),
            'Subgroup F1 Score': f1_score(subgroup_y_test, subgroup_y_pred)
        }

    return 'Invalid metric'

@app.route('/result')
def show_result():
    metric = request.args.get('metric')
    result = json.loads(request.args.get('result'))
    return render_template('result.html', metric=metric, result=result)

if __name__ == '__main__':
    app.run(debug=True)