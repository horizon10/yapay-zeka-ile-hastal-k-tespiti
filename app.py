from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Flask uygulamasını başlatma
app = Flask(__name__)

# Modeli ve scaler'ı yükleme
data = pd.read_csv('C:/..../cardio_train.csv', delimiter=';', header=0)
data['age'] = data['age'] // 365
X = data.drop(columns=['cardio', 'id'])
y = data['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)
    output = prediction[0]
    if output == 1:
        result = "Yüksek risk altında olduğunuzu gösteriyor. Bir doktora danışmanız tavsiye edilir."
    else:
        result = "Düşük risk altında olduğunuzu gösteriyor. Sağlıklı yaşam tarzınızı sürdürmeye devam edin!"
    return render_template('index.html', prediction_text=f'Tahmin edilen sonuç: {result}')

if __name__ == "__main__":
    app.run(debug=True)
