from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('random_forest_model.pkl')

# Load accuracy
with open('accuracy.txt', 'r') as f:
    accuracy = float(f.read())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    umur = int(request.form['umur'])
    # umur = 1 if umur <= 41 else 0
    jenis_kelamin = int(request.form['jenis_kelamin'])
    foto_toraks = int(request.form['foto_toraks'])
    status_hiv = int(request.form['status_hiv'])
    riwayat_diabetes = int(request.form['riwayat_diabetes'])

    features = [[umur, jenis_kelamin, foto_toraks, status_hiv, riwayat_diabetes]]
    prediction = model.predict(features)[0]
    prediction_text = 'Paru' if prediction == 1 else 'Ekstra paru'

    return render_template('result.html', prediction=prediction_text, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
