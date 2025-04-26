from flask import Flask, render_template, request
import os
from utils import predict_disease

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'leaf' not in request.files:
        return "No file uploaded"
    
    file = request.files['leaf']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result, confidence, severity, treatment = predict_disease(filepath)

    return render_template('index.html', prediction=result, confidence=confidence,
                           severity=severity, treatment=treatment, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)