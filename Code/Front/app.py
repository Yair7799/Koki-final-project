import os
from flask import Flask, render_template, request, jsonify

from back import extract_text_from_pdf, predict_character_traits, preprocess_text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input', methods=['GET', 'POST'])
def input_page():
    if request.method == 'POST':
        # Handle form data
        data = request.form.to_dict()
        print(f"Received data: {data}")  # Debugging line

        # Handle file uploads
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and file.filename.endswith('.pdf'):
            file1 = preprocess_text(file)
            text = extract_text_from_pdf(file1)
            predictions = predict_character_traits(text)
            return jsonify({"predictions": predictions})
        else:
            return jsonify({"error": "Invalid file format"}), 400

    return render_template('input.html')

@app.route('/results', methods=['POST', 'GET'])
def results():
    data = request.get_json()
    features = data.get('features', [])
    predictions = data.get('predictions', [])
    
    return render_template('results.html', predictions={'features': features, 'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
