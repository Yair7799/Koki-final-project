import os
import sys
from back import find_employment_avenues_recommendations
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input', methods=['GET', 'POST'])
def input_page():
    if request.method == 'POST':
        data = request.form.to_dict()
        recommendations = find_employment_avenues_recommendations(data)
        if 'resume' in request.files:
            resume = request.files['resume']
            if resume.filename != '':
                filename = secure_filename(resume.filename)
                resume.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                data['resume_filename'] = filename
        
        return redirect(url_for('results', **recommendations))
    
    return render_template('input.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    data = request.args.to_dict()
    return render_template('results.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
