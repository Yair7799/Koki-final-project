import os
import sys
from flask import Flask, request, jsonify, redirect, url_for, render_template, send_from_directory, session, after_this_request
from werkzeug.utils import secure_filename
from predictions import predict_5_jobs
from gpt_text_analysis import file_to_parsed_res
from adjust_to_job import optimize_resume, create_pdf

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set your secret key here

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input', methods=['GET', 'POST'])
def input_page():
    if request.method == 'POST':
        # Handle form data

        # Handle file uploads
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and file.filename.endswith('.pdf'):
            preprocessed_text = file_to_parsed_res(file)
            predictions = predict_5_jobs(preprocessed_text)
            print(type(preprocessed_text))
            session['predictions'] = {
                'features': preprocessed_text,  # Assuming you want to include form data as features
                'predictions': predictions
            }
            return redirect(url_for('results'))
        else:
            return jsonify({"error": "Invalid file format"}), 400
        
    return render_template('input.html')

@app.route('/results', methods=['GET'])
def results():
    if 'predictions' not in session:
        return redirect(url_for('home'))
    predictions = session['predictions']
    
    return render_template('results.html', predictions=predictions)


@app.route('/adjust', methods=['GET', 'POST'])
def adjust():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and file.filename.endswith('.pdf'):
            job_link = request.form.get('job')
            optimized_pdf_io = create_pdf(optimize_resume(file, job_link))
            
            # Extract byte data from the BytesIO object
            optimized_pdf = optimized_pdf_io.getvalue()
            
            # Save the optimized PDF to the UPLOAD_FOLDER
            optimized_filename = 'optimized.pdf'
            optimized_path = os.path.join(app.config['UPLOAD_FOLDER'], optimized_filename)
            with open(optimized_path, 'wb') as f:
                f.write(optimized_pdf)

            # Store the filename in the session to pass to the results page
            session['optimized_filename'] = optimized_filename
            
            return redirect(url_for('adjust_results'))
        
    return render_template('adjust.html')

@app.route('/adjust_results')
def adjust_results():
    # Get the filename from the session
    optimized_filename = session.get('optimized_filename', None)
    if not optimized_filename:
        return jsonify({"error": "No optimized file found"}), 400
    
    return render_template('adjust_results.html', filename=optimized_filename)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    @after_this_request
    def cleanup(response):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing or closing downloaded file: {e}")
        return response

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
