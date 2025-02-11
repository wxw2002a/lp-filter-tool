import os
import json
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

# Import your algorithm functions
from lp_filter import lp_filter, read_csv_signal

# Configuration parameters
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key'  # Please set this to a random string
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Define the database model: This will save the information of uploaded files and processing results.
class FileData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(128), unique=True, nullable=False)
    processed_result = db.Column(db.Text, nullable=True)

# Check the file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create database tables before the first request
@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/')
def index():
    # Retrieve all upload records from the database
    files = FileData.query.all()
    return render_template('index.html', files=files)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the file part exists
        if 'file' not in request.files:
            flash('No file part found!')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Ensure the upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)
            
            try:
                # Read the CSV file and extract a specific column (e.g., column index 1)
                data = read_csv_signal(file_path, column=1)
                # Process the data using your lp_filter algorithm (for example, using p=2 and N equal to the length of data)
                result, _ = lp_filter(2, signal='step', N=len(data))
                # Convert the result to a JSON string to save in the database
                result_json = json.dumps(result.tolist())
            except Exception as e:
                flash('Error processing the file: ' + str(e))
                return redirect(request.url)
            
            # Save the upload record to the database
            file_record = FileData(filename=filename, processed_result=result_json)
            db.session.add(file_record)
            db.session.commit()
            
            flash('File uploaded and processed successfully!')
            return redirect(url_for('index'))
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
