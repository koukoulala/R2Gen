#!/usr/bin/python3
from flask import Flask, render_template, request, send_from_directory, send_file
import os
import time
import requests
import sys
import codecs

from inference import args, inference
import torch
import numpy as np
from modules.tokenizers import Tokenizer
from models.r2gen import R2GenModel
from torchvision import transforms

from PIL import Image
import numpy as np
import base64
import io

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./upload"
app.config['latest_file'] = ""

if not os.path.isdir(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

headers = {'content-type': 'application/json'}

# fix random seeds
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

# create tokenizer
tokenizer = Tokenizer(args)

# build model architecture
model = R2GenModel(args, tokenizer)
model = model.to(device)
if args.resume is not None:
    resume_path = str(args.resume)
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    today = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    today_dir = os.path.join(args.save_dir, today)
    if not os.path.isdir(today_dir):
        os.mkdir(today_dir)

    app.config['UPLOAD_FOLDER'] = today_dir
    entries = os.listdir(today_dir)
    return render_template('index.html', entries=entries)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        app.config['latest_file'] = file.filename
        print(file.filename)

        image = Image.open(path).convert('RGB')
        report = inference(model, image, transform, device)
        print("report", report)

        # post
        data = {"imgUrl": file.filename,
                "report": report}
        # r = requests.post("http://49.235.248.19:8081/medical_care/upload/success", json=data)
        r = requests.post("http://127.0.0.1:8081/medical_care/upload/success", json=data)
        print(r.json())

        return render_template('upload.html')
    else:
        entries = os.listdir(app.config['UPLOAD_FOLDER'])
        return render_template('index.html', entries=entries)


@app.route('/download/<filename>')
def download(filename):
    print("filename exist", filename, app.config['UPLOAD_FOLDER'], app.config['latest_file'])
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/download_latest')
def download_latest():
    if app.config['latest_file'] == "":
        entries = os.listdir(app.config['UPLOAD_FOLDER'])
        return render_template('index.html', entries=entries)

    print(app.config['UPLOAD_FOLDER'], app.config['latest_file'])
    #return send_file(app.config['latest_file'], as_attachment=True)
    return send_from_directory(app.config['UPLOAD_FOLDER'], app.config['latest_file'], as_attachment=True)


@app.route('/get_latest_filename')
def get_latest_filename():
    print("get_latest_file_info", app.config['latest_file'], app.config['latest_time'])
    return app.config['latest_file']


@app.route('/get_report/<filename>')
def get_report(filename):
    images_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print("images_path", images_path)
    image = Image.open(images_path).convert('RGB')
    report = inference(model, image, transform, device)
    print("report", report)

    return report



if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
