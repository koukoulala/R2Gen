#!/usr/bin/python3
from flask import Flask, render_template, request, send_from_directory, send_file
import os
import time
from inference import args, inference
import torch
import numpy as np
from modules.tokenizers import Tokenizer
from models.r2gen import R2GenModel
from torchvision import transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./data/test_images"

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

@app.route('/get_report/<filename>')
def download(filename):
    today = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    today_dir = os.path.join(args.save_dir, today)
    if not os.path.isdir(today_dir):
        return "File doesn't exist!"

    app.config['UPLOAD_FOLDER'] = today_dir
    images_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print("images_path", images_path)
    report = inference(model, images_path, transform, device)

    return report



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
