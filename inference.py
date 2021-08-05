import os

import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from models.r2gen import R2GenModel
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()

# Data input settings
parser.add_argument('--image_dir', type=str, default='data/test_images/', help='the path to the directory containing the data.')
parser.add_argument('--ann_path', type=str, default='data/mimic_cxr/annotation.json',
                    help='the path to the directory containing the data.')

# Data loader settings
parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'],
                    help='the dataset to be used.')
parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

# Model settings (for visual extractor)
parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

# Model settings (for Transformer)
parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
# for Relational Memory
parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

# Sample related
parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
parser.add_argument('--group_size', type=int, default=1, help='the group size.')
parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

# Trainer settings
parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
parser.add_argument('--epochs', type=int, default=30, help='the number of training epochs.')
parser.add_argument('--save_dir', type=str, default='results/mimic_cxr', help='the patch to save the models.')
parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

# Optimization
parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
parser.add_argument('--amsgrad', type=bool, default=True, help='.')

# Learning Rate Scheduler
parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
parser.add_argument('--step_size', type=int, default=1, help='the step size of the learning rate scheduler.')
parser.add_argument('--gamma', type=float, default=0.8, help='the gamma of the learning rate scheduler.')

# Others
parser.add_argument('--seed', type=int, default=456789, help='.')
parser.add_argument('--resume', type=str, default="./ckpt/model_mimic_cxr.pth", help='whether to resume the training from existing checkpoints.')

args = parser.parse_args()

def inference(model, image, transform, device):

    model.eval()
    with torch.no_grad():
        image = transform(image).to(device)
        image = torch.unsqueeze(image, 0)
        output = model(image, mode='sample')
        reports = model.tokenizer.decode_batch(output.cpu().numpy())[0]

    return reports

def main():

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

    for images_id in os.listdir(args.image_dir):
        if images_id.endswith(".jpg"):
            images_path = os.path.join(args.image_dir, images_id)
            print("images_path", images_path)

            report = inference(model, images_path, transform, device)
            print("report", report)

if __name__ == '__main__':
    main()
