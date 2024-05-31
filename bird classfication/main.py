import argparse
import torch
from models.resnet import get_resnet18
from utils.data_loader import get_data_loaders
from utils.train import train_model
from utils.evaluate import evaluate_model

def main(args):
	# Get data loaders
	train_loader, val_loader = get_data_loaders(args.train_dir, args.val_dir, args.batch_size)

	# Initialize model
	model = get_resnet18(num_classes=200)

	# Define loss function and optimizer
	criterion = torch.nn.CrossEntropyLoss()
	fc_params = list(model.fc.parameters())
	other_params = list(model.parameters())[:-len(fc_params)]
	
	optimizer = torch.optim.SGD([
		{'params': fc_params, 'lr': args.learning_rate},
		{'params': other_params, 'lr': args.fine_tune_learning_rate}
	], momentum=args.momentum)
	
	# Create a unique experiment name
	experiment_name = f'lr_{args.learning_rate}_ftlr_{args.fine_tune_learning_rate}_bs_{args.batch_size}_epochs_{args.epochs}'

	# Train model
	train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs, experiment_name=experiment_name)

	# Evaluate model
	accuracy = evaluate_model(model, val_loader)
	print(f'Validation Accuracy: {accuracy}')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Hyperparameter tuning for bird classification")
	parser.add_argument('--train_dir', type=str, default='data/CUB_200_2011/train', help='Path to the training data')
	parser.add_argument('--val_dir', type=str, default='data/CUB_200_2011/test', help='Path to the validation data')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
	parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
	parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the final layer')
	parser.add_argument('--fine_tune_learning_rate', type=float, default=0.001, help='Learning rate for fine-tuning')
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
	args = parser.parse_args()
	main(args)

