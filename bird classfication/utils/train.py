import torch
from torch.utils.tensorboard import SummaryWriter
import csv 
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, log_dir='logs', model_dir='saved_models', experiment_name='experiment'):
	writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
	results = []
	for epoch in range(num_epochs):
		model.train()
		running_loss = 0.0
		for inputs, labels in train_loader:
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * inputs.size(0)

		epoch_loss = running_loss / len(train_loader.dataset)
		writer.add_scalar('training loss', epoch_loss, epoch)

		model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for inputs, labels in val_loader:
				outputs = model(inputs)
				_, predicted = torch.max(outputs, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		accuracy = correct / total
		writer.add_scalar('validation accuracy', accuracy, epoch)
		results.append((epoch, epoch_loss, accuracy))
	writer.close()
    
    # Save results to CSV
	os.makedirs(log_dir, exist_ok=True)
	results_file = os.path.join(log_dir, f'{experiment_name}_results.csv')
	with open(results_file, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['epoch', 'loss', 'accuracy'])
		writer.writerows(results)
	# Save model weights
	os.makedirs(model_dir, exist_ok=True)
	model_file = os.path.join(model_dir, f'{experiment_name}_model_weights.pth')
	torch.save(model.state_dict(), model_file)
	print(f"Model weights saved to {model_file}")
