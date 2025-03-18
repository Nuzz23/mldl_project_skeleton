from models.customModel import CustomNet
import torch 
from train import train
from eval import validate
from data.dataManagment import get_data_loaders
import wandb


def main(config):
    model = CustomNet().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    best_acc = 0

    train_loader, val_loader = get_data_loaders()

    # Run the training process for {num_epochs} epochs
    for epoch in range(1, config.epoches + 1):
        train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion)
        
        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)
        
        wandb.log({'val_accuracy': val_accuracy, 'best_val_accuracy': best_acc, 'epoch': epoch})


    print(f'Best validation accuracy: {best_acc:.2f}%')
