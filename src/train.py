from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from data import get_dataloaders
from model import MNISTClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT_DIR / "images"
WEIGHTS_PATH = ROOT_DIR / "mnist_classifier.pth"


def get_device():
    return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0.0
    model.train()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        loss = loss_fn(model(images), labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss = total_loss / len(dataloader)
    return total_loss


def evaluate_model(model, dataloader, loss_fn, device):
    total_loss = 0.0
    correct_predictions = 0
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            model_output = model(images)

            predictions = torch.argmax(model_output, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()

            loss = loss_fn(model_output, labels)
            total_loss += loss.item()

    total_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    return total_loss, accuracy


def epoch_train_eval(learning_rate, epochs, train_dataloader, test_dataloader, loss_fn, device):
    model = MNISTClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    list_of_training_losses = []
    list_of_test_losses = []
    list_of_accuracies = []

    for epoch in range(epochs):
        average_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, accuracy = evaluate_model(model, test_dataloader, loss_fn, device)
        list_of_training_losses.append(average_loss)
        list_of_test_losses.append(test_loss)
        list_of_accuracies.append(accuracy)

        print("Epoch number:", (epoch + 1))
        print(f"Average loss: {average_loss:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

    return list_of_training_losses, list_of_test_losses, list_of_accuracies, model


def plot_losses(list_of_training_losses, list_of_test_losses, save_path):
    epochs = len(list_of_training_losses)

    plt.figure()
    plt.plot(range(1, epochs + 1), list_of_training_losses, label="Training loss", color="red")
    plt.plot(range(1, epochs + 1), list_of_test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(range(1, epochs + 1))
    plt.title("Training and Test Loss per Epoch")
    plt.savefig(save_path)
    plt.close()


def main():
    torch.manual_seed(42)

    learning_rate = 0.1
    epochs = 10
    batch_size = 64

    device = get_device()
    print(f"Using {device} device")

    train_dataloader, test_dataloader = get_dataloaders(batch_size)
    loss_fn = nn.CrossEntropyLoss()

    list_of_training_losses, list_of_test_losses, list_of_accuracies, model = epoch_train_eval(
        learning_rate, epochs, train_dataloader, test_dataloader, loss_fn, device
    )

    print(f"Final test accuracy: {list_of_accuracies[-1] * 100:.2f}%")

    IMAGES_DIR.mkdir(exist_ok=True)
    plot_losses(list_of_training_losses, list_of_test_losses, IMAGES_DIR / "train_eval_loss.png")
    print("Saved loss chart to", IMAGES_DIR / "train_eval_loss.png")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print("Saved model weights to", WEIGHTS_PATH)


if __name__ == "__main__":
    main()
