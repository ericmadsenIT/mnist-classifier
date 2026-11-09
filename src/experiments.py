import matplotlib.pyplot as plt
import torch
from torch import nn

from data import get_dataloaders
from model import MNISTClassifier
from train import IMAGES_DIR, WEIGHTS_PATH, epoch_train_eval, get_device


def get_misclassified_images(model, dataloader, device):
    misclass_images = []
    misclass_predictions = []
    misclass_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            model_output = model(images)
            predictions = torch.argmax(model_output, dim=1)

            images = images.cpu().numpy()
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()
            misclass_mask = predictions != labels

            misclass_images.extend(images[misclass_mask])
            misclass_predictions.extend(predictions[misclass_mask].tolist())
            misclass_labels.extend(labels[misclass_mask].tolist())

    return misclass_images, misclass_predictions, misclass_labels


def plot_misclassified_images(misclass_images, misclass_predictions, misclass_labels, save_path):
    num_to_plot = min(10, len(misclass_images))
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(num_to_plot):
        img = misclass_images[i].squeeze()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Pred: {misclass_predictions[i]}\nTrue: {misclass_labels[i]}", fontsize=10)
        axes[i].axis("off")

    for i in range(num_to_plot, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def compare_learning_rates(learning_rate_list, epochs, train_dataloader, test_dataloader, loss_fn, device):
    learning_rate_result = {
        "learning_rate": [],
        "training_loss": [],
        "test_loss": [],
        "final_accuracy": [],
    }

    for learning_rate in learning_rate_list:
        print(f"--- Learning rate {learning_rate} ---")
        list_of_training_losses, list_of_test_losses, list_of_accuracies, _ = epoch_train_eval(
            learning_rate, epochs, train_dataloader, test_dataloader, loss_fn, device
        )

        learning_rate_result["learning_rate"].append(learning_rate)
        learning_rate_result["training_loss"].append(list_of_training_losses)
        learning_rate_result["test_loss"].append(list_of_test_losses)
        learning_rate_result["final_accuracy"].append(list_of_accuracies[-1])

    return learning_rate_result


def plot_learning_rate_results(learning_rate_result, save_path):
    epochs = len(learning_rate_result["training_loss"][0])

    plt.figure(figsize=(10, 6))
    for lr, train_loss, final_acc in zip(
        learning_rate_result["learning_rate"],
        learning_rate_result["training_loss"],
        learning_rate_result["final_accuracy"],
    ):
        plt.plot(range(1, epochs + 1), train_loss, label=f"Train (lr={lr}, final acc={final_acc:.1%})")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.yscale("log")
    plt.xticks(range(1, epochs + 1))
    plt.title("Training loss across Learning Rates")
    plt.savefig(save_path)
    plt.close()


def main():
    torch.manual_seed(42)

    device = get_device()
    print(f"Using {device} device")

    train_dataloader, test_dataloader = get_dataloaders(batch_size=64)
    loss_fn = nn.CrossEntropyLoss()
    IMAGES_DIR.mkdir(exist_ok=True)

    # Experiment 1: misclassified images.
    # Use the weights saved by train.py if they exist, otherwise train a new model.
    if WEIGHTS_PATH.exists():
        model = MNISTClassifier().to(device)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("Loaded model weights from", WEIGHTS_PATH)
    else:
        _, _, _, model = epoch_train_eval(0.1, 10, train_dataloader, test_dataloader, loss_fn, device)

    misclass_images, misclass_predictions, misclass_labels = get_misclassified_images(model, test_dataloader, device)
    print(f"Misclassified Images: {len(misclass_images)}")
    plot_misclassified_images(
        misclass_images, misclass_predictions, misclass_labels, IMAGES_DIR / "misclassified_images.png"
    )

    # Experiment 2: learning rate comparison.
    learning_rate_result = compare_learning_rates(
        [0.001, 0.01, 0.1, 1.0], 5, train_dataloader, test_dataloader, loss_fn, device
    )
    plot_learning_rate_results(learning_rate_result, IMAGES_DIR / "learning_rate_results.png")

    for lr, acc in zip(learning_rate_result["learning_rate"], learning_rate_result["final_accuracy"]):
        print(f"Learning rate {lr}: final accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
