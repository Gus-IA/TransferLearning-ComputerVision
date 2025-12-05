import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

# activamos el modo bechmark para que pruebe diferentes algoritmos de convolución y escogerá el más rápido
# cudnn es una biblioteca de nvidia para optimizar redes neuronales
cudnn.benchmark = True

# aplicamos varias transformaciones en el dataset
# recortar, invertir, convertir a tensor y normaliza
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# directorio del dataset
data_dir = "hymenoptera_data"

# separamos el dataset en train y test
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}

# creamos el dataloader con su batch, mezclamos imágenes
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=4, shuffle=True, num_workers=4
    )
    for x in ["train", "val"]
}

# guardamos las imágenes
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

# extraemos la lista de clases
class_names = image_datasets["train"].classes

# uso de la gpu
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


# mostramos varios ejemplos del dataset
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# cogemos un batch para mostrarlo
inputs, classes = next(iter(dataloaders["train"]))

# creamos una rejilla
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
plt.show()


# función de entrenamiento
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # creamos un directorio temporal para ir guardando checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # en cada epoch se entrena y valida
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # iteramos por los datos y los enviamos a la gpu
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # A zero el parametro de los gradientes
                    optimizer.zero_grad()

                    # calculamos los gradientes
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimizer si está en entrenamiento
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # estadísticas
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # guardamos los mejores pesos del modelo
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(
            f"Entrenamiento completado en: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Mejor valor Acc: {best_acc:4f}")

        # cargamos el modelo con los mejores parámetros
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


# función de evaluación y visualización de imágenes
def visualize_model(model, num_images=6):
    # si está en modo train, se pone en modo evaluación
    was_training = model.training
    model.eval()
    # creamos contadores y figuras para la visualización
    images_so_far = 0
    fig = plt.figure()

    # desactivamos los gradientes
    with torch.no_grad():
        # iteramos sobre los dataloader
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward y predicciones
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # visualización con su predicción
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {class_names[preds[j]]}")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# Fine-tuning
# cargamos un modelo resnet18 preentrenado
model_ft = models.resnet18(weights="IMAGENET1K_V1")

# obtenemos el número de entradas del clasificador final
num_ftrs = model_ft.fc.in_features

# reemplazamos la última capa
model_ft.fc = nn.Linear(num_ftrs, 2)

# añadimos el modelo a la gpu
model_ft = model_ft.to(device)

# función de pérdida
criterion = nn.CrossEntropyLoss()

# optimizador
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# multiplicamos el lr por gama cada 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# entrenamos el modelo
model_ft = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25
)

visualize_model(model_ft)


# Feature extraction
# hacemos lo mismo pero congelando todas las capas de preenetrenado
model_conv = torchvision.models.resnet18(weights="IMAGENET1K_V1")
for param in model_conv.parameters():
    param.requires_grad = False


num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# ponemos el modelo a entrenar
model_conv = train_model(
    model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25
)


visualize_model(model_conv)

plt.ioff()
plt.show()
