import os
import zipfile
import urllib.request
import shutil
import random
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import streamlit as st
from PIL import Image

# Step 1: Extract the zip file
def extract_dataset(url='https://clck.ru/3FkU9z', extract_path="/content/Dataset"):

    zip_file_path = 'agricultural_crops.zip'
    
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_file_path)
    
    print("Dataset downloaded successfully. Extracting...")

    # Извлечение zip-файла
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print("Dataset extracted successfully.")

# Step 2: Create train-validation split
def create_train_val_split(src_dir, dest_dir, val_ratio=0.2):
    os.makedirs(os.path.join(dest_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'val'), exist_ok=True)

    for class_name in os.listdir(src_dir):
        if class_name.startswith('.'):
            continue
        class_dir = os.path.join(src_dir, class_name)
        if os.path.isdir(class_dir):
            files = [f for f in os.listdir(class_dir) 
                     if os.path.isfile(os.path.join(class_dir, f)) and not f.startswith('.')]
            random.shuffle(files)
            num_val = int(len(files) * val_ratio)

            os.makedirs(os.path.join(dest_dir, 'train', class_name), exist_ok=True)
            os.makedirs(os.path.join(dest_dir, 'val', class_name), exist_ok=True)

            for i, file_name in enumerate(files):
                if i < num_val:
                    shutil.copy(os.path.join(class_dir, file_name), os.path.join(dest_dir, 'val', class_name))
                else:
                    shutil.copy(os.path.join(class_dir, file_name), os.path.join(dest_dir, 'train', class_name))

# Step 3: Load and preprocess the data
def load_data(data_dir='/content/Dataset_split'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

# Step 4: Load a pre-trained model
def initialize_model(num_classes):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

# Step 5: Train the model
def train_model(model, dataloaders, dataset_sizes, num_epochs=15):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_ft.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

# Step 6: Define a function to predict the class of an uploaded image
def predict_image(image_path, model, class_names, device):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    _, preds = torch.max(output, 1)
    predicted_class = class_names[preds.item()]

    return predicted_class

# Main Streamlit App
def main():
    st.title("Классификация сельскохозяйственных культур")
    
    # Extract and prepare dataset
    extract_dataset()
    src_dir = os.path.join('/content/Dataset', 'Agricultural-crops')
    dest_dir = '/content/Dataset_split'
    create_train_val_split(src_dir, dest_dir)
    dataloaders, dataset_sizes, class_names = load_data(data_dir=dest_dir)

    # Initialize and train model
    model = initialize_model(num_classes=len(class_names))
    model = train_model(model, dataloaders, dataset_sizes, num_epochs=15)

    # Save the trained model
    torch.save(model.state_dict(), 'crop_classification_resnet.pth')

    st.write("Модель обучена и сохранена.")

    uploaded_file = st.file_uploader("Загрузите изображение...", type=["png", "jpg", "jpeg", ""])
    if uploaded_file is not None:
        image_path = os.path.join('/tmp', uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load('crop_classification_resnet.pth', map_location=device))
        model.eval()

        predicted_class = predict_image(image_path, model, class_names, device)
        st.image(image_path, caption='Загруженное изображение.', use_column_width=True)
        st.write(f'Предсказанный класс: {predicted_class}')

if __name__ == "__main__":
    main()
