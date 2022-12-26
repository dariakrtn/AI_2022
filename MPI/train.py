import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
import os
import pandas as pd
from PIL import Image
from mpi4py import MPI
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train(model, criterion, optimizer, dataloader_train, dataset_sizes_train, dataloader_test, dataset_sizes_test, my_rank, num_epochs=10):

    best_score = 0.0 # лучшая точность модели
    for epoch in range(num_epochs): # итерируемся по заданному числу эпох
        model.train() # перевод модели в состояни тренировки
        runing_loss = 0.0 # текущая ошибка
        score = 0 # текущая точность
        
        if my_rank == 0: # если ранг процесса 0, то отправляем данные трем оставшимся процессам
            for image, label in tqdm(dataloader_train):
                for procid in range(1, processes):
                    N = int(image.size(0) / 3) # чиселка при помощи которой мы делим данные на три процесса
                    comm.send(image[(procid-1) * N : procid * N], dest=procid, tag=0) # срезами отправляем частичку изображений каждому процессу
                    comm.send(label[(procid-1) * N : procid * N], dest=procid, tag=1) # срезами отправляем частичку меток каждому процессу
        
        if my_rank != 0: # если ранг процесса не нулевой
            for _ in range(len(dataloader_train)): # поскольку мы отправили какое то количество раз изображения и метки, то должны столько же раз принять эти данные
                image = comm.recv(source=0, tag=0) # приняли изображения
                label = comm.recv(source=0, tag=1) # приняли метки
                optimizer.zero_grad() # занулил градиенты, в pytorch сам не обнуляет градиенты, поэтому большое накопление приводит к плохим результатам
                out = model(image) # прогнали изображения через модель
                _, preds = torch.max(out, 1) # выбрали максимальную вероятность для кажого изображения
                loss = criterion(out, label) # посчитали ошибку на данном батче
                loss.backward() # посчитали градиенты для каждого параметра сети
                optimizer.step() # сделали шаг в нашем многомерном пространстве + изменили веса(парамтры) сети
                runing_loss += loss.item() * image.size(0) # посчитали текущую ошибку
                score += torch.sum(preds == label.data) # посчитали текущую точность
            epoch_acc = score.double() / (dataset_sizes_train / 3) # посчитали точность на данной эпохе  
            runing_loss = runing_loss / (dataset_sizes_train / 3) # посчитали ошибку на данной эпохе
            print("Epoch of train:", epoch + 1, "score: [", epoch_acc.item(), "], loss: [", runing_loss, "]", my_rank) # вывели ранее расчитанные значения
        
        MPI.Comm.Barrier(MPI.COMM_WORLD) 
        
        score = 0
        runing_loss = 0.0
        model.eval() # перевели модель в режим тестирования
        
        with torch.no_grad(): # замораживает градиенты (они не меняются)
            if my_rank == 0:
                for image, label in tqdm(dataloader_test):
                    for procid in range(1, processes):
                        N = int(image.size(0) / 3)
                        comm.send(image[(procid-1) * N : procid * N - 1], dest=procid, tag=0)
                        comm.send(label[(procid-1) * N : procid * N - 1], dest=procid, tag=1)
            
            if my_rank != 0:
                for _ in range(len(dataloader_test)):
                    image = comm.recv(source=0, tag=0)
                    label = comm.recv(source=0, tag=1)
                    out = model(image)
                    _, preds = torch.max(out, 1)
                    loss = criterion(out, label)
                    runing_loss += loss.item() * image.size(0)
                    score += torch.sum(preds == label.data)
                epoch_acc = score.double() / (dataset_sizes_test / 3)
                runing_loss = runing_loss / (dataset_sizes_test / 3)
                print("Epoch of val:", epoch + 1, "score: [", epoch_acc.item(), "], loss: [", runing_loss, "]", my_rank)
            MPI.Comm.Barrier(MPI.COMM_WORLD)
        
        if my_rank != 0: # если ранг не 0, то сохраняем модели с лучшей точностью и минимальной ошибкой
            if epoch == 0:
                best_loss = runing_loss
            if epoch_acc > best_score and runing_loss <= best_loss:
                best_score = epoch_acc
                best_loss = runing_loss
                torch.save(model.state_dict(), f"./weights/model_{my_rank}.pth")
    return model


class MyDataset(Dataset):
    
    def __init__(self, type_data, img_dir, transforms=None):
        """
        type_data - тип формируемых данных (тренировочные или тестовые)
        img_dir - общий путь к данным
        transforms - трансформации (аугментации) к изображениям
        data - создаем pandas таблицу для дальнейшего доступа к изобрадениям и их меткам (лэйблам) 
        """
        self.type_data = type_data 
        self.img_dir = img_dir 
        self.transforms = transforms 
        self.data = self.Generate_Dataframe()  

    def Generate_Dataframe(self):
        """
        work_dir - создали путь до даирректории c данными
        data_frame - создали пустой лист, в него будем пихать данные
        all_paths - названия всех картинок в папке c медведями/пандами и добавили каждую картинку c меткой в наш будующий датафрейм
        Возвращем то что насобирали в фомате pandas
        """
        work_dir = self.img_dir + '/' + self.type_data + '/' 
        data_frame = list() 
        
        all_paths = os.listdir(work_dir + 'Bears') 
        for it in all_paths: 
            data_frame.append([work_dir + 'Bears' + '/' + it, 0])

        all_paths = os.listdir(work_dir + 'Pandas') 
        for it in all_paths: 
            data_frame.append([work_dir + 'Pandas' + '/' + it, 1])

        return pd.DataFrame(data_frame, columns=[0, 1]) 

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        path = self.data.iloc[idx, 0] # из первой калонки в нешй таблице получили путь к изображению
        image = Image.open(path).convert('RGB') # загрузили изображение
        label = self.data.iloc[idx, 1] # из второй колонки поличили метку этого изображения
        if self.transforms: # применили преобразования 
            image = self.transforms(image)
        return image, label

def Create_Dataloader(type_data, img_dir, transform=None, shuffle=None):
    dataset = MyDataset(type_data=type_data, img_dir=img_dir, transforms=transform) # сформировали кастомный датасет при помощи класса MyDataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=shuffle)  # создали даталоадер, batch_size -- сколько картинок будет подано модели
    dataset_sizes = len(dataset) # получили размер датасета
    return dataloader, dataset_sizes

def Create_Model():
    model_ft = models.resnet18(pretrained=True) # загрузили модель
    num_ftrs = model_ft.fc.in_features # получили число выходных нейронов в последнем слое
    model_ft.fc = nn.Linear(num_ftrs, 2) # заменили последний слой (число входных нейронов осталось темже, а выход стал 2мя нейронами поскольку у нас всего два класса)
    criterion = nn.CrossEntropyLoss() # создали функцию ошибки
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01) # создали оптимизатор
    return model_ft, criterion, optimizer_ft

if __name__ == "__main__":

    """
    transform - Resize((32, 32)) изменение размера картинок на 32x32
                ToTensor() перевод в формат торчового тензора
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) нормализация изображения

    img_dir - общий путь до тренировочных и тестовых данных
    dataloader_train, dataset_sizes_train создали dataloader для тренировочный данных и размер тренировочных данных
    dataloader_test, dataset_sizes_test создали dataloader для тестовых данных и размер тестовых данных

    comm - коммуникатор объединяющий работу (взимодействие) всех процессов
    my_rank - ранг текущего процесса
    p  кол-во процессов   

    model, criterion, optimizer  - создали модель, функцию ошибки, оптимизатор
    """

    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
    img_dir = '/content/PandasBears' 
    dataloader_train, dataset_sizes_train = Create_Dataloader(type_data='Train', img_dir=img_dir, transform=transform, shuffle=True) 
    dataloader_test, dataset_sizes_test = Create_Dataloader(type_data='Test', img_dir=img_dir, transform=transform, shuffle=False) 
    
    comm = MPI.COMM_WORLD 
    my_rank = comm.Get_rank() 
    processes = comm.Get_size()

    model, criterion, optimizer = Create_Model()  
    model = train(model, criterion, optimizer, dataloader_train, dataset_sizes_train, dataloader_test, dataset_sizes_test, my_rank, num_epochs=2) 

    MPI.Finalize 