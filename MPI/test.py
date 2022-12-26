import torch
from train import Create_Dataloader, Create_Model
from mpi4py import MPI
from tqdm import tqdm
from torchvision import transforms, models
import torch.nn as nn

def test(model, criterion, dataloader_test, dataset_sizes_test):
    score = 0
    runing_loss = 0.0
    model.eval()
    result = 0

    with torch.no_grad():
        if my_rank != 0:
            for image, label in tqdm(dataloader_test): # прогоняем все тестовые данные через обученные модели
                out = model(image)
                comm.send(out, dest=0, tag=0) # каждый процесс посылает выход своей модели нулевому процессу
                if my_rank == 1:
                    comm.send(label, dest=0, tag=1) # т.к. нулевой процесс не получил лэйблы, то их ему нужно отправить (например первый мпроцессом)
                _, preds = torch.max(out, 1)
                loss = criterion(out, label)
                runing_loss += loss.item() * image.size(0)
                score += torch.sum(preds == label.data)
            epoch_acc = score.double() / dataset_sizes_test
            runing_loss = runing_loss / dataset_sizes_test
            print("Test process ", my_rank, ": score: [", epoch_acc.item(), "], loss: [", runing_loss, "]", my_rank)
        
        MPI.Comm.Barrier(MPI.COMM_WORLD)

        if my_rank == 0:
            result = 0
            for _ in range(len(dataloader_test)): # принимаем столько же раз сколько и послали
                result_tmp = 0
                label = comm.recv(source=1, tag=1) # нулевой процесс принял лэйблы от первого процесса
                for procid in range(1, process): # здесь принимаются выходы сетей для кажого из процессов
                    out = comm.recv(source=procid, tag=0)
                    if procid == 1:
                        result_all_models = out
                    else:
                        result_all_models += out
                result_all_models /= 3 # и устредняются
                _, preds = torch.max(result_all_models, 1)
                result += torch.sum(preds == label.data)
            result = result.double() / dataset_sizes_test
            print("Test process result", my_rank, ": score: [", result.item(), "]") # выводится итоговый результат


def LoadModel():
    model_ft = models.resnet18(pretrained=False, num_classes=2) # создали непредобученную модель
    model_ft.load_state_dict(torch.load(f'/content/weights/model_{my_rank}.pth')) # загрузили ранее сохраненные веса
    criterion = nn.CrossEntropyLoss()
    return model_ft, criterion


if __name__ == "__main__":
    
    """
    transform - Resize((32, 32)) изменение размера картинок на 32x32
                ToTensor() перевод в формат торчового тензора
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) нормализация изображения

    img_dir - общий путь до тренировочных и тестовых данных
    
    comm - коммуникатор объединяющий работу (взимодействие) всех процессов
    my_rank - ранг текущего процесса
    process  кол-во процессов   

    dataloader_test, dataset_sizes_test создали dataloader для тестовых данных и размер тестовых данных
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img_dir = '/content/PandasBears'

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    process = comm.Get_size()

    dataloader_test, dataset_sizes_test = Create_Dataloader(type_data='Test', img_dir=img_dir, transform=transform, shuffle=False)
    if my_rank != 0:
        model, criterion = LoadModel() # загружаем ранее обученные модели
    else:
        model, criterion, optimizer = Create_Model() # чтобы нулевой процесс мог войти в функцию тестирования ему тоже нужно создать модель и ошибку
    test(model, criterion, dataloader_test, dataset_sizes_test)

    MPI.Finalize