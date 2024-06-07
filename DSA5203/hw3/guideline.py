import time
import glob
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import SceneDataset
from torch.utils.data import DataLoader
from model import ResNet50

"""
Guideline of your submission of HW3.
If you have any questions in regard to submission,
please contact TA: Ma Zhiyuan <e0983565@u.nus.edu>
"""

###################################### Subroutines #####################################################################
"""
Example of subroutines you might need.
You could add/modify your subroutines in this section. You can also delete the unnecessary functions.
It is encouraging but not necessary to name your subroutines as these examples.
"""

def get_accuracy(softmax_outputs, labels):
    _, predicted_classes = torch.max(softmax_outputs, dim=1) #dim=1 check for max in a row, dim=0 check for max in a col
    correct_predictions = (predicted_classes == labels).sum().item()
    accuracy = correct_predictions / labels.size(0) * 100
    return accuracy

###################################### Main train and test Function ####################################################
"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test,
computing accuracy, etc.
"""

def train(train_data_dir, model_dir, val_size=0.1, random_state=1, input_dim=224, batch_size=256, lr=1e-3, epochs=1000):
    """
    Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
    # GET IMAGE PATHS
    image_paths = glob.glob(train_data_dir+'/**/*.jpg')
    labels = [path.split('/')[-2] for path in image_paths]

    # TRAIN VAL SPLIT IMAGE PATHS
    train_image_paths,val_image_paths = train_test_split(image_paths,test_size=val_size,random_state=random_state,shuffle=True,stratify=labels)

    # DEFINE DATASET
    train_dataset = SceneDataset(train_image_paths,split_type='train',input_dim=input_dim)
    val_dataset = SceneDataset(val_image_paths,split_type='val',input_dim=input_dim)

    # DEFINE DATALOADER
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    print('\ntrain val dataset and dataloader processed')


    # DEFINE TRAINING VARIABLES
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # DEFINE MODEL
    model=ResNet50(15, 1)

    # ensure the weigshts and biases have same format as the input dtype
    model = model.to(torch.float32)
    model = model.to(device)
    print('model defined')

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    early_stop_count = 0
    min_val_loss = float('inf')

    # TRAIN MODEL - IDENTIFY EARLY STOPPING EPOCH
    train_losses,val_losses=[],[]
    train_accs,val_accs=[],[]

    trg_start_time = time.time()
    print('Start training - early stopping with retraining')

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # TRAINING LOOP
        model.train()

        train_running_loss = []
        correct_predictions = 0
        total_predictions = 0
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            softmax_outputs=nn.Softmax(dim=1)(logits)
            loss = criterion(logits,labels)
            train_running_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            _, predicted_classes = torch.max(softmax_outputs, dim=1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)

        train_loss = np.mean(train_running_loss)
        train_acc = (correct_predictions / total_predictions) * 100
        train_losses.append(train_loss)
        train_accs.append(train_acc)


        # VALIDATION LOOP
        model.eval()

        val_running_loss = []
        correct_predictions = 0
        total_predictions = 0
        for batch in val_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            softmax_outputs=nn.Softmax(dim=1)(logits)
            loss = criterion(logits,labels)
            val_running_loss.append(loss.item())


            _, predicted_classes = torch.max(softmax_outputs, dim=1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)

        val_loss = np.mean(val_running_loss)
        val_acc = (correct_predictions / total_predictions) * 100
        val_losses.append(val_loss)
        val_accs.append(val_acc)


        # SAVE TRAINING STATISTICS
        train_loss,val_loss=round(train_loss,4),round(val_loss,4)
        train_acc,val_acc=round(train_acc,2),round(val_acc,2)

        epoch_time = round(time.time()-epoch_start_time,1)
        print(f'Epoch {epoch+1}, train_loss: {train_loss}, val_loss: {val_loss}, train_acc: {train_acc}, val_acc: {val_acc}, time taken: {epoch_time}s')

        scheduler.step(val_loss)

        # SAVE MODEL IF MIN VAL LOSS
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), model_dir)
            print(f'model epoch {epoch} saved at {model_dir}')
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= 10:
            print("Early stopping!")
            break

    total_time_taken=round(time.time()-trg_start_time,1)
    print(f'\nTraining ended: time elapsed: {total_time_taken}s')

    optimal_epoch = epoch

    
    ########################################################################
    # RETRAIN MODEL WITH ENTIRE TRAINING SET AFTER IDENTIFYING OPTIMAL EPOCH
    ########################################################################    
    print()
    print()
    print('Prepare retraining process')
    print()
    print('redefine training set as entire train images')

    # model_dir = model_dir[:-4]+'_retraining.pth'

    # DEFINE DATASET
    train_dataset = SceneDataset(image_paths,split_type='train',input_dim=input_dim) # use all image paths

    # DEFINE DATALOADER
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # DEFINE TRAINING VARIABLES
    # DEFINE MODEL
    model=ResNet50(15, 1)

    # ensure the weigshts and biases have same format as the input dtype
    model = model.to(torch.float32)
    model = model.to(device)
    print('model defined')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    min_train_loss = float('inf')

    # TRAIN MODEL - RETRAINING WITH ENTIRE TRAIN DATA
    train_losses,train_accs=[],[]

    trg_start_time = time.time()
    print('Start training - Retraining with entire dataset')

    for epoch in range(optimal_epoch):
        epoch_start_time = time.time()

        # TRAINING LOOP
        model.train()

        train_running_loss = []
        correct_predictions = 0
        total_predictions = 0
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            softmax_outputs=nn.Softmax(dim=1)(logits)
            loss = criterion(logits,labels)
            train_running_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            _, predicted_classes = torch.max(softmax_outputs, dim=1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)

        train_loss = np.mean(train_running_loss)
        train_acc = (correct_predictions / total_predictions) * 100
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # SAVE TRAINING STATISTICS
        train_loss=round(train_loss,4)
        train_acc=round(train_acc,2)

        epoch_time = round(time.time()-epoch_start_time,1)
        print(f'Epoch {epoch+1}, train_loss: {train_loss}, train_acc: {train_acc}, time taken: {epoch_time}s')

        scheduler.step(train_loss)

        # SAVE MODEL IF MIN TRAIN LOSS
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(model.state_dict(), model_dir)
            print(f'model epoch {epoch} saved at {model_dir}')

    total_time_taken=round(time.time()-trg_start_time,1)
    print(f'\nRetraining ended: time elapsed: {total_time_taken}s')

    return train_acc


def test(test_data_dir, model_dir, input_dim=224, batch_size=32):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """

    # load images
    test_image_paths = glob.glob(test_data_dir+'/**/*.jpg')
    # pass paths into dataset class - do transformations here
    test_dataset = SceneDataset(test_image_paths,split_type='test',input_dim=input_dim)
    # setup dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print('\ndata loaded and processed')

    # setup training variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # setup model - try VGGNet
    model=ResNet50(15, 1)

    # ensure the weigshts and biases have same format as the input dtype
    model = model.to(torch.float32)
    model = model.to(device)
    print('model defined')
    # load model
    model.load_state_dict(torch.load(model_dir,map_location=torch.device('cuda')))
    print('model loaded')

    print('Start evaluation')
    model.eval()
    correct_predictions_count = 0
    labels_count = 0

    for batch in tqdm(test_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        softmax_outputs=nn.Softmax(dim=1)(logits)

        _, predicted_classes = torch.max(softmax_outputs, dim=1) #dim=1 check for max in a row, dim=0 check for max in a col
        correct_predictions = (predicted_classes == labels).sum().item()
        correct_predictions_count += correct_predictions
        labels_count += labels.size(0)

    test_acc = round(correct_predictions_count/labels_count * 100,2)
    print(f'\ntest acc: {test_acc}')

    return test_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='test', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./test/', help='the directory of testing data')
    parser.add_argument('--train_model_dir', default='model.pth', help='set filename for new model you want to train')
    parser.add_argument('--test_model_dir', default='model_trained.pth', help='test the model our group already trained')
    opt = parser.parse_args()

    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.train_model_dir)
        print(f'training_accuracy: {training_accuracy}')

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.test_model_dir)
        print(f'testing_accuracy: {testing_accuracy}')
