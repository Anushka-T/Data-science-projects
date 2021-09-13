import os
import numpy as np 
import pandas as pd
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
import matplotlib.patches as patches
import torchvision.transforms.functional as TF

class cells_image():
    '''
    This class is used to plot a blood smear image and plot and predict the bounding boxes and class labels of individual cells.

    Parameters
    --------------

    number: integer
    image id 

    json_name: string
    name of json file containing image data e.g. file path, ground truth bounding boxes etc.

    axis: matplotlib axes object
    the axes to plot the image on 

    '''
    def __init__(self, number, json_name, axis):
        self.number = number
        self.json_name = json_name
        self.path = json_name[number]['image']['pathname'][1:]
        self.ax = axis
    
    def plot_cells(self):
        '''
        Plot blood smear image
        '''
        cells = image.imread(self.path)
        self.ax.imshow(cells)
        self.ax.axis('off')
    
    def plot_ground(self):
        '''
        Plot ground truth bounding boxes and class labels, as specified in json file
        '''  
        for box in self.json_name[self.number]['objects']:     
            x_1, y_1 = box['bounding_box']['minimum']['r'], box['bounding_box']['minimum']['c']
            x_2, y_2 = box['bounding_box']['maximum']['r'], box['bounding_box']['maximum']['c']
            width = x_2 - x_1
            height = y_2 - y_1  
            
            if box['category'] == 'red blood cell':
                name = 'RBC'
            else:
                name = box['category']
            self.ax.text(y_1, x_1, name, color = 'b')
            rect = patches.Rectangle((y_1, x_1), width, height, linewidth=0.5, edgecolor='b', facecolor='none')
            self.ax.add_patch(rect)
            
    def detect(self, dataset, model):
        '''
        Predict bounding boxes and class labels using fine-tuned pre-trained DETR model
        '''
        cells = Image.open(self.path)
        pixel_values, labels = dataset[self.number]
        pixel_values = pixel_values.unsqueeze(0).to('cpu')
        pred = model(pixel_values)

        probas = pred['pred_logits'].softmax(-1)[0, :, :]
        keep = probas.max(-1).values > 0.85
        class_labels = probas[keep].argmax(axis=1).numpy()
        # Rescale bounding boxes
        scaled_boxes = self.rescale_bboxes(pred['pred_boxes'][0,keep].cpu(), cells.size).tolist()
        return scaled_boxes, class_labels
    
    def classify(self, scaled_boxes, model):
        '''
        Predict class labels using custom convolutional network
        '''
        predictions = []
        cells = Image.open(self.path)
        for i in scaled_boxes: 
            cell = cells.crop(tuple(i))
            cell = cell.resize((53,53))
            cell = TF.to_tensor(cell).unsqueeze_(0)   
            pred = model(cell)
            pred = torch.argmax(pred, dim=1)

            predictions.append(pred.tolist())
        
        return predictions
    
    def plot_pred(self, scaled_boxes, class_labels, id2label):
        '''
        Plot predicted bounding boxes and class labels.
        '''
        for i,j in zip(scaled_boxes, class_labels):
            rect = patches.Rectangle((i[0], i[1]), i[2]-i[0],i[3]-i[1], linewidth=0.5, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
            try: 
                text = id2label[j]
            except TypeError:
                text = id2label[j[0]]
            self.ax.text(i[0], i[1], text, fontsize=10, color = 'r')
    
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

def plot_class_counts(dataset_json):
    '''
    Plot bar chart of percentage of dataset that each class makes up
    '''
    class_counts = {}
    for slide in dataset_json:
        for box in slide['objects']:
            if box['category'] == 'red blood cell':
                box['category'] = 'RBC'
            class_counts[box['category']] = class_counts.get(box['category'], 0) + 1

    total = 0
    for i in class_counts.values():
        total += i

    for k,v in class_counts.items():
        class_counts[k] = (v/total)*100

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
    ax.bar(range(len(class_counts)), list(class_counts.values()))
    ax.set_title('Infected cells make up less than 5% of the dataset')
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(list(class_counts.keys()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_images_dir(directory, number, title):
    '''
    Plot specified number of images in a given directory
    '''
    fig, ax = plt.subplots(nrows=1, ncols=number, figsize = (20,20), squeeze=False)
    r = 0
    c = 0
    for i in os.listdir(directory)[0:number+1]:
        if not i.startswith('.'):   
            cell = image.imread(os.path.join(directory,i))
            ax[r][c].imshow(cell)
            ax[r][c].axis('off')
            if title:
                ax[r][c].set_title(i[0:-4])
            c+=1
            if c%number==0:
                r += 1
                c = 0
    plt.show()
    return

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    Plot  confusion matrix.
    '''
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def process_images(dataset, dim, folder):
    '''
    Extract each cell in an image, resize it and save to new directory according to its class label.
    '''
    for path in dataset['pathname'].unique():
        img = cv2.imread(path[1:], cv2.COLOR_BGR2RGB)
        data = dataset[dataset['pathname']==path]
        
        for index, row in data.iterrows():
            cell = img[row['y_min']:row['y_min']+row['height'],row['x_min']:row['x_min']+row['width']]
            cell = cv2.resize(cell, dim, interpolation = cv2.INTER_AREA) 

            #Center and normalise
            pixels = np.asarray(cell).astype('float32')
            mean = pixels.mean(axis=(0,1), dtype='float64')
            pixels = (pixels - mean)/255

            separator = '_'
            cell_name = separator.join([str(row['ann_id']), str(row['image_id'])])
            cell_name += '.png'
            directory = row['category']
            cell_name = os.path.join(folder, directory, cell_name)
            cv2.imwrite(cell_name, cell)
    return

def info(predictions):
    '''
    Output information about diagnosis, Plasmodium species and percentage parasitemia.
    '''
    counts_d = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    for i in predictions:
        counts_d[i[0]] += 1 
    if counts_d[1] > 0 or counts_d[2] > 0 or counts_d[4] > 0 or counts_d[5] > 0 or counts_d[6] > 0:
        print('Diagnosis: positive for malaria')
        print('Species: Plasmodium vivax')
        parasitsed = (counts_d[4] + counts_d[5] + counts_d[6])/(counts_d[3] + counts_d[4] + counts_d[5] + counts_d[6])
        print(f'Percentage parasitemia: {round(parasitsed*100,0)}')
    else:
        print('Diagnosis: negative for malaria')