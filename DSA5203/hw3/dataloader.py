import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# define dataset class
class SceneDataset(Dataset):
    def __init__(self,image_paths,split_type,input_dim):
        self.image_paths = image_paths
        self.split_type = split_type
        self.input_dim = input_dim

        # # preload the data since its small only 1.5k images
        # print(f'{split_type}, preloading images and labels..')
        # self.images = [load_image(image_path=image_path, input_dim=input_dim) for image_path in tqdm(image_paths)]
        # self.labels = [image_path.split('/')[-2] for image_path in image_paths]

    def __len__(self): return len(self.image_paths)

    def __getitem__(self,index):

        image = load_image(image_path=self.image_paths[index], input_dim=self.input_dim)
        label = self.image_paths[index].split('/')[-2]

        label = label_mapping[label]-1 # need to -1 since the labels run from 1-15 but for processing need start from 0-14

        if self.split_type=='train':
            tfms = transforms.Compose([
                transforms.ToTensor(),
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomInvert(),
                transforms.GaussianBlur(kernel_size=5),
            ])
            image = tfms(image)

            return image,label

        else:
            # if val or test - only apply to tensor transformations
            tfms = transforms.Compose([
                transforms.ToTensor()
            ])
            image = tfms(image)

            return image,label
            
# DEFINE CLASS MAPPINGS
label_mapping = {
    "bedroom": 1,
    "Coast": 2,
    "Forest": 3,
    "Highway": 4,
    "industrial": 5,
    "Insidecity": 6,
    "kitchen": 7,
    "livingroom": 8,
    "Mountain": 9,
    "Office": 10,
    "OpenCountry": 11,
    "store": 12,
    "Street": 13,
    "Suburb": 14,
    "TallBuilding": 15
}
reverse_mapping = {v:k for k,v in label_mapping.items()}

# define load image function
def load_image(image_path,input_dim):
    # load image
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    # # resize
    image = cv2.resize(image,(input_dim,input_dim))

    return image