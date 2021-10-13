import classification
from detection import LaysDetector
import os
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image


transform = T.Compose([
    T.ToTensor(),
])

#root = "data/for_detection/"
root = "data/for_testing/"
img_list = os.listdir(root)

classification_model = classification.load_model()


for img_name in img_list:
    img_name='test2.jpeg'
    path = root+img_name
    img = Image.open(path).convert("RGB")
    img = transform(img)


    detector = LaysDetector(path) 
    boxes = detector.get_boxes()
    #detector.draw_detection()

    #print(boxes)

    detected_images = []

    count=0

    folder_name='root'

    for box in boxes:
        count+=1
        x0,y0,x1,y1= [int(val) for val in box]
        imgx = img.permute(1,2,0)
        cropx=imgx[y0:y1,x0:x1]
        #print(type(cropx))
        #print(cropx.shape)
        #detected_images.append(cropx)

        save_image(cropx.permute(2, 0, 1), folder_name+'/sample{0}.png'.format(count))
        pathx = folder_name+'/sample{0}.png'.format(count)
        imgName='sample{0}.png'.format(count)

        #model = classification.load_model()
        imgo = Image.open(pathx)

        detected_flv = classification.predict_image(imgo, classification_model)
        print(imgName, " : ", detected_flv )
        detected_images.append(detected_flv)
    detector.draw_detection(detected_images)

    # for f in os.scandir(folder_name):
    #     os.remove(f)

    again = input('want to see next output (y/n):')
    if again == "y":
        continue
    elif again == "n":
        break