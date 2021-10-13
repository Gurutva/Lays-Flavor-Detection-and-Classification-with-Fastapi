import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
import numpy as np
import torchvision.transforms as T
from torchvision.utils import save_image

transform = T.Compose([
    T.ToTensor(),
])


class LaysDetector:
    def __init__(self):
        self.model = self.get_model(num_classes=2)
        #self.imgPath = imgPath
        self.boxes = []
        self.prediction = None

    def get_model(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load("models/detection_model.zip" , map_location=torch.device('cpu')))
        return model

    def get_prediction(self, img):
        with torch.no_grad():
            self.prediction = self.model([img])
        return self.prediction        

    def draw_detection(self , detected_images,img):
        #img = self.imgPath
        self.model.eval()

        #img = Image.open(img).convert("RGB")
        #img = transform(img)
        prediction = self.get_prediction(img)

        image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        draw = ImageDraw.Draw(image)
        new_boxes = []
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()
            #           print(boxes)
            detection_score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=3)

            flv_name = detected_images[element][0]
            flv_score = detected_images[element][1]

            #out_text = flv_name + '=box {0}'.format(element) + '|p=' + str(detection_score) + '|c=' + str(flv_score)
            # out_text='box{0}'.format(element)
            out_text = flv_name

            if detection_score > 0.1 and detected_images[element][1] > 0.5:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="green", width=5)
                draw.text((boxes[0], boxes[1]), text=out_text, )
                new_boxes.append((element, boxes))
            # else:
            #     draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],outline="red", width=3)
            #     draw.text((boxes[0], boxes[1]), text=out_text, )

        image = transform(image)
        save_image(image, 'sample.png')

        return 'sample.png',new_boxes


    def get_boxes(self,img):
        #img = self.imgPath
        #ans_image = []
        self.model.eval()

        #img = Image.open(img).convert("RGB")
        #img = transform(img)
#        print(type(img))
#        print(img.shape)

        prediction = self.get_prediction(img)
        #print(prediction)
        return prediction[0]["boxes"].cpu().numpy()
