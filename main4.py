import classification
from detection import LaysDetector
import os
from PIL import Image
import torchvision.transforms as T

import uvicorn
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


transform = T.Compose([
    T.ToTensor(),
])

transform_toPil = T.Compose([
    T.ToPILImage(),
])

#root = "data/for_detection/"
root = "data/for_testing/"
img_list = os.listdir(root)

classification_model = classification.load_model()


# for img_name in img_list:
#     img_name='test2.jpeg'
#     print(img_name)
#     path = root+img_name
#     img = Image.open(path).convert("RGB")
#     img = transform(img)
#
#     detector = LaysDetector(path)
#     boxes = detector.get_boxes()
#     #detector.draw_detection()
#
#     #print(boxes)
#
#     detected_images = []
#
#     count=0
#     for box in boxes:
#         count+=1
#         x0,y0,x1,y1= [int(val) for val in box]
#         imgx = img.permute(1,2,0)
#         cropx=imgx[y0:y1,x0:x1]
#
#         cropy = transform_toPil(cropx.permute(2, 0, 1))
#
#         imgName='Detected box {0}'.format(count)
#
#         #model = classification.load_model()
#
#         #imgo = Image.open(pathx)
#
#         detected_flv,detected_score = classification.predict_image(cropy, classification_model)
#         #print(imgName," : ", detected_flv ,' || dectected score :',detected_score)
#         detected_images.append((detected_flv,detected_score))
#
#     detector.draw_detection(detected_images)
#
#     again = input('want to see next output (y/n):')
#     if again == "y":
#         continue
#     elif again == "n":
#         break

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    #image = read_imagefile(await file.read())

    img = transform(image)

    detector = LaysDetector('')
    boxes = detector.get_boxes()
    # detector.draw_detection()

    # print(boxes)

    detected_images = []

    count = 0
    for box in boxes:
        count += 1
        x0, y0, x1, y1 = [int(val) for val in box]
        imgx = img.permute(1, 2, 0)
        cropx = imgx[y0:y1, x0:x1]

        cropy = transform_toPil(cropx.permute(2, 0, 1))

        imgName = 'Detected box {0}'.format(count)

        # model = classification.load_model()

        # imgo = Image.open(pathx)

        detected_flv, detected_score = classification.predict_image(cropy, classification_model)
        # print(imgName," : ", detected_flv ,' || dectected score :',detected_score)
        detected_images.append((detected_flv, detected_score))

    detector.draw_detection(detected_images)

    #prediction = predict(image)
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, debug=True)