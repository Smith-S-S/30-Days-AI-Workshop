import json
import base64
import requests
import cv2
#from Secret_Key import key

SECRET_KEY = 'sk_203d0c3dd0ff6bb769da7167'


IMAGE_PATH = 'images/car.jpg'
SECRET_KEY = 'sk_203d0c3dd0ff6bb769da7167'

with open(IMAGE_PATH, 'rb') as image_file:
    img_base64 = base64.b64encode(image_file.read())

url = 'https://api.openalpr.com/v3/recognize_bytes?recognize_vehicle=1&country=us&secret_key=%s' % (SECRET_KEY)
r = requests.post(url, data = img_base64)

#print(json.dumps(r.json(), indent=2))


num_plate=(json.dumps(r.json(), indent=2))
info= (list(num_plate.split("candidates")))

plate=info[1]


plate=plate.split(",")[0:3]
p=plate[1]
p1= p.split(":")
number=p1[1]
number=number.replace('"','')
number=number. lstrip()
print(number)

if number =="MH120E4433":
    print("-----------------------------")
    print ("Owner Name: Champ")
    print ("Vehicle Nmuber: %s" %number)
    print ("Address: MAHARASHTRA")



