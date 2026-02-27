import base64

local_pic = r"D:\develop\develop\workspace\pycharm\251020\shopkeeper_brain\knowledge\test\import\img\test.jpg"

with open(local_pic, "rb") as img:
    print(base64.b64encode(img.read()).decode("utf-8"))
