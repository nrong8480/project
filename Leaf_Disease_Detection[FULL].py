import numpy as np
from pickle import load
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import sklearn


default_image_size = tuple((256, 256))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
with open('label_transform.pkl','rb') as f:
    loaded_label= load(f)

loaded_model = load_model("best_model")
#---------------------------------------------------------------------------------------------------------------------------------------------------#
from tkinter import *
from PIL import ImageTk, Image,ImageGrab
from tkinter import filedialog
from tkinter import ttk
import urllib
import imutils
from pynput.keyboard import Listener, Key

root = Tk()
root.title('Leaf Disease Detection')
root.iconbitmap('')
root.resizable(True, True)
root.geometry('800x500')


def file():
    
    filename = filedialog.askopenfilename(initialdir="/", title="Select An Image or Mp4 Video",filetypes=(("jpg files", "*.jpg"),("mp4 videos","*.mp4")))
    if(filename is ''):
        return
    elif filename.endswith('.mp4'):
        classes = []
        with open("YOLO-v4\data\classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                classes.append(class_name)
        cap = cv2.VideoCapture(filename)

        net = cv2.dnn.readNet(r"YOLO-v4\cfg\yolov4-custom_final.weights",r"YOLO-v4\cfg\yolov4-custom.cfg")
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1/255)

        while True:
            _, frame = cap.read()
            
            frame = imutils.resize(frame, width=1000, height=600)

            (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                (x, y, w, h) = bbox
                class_name = classes[class_id]
                if class_name == 'tomato':
                    loaded_model = load_model('Tomato_best_model')
                    loaded_label = load(open('Tomato_label_transform.pkl','rb'))
                elif class_name == 'potato':
                    loaded_model = load_model('Potato_best_model')
                    loaded_label = load(open('Potato_label_transform.pkl','rb'))
                elif class_name == 'pepper_bell':
                    loaded_model = load_model('PepperBell_best_model')
                    loaded_label = load(open('PepperBell_label_transform.pkl','rb'))

                cropped_image = frame[y:y+h, x:x+w]
                cv2.imwrite('cropped_image.jpg',cropped_image)

                def resized ():
                    resized_image = cv2.imread('cropped_image.jpg')
                    if resized_image is not None:
                        resized_image = cv2.resize(resized_image, default_image_size)
                        cv2.imwrite('resized_frame.jpg',resized_image)
                        return 'resized_frame.jpg'
                
                resized()

                image_dir = "resized_frame.jpg"
                im=convert_image_to_array(image_dir)
                np_image_li = np.array(im, dtype=np.float16) / 225
                npp_image = np.expand_dims(np_image_li, axis=0)

                result = loaded_model.predict(npp_image)
                check = np.max(result)
                itemindex = np.where(result==np.max(result))
                pred = ("Probability: "+str(round(np.max(result),3))+"  "+loaded_label.classes_[itemindex[1][0]])
                if(check >= 0.10):
                    cv2.putText(frame, pred,(x, y - 10), cv2.FONT_HERSHEY_PLAIN,1,(0,0,225),2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,225), 1)
            cv2.imshow("Leaf Disease Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                break

        cv2.destroyAllWindows()
        cap.release()
    global my_image
    global my_btn
    def resized():
        resized_image = cv2.imread(filename)
        if resized_image is not None:
            resized_image = cv2.resize(resized_image, default_image_size)
            cv2.imwrite('resized_image.jpg',resized_image)
            return 'resized_image.jpg'
    resized()
    image_dir = 'resized_image.jpg'
    im=convert_image_to_array(image_dir)
    np_image_li = np.array(im, dtype=np.float16) / 225
    npp_image = np.expand_dims(np_image_li, axis=0)
    loaded_model=load_model("best_model")
    result = loaded_model.predict(npp_image)
    with open('label_transform.pkl','rb') as f:
        loaded_label= load(f)
    itemindex = np.where(result==np.max(result))
    pred = ("\n"+"\n"+"Probability: "+str(round(np.max(result),3))+"\n"+loaded_label.classes_[itemindex[1][0]])

    
    my_label = Label(root, text=pred)    
    my_label.grid(row=1, column=1,sticky=N)
    my_image = ImageTk.PhotoImage(Image.open('resized_image.jpg'))
    my_image_label = Label(image=my_image)
    my_image_label.grid(row=2, column=1,rowspan=10,sticky=N)
    def close():
        my_label.destroy()
        my_image_label.destroy()
        _.destroy()
        
    _ = ttk.Button(root,text="   Close Image   ", command=close)
    if my_label is None:
        _.destroy()
    _.grid(row=5, column=0,pady=15,sticky=W)    

def camera():
    try:
        classes = []
        with open("YOLO-v4\data\classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                classes.append(class_name)
        cap = cv2.VideoCapture(0)
        
        net = cv2.dnn.readNet(r"YOLO-v4\cfg\yolov4-custom_final.weights",r"YOLO-v4\cfg\yolov4-custom.cfg")
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1/255)

        while True:
            _, frame = cap.read()
            
            frame = imutils.resize(frame, width=1000, height=600)

            (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                (x, y, w, h) = bbox
                
                class_name = classes[class_id]
                if class_name == 'tomato':
                    loaded_model = load_model('Tomato_best_model')
                    loaded_label = load(open('Tomato_label_transform.pkl','rb'))
                elif class_name == 'potato':
                    loaded_model = load_model('Potato_best_model')
                    loaded_label = load(open('Potato_label_transform.pkl','rb'))
                elif class_name == 'pepper_bell':
                    loaded_model = load_model('PepperBell_best_model')
                    loaded_label = load(open('PepperBell_label_transform.pkl','rb'))

                cropped_image = frame[y:y+h, x:x+w]
                cv2.imwrite('cropped_image.jpg',cropped_image)

                def resized():
                    resized_image = cv2.imread('cropped_image.jpg')
                    if resized_image is not None:
                        resized_image = cv2.resize(resized_image, default_image_size)
                        cv2.imwrite('resized_frame.jpg',resized_image)
                        return 'resized_frame.jpg'
                
                resized()

                image_dir = "resized_frame.jpg"
                im=convert_image_to_array(image_dir)
                np_image_li = np.array(im, dtype=np.float16) / 225
                npp_image = np.expand_dims(np_image_li, axis=0)

                result = loaded_model.predict(npp_image)
                check = np.max(result)
                itemindex = np.where(result==np.max(result))
                pred = ("Probability: "+str(round(np.max(result),3))+"  "+loaded_label.classes_[itemindex[1][0]])
                if(check >= 0.10):
                    cv2.putText(frame, pred,(x, y - 10), cv2.FONT_HERSHEY_PLAIN,1,(0,0,225),2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,225), 1)
            cv2.imshow("Leaf Disease Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                break

        cv2.destroyAllWindows()
        cap.release()
    except Exception as e:
        text_box.insert(0,"ERROR: "+str(e)+" !!!")
        return
def link():
    # urllib.request.urlretrieve('https://media.geeksforgeeks.org/wp-content/uploads/20210318103632/gfg-300x300.png', "loaded_image")    
    try:
        urllib.request.urlretrieve(text_box.get(), "loaded_image")
        filename = "loaded_image"
    except Exception as e:
        text_box.delete(0,"end")
        text_box.insert(0,"ERROR: "+str(e)+" PLEASE ENTER THE ACCURACY LINK !!!")
        return
    global my_image
    global my_btn
    text_box.delete(0,"end")
    
    
    def resized():
        resized_image = cv2.imread(filename)
        if resized_image is not None:
            resized_image = cv2.resize(resized_image, default_image_size)
            cv2.imwrite('resized_image.jpg',resized_image)
            return 'resized_image.jpg'
    resized()

    image_dir = 'resized_image.jpg'
    im=convert_image_to_array(image_dir)
    np_image_li = np.array(im, dtype=np.float16) / 225
    npp_image = np.expand_dims(np_image_li, axis=0)

    result=loaded_model.predict(npp_image)

    itemindex = np.where(result==np.max(result))
    pred = ("\n"+"\n"+"Probability: "+str(round(np.max(result),3))+"\n"+loaded_label.classes_[itemindex[1][0]])

    
    my_label = Label(root, text=pred)    
    my_label.grid(row=1, column=1,sticky=N)
    my_image = ImageTk.PhotoImage(Image.open('resized_image.jpg'))
    my_image_label = Label(image=my_image)
    my_image_label.grid(row=2, column=1,rowspan=10,sticky=N)
    def close():
        my_label.destroy()
        my_image_label.destroy()
        _.destroy()
        
    _ = ttk.Button(root,text="   Close Image   ", command=close)
    if my_label is None:
        _.destroy()
    _.grid(row=5, column=0,pady=15,sticky=W)


def clipboard():
    clipboard = ImageGrab.grabclipboard()
    clipboard.save('clipboard.png')
    filename = 'clipboard.png'
    global my_image
    global my_btn
    
    def resized():
        resized_image = cv2.imread(filename)
        if resized_image is not None:
            resized_image = cv2.resize(resized_image, default_image_size)
            cv2.imwrite('resized_image.jpg',resized_image)
            return 'resized_image.jpg'
            
    resized()
    image_dir = 'resized_image.jpg'
    im=convert_image_to_array(image_dir)
    np_image_li = np.array(im, dtype=np.float16) / 225
    npp_image = np.expand_dims(np_image_li, axis=0)

    result=loaded_model.predict(npp_image)

    itemindex = np.where(result==np.max(result))
    pred = ("\n"+"\n"+"Probability: "+str(round(np.max(result),3))+"\n"+loaded_label.classes_[itemindex[1][0]])

    
    my_label = Label(root, text=pred)    
    my_label.grid(row=1, column=1,sticky=N)
    my_image = ImageTk.PhotoImage(Image.open('resized_image.jpg'))
    my_image_label = Label(image=my_image)
    my_image_label.grid(row=2, column=1,rowspan=10,sticky=N)
    def close():
        my_label.destroy()
        my_image_label.destroy()
        _.destroy()
        
    _ = ttk.Button(root,text="    Close Image   ", command=close)
    if my_label is None:
        _.destroy()
    _.grid(row=5, column=0,pady=15,sticky=W)
root.columnconfigure(1,weight=10)
# root.rowconfigure(0, weight=2)
# root.rowconfigure(2, weight=2)
my_btn3 = ttk.Button(root,text="        Detect         ", command=link).grid(row=6, column=0,pady=15,sticky=NW)#expand=True,anchor=N
my_btn2 = ttk.Button(root,text="Realtime Detect", command=camera).grid(row=7, column=0,pady=15,sticky=W)#expand=True
my_btn = ttk.Button(root,text="    Choose File   ", command=file).grid(row=8, column=0,pady=15,sticky=SW)#expand=True,anchor=S
text_box = ttk.Entry(root,justify=CENTER)
text_box.grid(row=0, column=1,padx=50,ipadx=250,pady=15,sticky=N)#expand=True,fill=X,ipady=30,anchor=CENTER

# from pynput import mouse
# def onMouseClick(*args):
#     print(args)

# listener = mouse.Listener(on_click=onMouseClick)
def on_press(key):
    # print('{0} pressed'.format(key))
    if key == Key.right:
        try:
            clipboard()
            text_box.delete(0,"end")
        except Exception as e:
            text_box.delete(0,"end")
            text_box.insert(0,"ERROR: "+str(e)+" PLEASE PASTE THE ACCURACY IMAGE !!!")
            return

# def on_release(key):
    # print('{0} release'.format(key))
    # if key == Key.esc:
    #     return False

listener = Listener(on_press=on_press)#,on_release=on_release

listener.start()
root.mainloop()