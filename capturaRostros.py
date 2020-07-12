#pylint:disable=W0604
import cv2
import tkinter as tk
from PIL import ImageTk, Image
import os
import imutils
from tkinter import simpledialog
import pygame_sdl2

global imgtk


root= tk.Tk()
root.withdraw()
USER_INP = simpledialog.askstring(title="Guardar Modelo como",prompt="Nombre:")
personName= USER_INP
dataPath = 'Data'
personPath = dataPath + '/' + personName
if not os.path.exists(personPath):
    os.makedirs(personPath)
root.destroy()
pygame_sdl2.key.stop_text_input()


count=0
root= tk.Tk()
root.rowconfigure(1, weight=1)
root.columnconfigure(1, weight=1)
titulo= tk.Label(root, text="Capturar modelo")
titulo.grid(row=0, column=0)
lmain = tk.Label(root)
lmain.grid(row=1, column=0, sticky="nsew")


# Initialize the camera with index 0
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    lmain.config(text="Unable to open camera: please grant appropriate permission in Pydroid permissions plugin and relaunch.\nIf this doesn't work, ensure that your device supports Camera NDK API: it is required that your device supports non-legacy Camera2 API.", wraplength=lmain.winfo_screenwidth())
    root.mainloop()
else:
    # You can set the desired resolution here
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


def refresh():
    ret, frame = cap.read()
    if not ret:
        # Error capturing frame, try next time
        lmain.after(0, refresh)
        return
    
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray,1.3,5)
    
    global count
    contador= tk.Label(root, text="caras encontradas "+str(count))
    contador.grid(row=2,column=0, pady=15)
    for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count+1),rostro)
            
            count = count + 1
            
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    w = lmain.winfo_screenwidth()
    h = lmain.winfo_screenheight()
    cw = cv2image.shape[0]
    ch = cv2image.shape[1]
    # In portrait, image is rotated
    cw, ch = ch, cw
    if (w > h) != (cw > ch):
        # In landscape, we have to rotate it
        cw, ch = ch, cw
        # Note that image can be upside-down, then use clockwise rotation
        cv2image = cv2.rotate(cv2image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Keep aspect ratio
    w = min(cw * h / ch, w)
    h = min(ch * w / cw, h)
    w, h = int(w), int(h)
    # Resize to fill the whole screen
    cv2image = cv2.resize(cv2image, (w, h), interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.configure(image=imgtk)
    lmain.update()
    lmain.after(0, refresh)
    contador.update()
    contador.after(0,refresh)


refresh()
root.mainloop()
