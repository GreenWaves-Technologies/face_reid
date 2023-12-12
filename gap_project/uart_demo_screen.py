import sys, serial, os, io
import PIL.Image as Image
import cv2
#Import all module inside tkinter
from tkinter import *
import tkinter.font as tkFont
#import pillow module and related module like
from PIL import Image,ImageTk
import numpy as np
from time import time
from scipy.spatial import distance


GAP_UART_BAUDRATE=1152000
UART_DEV='/dev/ttyUSB1'

THRESHOLD=0.50

UART_START_STREAM=b'\x0F\xF0'

def read_image_byte(ser):
    size_payload = ser.read(1)
    data_size = int.from_bytes(size_payload, "little")
    return data_size

def receive_image(ser,l,t,face_db):
    count=0
    start = time()
    while True:
        read_bytes = ser.read(2)
        #print(read_bytes)

        offset=0
        if read_bytes == UART_START_STREAM:
            try: 
                #size_payload = ser.read(1)
                #data_size = int.from_bytes(size_payload, "little")
                #print("num payloads:" + str(data_size) )
                scene = np.zeros((480, 800, 3), np.uint8) #320 left
                scene[:,480:,:] = 255
                while read_image_byte(ser)!=255:
                #for x in range(data_size):
                    coords_data = ser.read(4)
                    while len(coords_data) < 4*4:
                        coords_data += ser.read(1)
                    coord_data_array = np.frombuffer(coords_data, np.int32)

                    #print(coord_data_array)
                    x = coord_data_array[0]
                    y = coord_data_array[1]
                    w = coord_data_array[2]
                    h = coord_data_array[3]
                    #print('x:',x,' y: ',y,' w: ',w,' h: ',h)

                    image_size = w*h*3
                    #reading image
                    img_data = ser.read(3)
                    while len(img_data) < image_size:
                        img_data += ser.read(1)
                    #reading FaceID
                    faceid_data = ser.read(1)
                    while len(faceid_data) < 256:
                        faceid_data += ser.read(1)
                    face_id = np.frombuffer(faceid_data, np.float16)
                    #print(face_id)
                    face_bb = np.frombuffer(img_data, np.uint8)
                    face_bb = face_bb.reshape(h,w,3)
                    if x+w>480:
                        w=480-x
                    if y+h>=480:
                        h=480-y
                    if x<0:
                        x=0
                        w=w+x
                    if y<0:
                        h=h+y
                        y=0
                    ##Moved down
                    #scene[y:y+h,x:x+w,:]=face_bb[0:h,0:w,:]
                    
                    name=""
                    score=0
                    for entry in face_db:
                        score_tmp = 1-distance.cosine(face_id,entry['face_id'])
                        if score_tmp > score and score_tmp > THRESHOLD:
                            score = score_tmp
                            name = entry['name']
                        #print("name: ",name," score: ",score)
                    
                    if score>1:
                        score=1
                    if score<0:
                        score=0
                    
                    ## Preparing reclangle for results
                    rect_res = np.ones((120,320,3),np.uint8)
                    rect_res[0:120,0:10] = 125
                    rect_res[0:10,0:320] = 125
                    rect_res[0:120,-10:-1] = 125
                    rect_res[-10:-1,0:320] = 125
                    if name != "":
                        cv2.putText(rect_res, f'{name}',
                            (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(rect_res, f'score: {round(score,2)}%',
                            (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(rect_res, f'Unknown person',
                            (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
                        # cv2.putText(rect_res, f'score: {round(score,2)}%',
                        #     (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
                    scene[offset:offset+120,480:800,:]=rect_res
                    point_y=int(y+(h/2))
                    point_x=int(x+w/2)
                    cv2.line(scene, (point_x,point_y), (480,offset+60), (125, 125, 125), thickness=3, lineType=8)
                    scene[y:y+h,x:x+w,:]=face_bb[0:h,0:w,:]
                    offset=offset+120
                
                #scene = scene[:, :, ::-1]
                #cv2.imshow('Image from GAP', resized)
                #cv2.waitKey(1)
                resized_pil =Image.fromarray(scene)
                imgtk = ImageTk.PhotoImage(image=resized_pil)
                l.imgtk = imgtk
                l.configure(image=imgtk)
                
            except Exception as e: 
                print(e)
                print("Something went wrong with this image...")
                continue

            count += 1
            start = time()
            break


    l.after(1,receive_image,ser,l,t,face_db)
    

def load_face_db():
    # iterate over files in
    # that directory
    face_db = []
    for filename in os.listdir("signatures"):
        f = os.path.join("signatures", filename)
        # checking if it is a file
        if os.path.isfile(f):
            print("Loading signature from file:",f)
            face_db.append(
                {'name': filename.replace(".bin",""), 'face_id': np.fromfile(f, dtype=np.float16)}
                )
    return face_db

def main():
    #Init the window otherwise the protocol get screwed
    #create a tkinter window
    t=Tk()
    t.geometry("1200x1200")#here use alphabet 'x' not '*' this one
    #Create a label
    #app = Frame(t, bg="white")
    #app.grid()
    face_db = load_face_db()
    #print(face_db)
    
    l=Label(t,font="bold", width=480, height=480)
    l.place(x=120,y=120,width=480, height=480)
    l.pack(side = "bottom", fill = "both", expand = "yes")
    GLabel_413=Label(t)
    ft = tkFont.Font(family='Helvetica',size=30)
    GLabel_413["font"] = ft
    GLabel_413["fg"] = "#333333"
    GLabel_413["justify"] = "center"
    GLabel_413["text"] = "Greenwaves Technologies Face ReID"
    GLabel_413.place(x=0,y=0,width=1200,height=120)
    
    # #l.grid()
    # p_1 = Label(t,font="bold", width=300, height=112)
    # p_1.place(x=120+640,y=120+480,width=300, height=112)
    # p_1.pack(side = "right", fill = "both", expand = "yes")

    ser = serial.Serial(
        port=UART_DEV,\
        baudrate=GAP_UART_BAUDRATE,\
        parity=serial.PARITY_NONE,\
        stopbits=serial.STOPBITS_ONE,\
        bytesize=serial.EIGHTBITS,\
            timeout=0.01)
    ser.flushInput()
    ser.flushOutput()
    print("connected to: " + ser.portstr)
    count=0
    
    l.after(5,receive_image,ser,l,t,face_db)
    t.mainloop() 
    

    ser.close()

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit