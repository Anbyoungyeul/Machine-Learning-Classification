import array
import cv2
import os
import np
import codecs
import glob




def convert_asm_to_images(sourcepath, destpath):
    files = os.listdir(sourcepath) # 모든 파일과 디렉토리 리턴 
    print('Sourcepath :', sourcepath)
    print('Destination path :', destpath)
    count =1
    for file in files:
        if file.endswith(".asm"):
            f = codecs.open(sourcepath+file, 'rb')
            ln = os.path.getsize(sourcepath+file)
            width = int(ln**0.5)
            rem = int(ln/width)
            a = array.array("B") # 바이트 요소 담기 위한 바이트 배열 선언 
            a.frombytes(f.read())
            f.close()
            g = np.reshape(a[:width * width], (width, width))
            g = np.uint8(g)
            cv2.imwrite(destpath+file+'.png', g)
            print(str(count) +'번째 :' + file)
            count += 1
    print('Files converted sucessfully')

def convert_start():
    sourcepath = r"D:\disasmbled\train__benign\\"
    destpath = r"D:\image\\"
    convert_asm_to_images(sourcepath, destpath)