
import numpy as np
import matplotlib.pyplot as plt

def segmentador_alb(image, resultado, model='enet-cityscapes/enet-model.net',
                    classes='enet-cityscapes/enet-classes.txt',
                    colors='enet-cityscapes/enet-colors.txt', widthImg=500):
    import argparse
    import imutils
    import time
    import cv2
    
    clases = open(classes).read().strip().split('\n')
    if colors:
        colores = open(colors).read().strip().split('\n')
        colores = [np.array(c.split(",")).astype('int') for c in colores]
        colores = np.array(colores, dtype="uint8")
    else:
        np.random.seed(42)
        colores = np.random.randint(0, 255, size=(len(clases) - 1, 3), dtype="uint8")
        colores = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
    
    legend = np.zeros(((len(clases) * 25) + 25, 300, 3), dtype="uint8")
    # loop over the class names + colors
    for (i, (className, color)) in enumerate(zip(clases, colores)):
        color = [int(c) for c in color]
        cv2.putText(legend, className, (5, (i * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color), -1)

    net = cv2.dnn.readNet(model)
    img = cv2.imread(image)
    img = imutils.resize(img, width=widthImg)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (1024, 512), 0, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()

    print("[INFO] La inferencia tardó {:.4f} segundos".format(end - start))

    (numClasses, height, width) = output.shape[1:4]
    classMap = np.argmax(output[0], axis=0)
    mask = colores[classMap]
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    classMap = cv2.resize(classMap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    output = ((0.4 * img) + (0.6 * mask)).astype("uint8")
    
    #cv2.imwrite(resultado, output)
    
    #cv2.imshow("Leyenda", legend)
    #cv2.waitKey(0)
    
    #original = cv2.imread(image,-1)
    #original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    #res = cv2.imread(resultado,-1)
    #res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    
    #f = plt.figure(figsize=(25,15))
    #ax1 = f.add_subplot(121)
    # ax1.imshow(original)
    # ax1.set_title('Original')
    
    # ax2 = f.add_subplot(122)
    # ax2.imshow(res)
    # ax2.set_title('Imagen segmentada')
    # plt.show()
    
    return output,classMap
    
#%% 
    
def segmentador_Luis(original,resultado,show=None):
    
    import pixellib
    from pixellib.instance import instance_segmentation
    import cv2
    
    
    seg = instance_segmentation()
    seg.load_model("mask_rcnn_coco.h5")
    target_classes = seg.select_target_classes(car=True)
    
    segvalues, output = seg.segmentImage(original,segment_target_classes= target_classes,
                                         output_image_name=resultado, show_bboxes=True,
                                         extract_segmented_objects= True )
    #segment_image.segmentAsPascalvoc(original, output_image_name = resultado ,overlay = Superp)
    #cv2.imwrite(resultado, output)
    
    if show == True:
        img=cv2.imread(resultado)
        plt.imshow(img)
        plt.show()
    
    return segvalues, output

#%% 
#%% 
#Segmentador iñigo
def circle_points(resolution, center, radius1,radius2):
    """
    Generate points which define a circle on an image.Centre refers to the centre of the       circle
    """   
    import numpy as np
    radians = np.linspace(0, 2*np.pi, resolution)
    c = center[1] + radius1*np.cos(radians)#polar co-ordinates
    r = center[0] + radius2*np.sin(radians)
    points= np.array([c, r]).T
    return  points
#%% 

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def bb_intersection_over_union(boxA,boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def dibujarCuadrado (maxMinValues):
 
    minRow,minCol,maxRow,maxCol=maxMinValues
    valFila=list(range(minRow,maxRow))
    valCol= list(np.full(shape=maxRow-minRow,fill_value=minCol,dtype=np.int))
    valFila=valFila+ list(np.full(shape=maxCol-minCol,fill_value=minRow,dtype=np.int))
    valCol=valCol+list(range(minCol,maxCol))
    valFila=valFila+list(range(minRow,maxRow))
    valCol= valCol+list(np.full(shape=maxRow-minRow,fill_value=maxCol,dtype=np.int))
    valFila=valFila+ list(np.full(shape=maxCol-minCol,fill_value=maxRow,dtype=np.int))
    valCol=valCol+list(range(minCol,maxCol))
    return valFila,valCol
def showImages(img1,titleImg1,img2,titleImg2):
    """
    Función para enseñar en pantalla 2 imagenes
    """
    plt.figure(figsize=(16,10))
    plt.subplot(121), plt.imshow(img1),plt.title(titleImg1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2),plt.title(titleImg2)
    plt.xticks([]), plt.yticks([])
    plt.show()

def getRectangleValues(snake):
    #calculamos los valores maximos y minimos de las filas yo columnas.
    maxRow= int(max(snake[:,0]))
    maxCol= int(max(snake[:,1]))
    minRow= int(min(snake[:,0]))
    minCol=int(min(snake[:,1]))
    return [minRow,minCol,maxRow,maxCol]

def getVertices(img):
    minf=255
    minc=255
    maxf=0
    maxc=0
    for row in range(len(img)):
        for col in range(len(img[row])):
            if(img[row][col]==255):
                if(minf>row):
                    minf=row
                if(minc>col):
                    minc=col
                if(maxf<row):
                    maxf=row
                if(maxc<col):
                    maxc=col
    return [minc,minf,maxc,maxf]
