from sre_constants import SUCCESS
from cv2 import VideoWriter
import numpy as np
import cv2
import time #to set  frames time and fps
from os.path import splitext, basename, join


class Colorizer:
    # setting default resolution of the images and videos
    def __init__(self, height=480, width=600):
        (self.height, self.width) = height, width
    
                # Define Model Paths
    
        # read the model using dnn module of cv2
        # as the model is caffe model we will use readNetFromCaffe() 
        # and provide path to prototxt and caffe model files
        self.colorModel = cv2.dnn.readNetFromCaffe("model/colorization_deploy_v2.prototxt",
        caffeModel="model/colorization_release_v2.caffemodel")

        # this model contains pretrained cluster centroids which are provided as numpy dump
        
                # Load serialized black and white colorizer model and cluster
        
        # load the file using np.load()
        clusterCenters = np.load("model/pts_in_hull.npy")
        # take transpose of cluster centers and set the particular layer
        clusterCenters = clusterCenters.transpose().reshape(2, 313, 1, 1)

                # Add the cluster centers as 1x1 convolutions to the model

        # first layer is class8_ab
        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [clusterCenters.astype(np.float32)]
        # second layer is conv8_313_rh we set an array of 1 cross 313
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [
            np.full([1, 313], 2.606, np.float32)]

    def processImage(self, imgName):
        #this method takes image name as an argument reads that image and loads it in self.img
        self.img = cv2.imread(imgName)
        #resize image as per given width and height
        self.img = cv2.resize(self.img, (self.width, self.height))

        self.processFrame()
        # final image ready saved in output folder and also displayed on the screen
        cv2.imwrite(join("output", basename(imgName)), self.imgFinal)

        cv2.imshow("Output", self.imgFinal)

    def processVideo(self, videoName):#takes video path as input and tries to read the video
        cap = cv2.VideoCapture(videoName)

        # if reading unsuccessful return error message
        if (cap.isOpened() == False):
            print("Error opening video")
            return

        # else we read the frame
        (success, self.img) = cap.read()

        # As we want to display fps on the video declare previous and next frame variables
        # set  inital time to zero
        prevFrameTime = 0
        nextFrameTime = 0

        # we need to save the output video as well so we initialize 
        # VideoWriter() that takes the filename from the original video path and adds an extension .mp4 at the end 
        # and mp4v will be our encoder
        # we want to maintain the original fps for the video
        # set the width and height, multiply width by 2 
        # because we also want to append the original b&w frame for side by side comparison
        # so the width will be doubled
        
        out = VideoWriter(join("output", splitext(basename(videoName))[0] + '.mp4'),
        cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (self.width * 2, self.height))

        # run loop as long as the frame is successfully captured 
        while success:

            # resize the frame
            self.img = cv2.resize(self.img, (self.width, self.height))

            # call the procesFrame()
            # write the frame using VideoWriter()
            self.processFrame()
            out.write(self.imgFinal)

            # now we calculate the fps
            nextFrameTime = time.time()
            fps = 1 / (nextFrameTime - prevFrameTime)
            prevFrameTime = nextFrameTime

            # and convert it to string and put it on the top left corner 
            fps = "FPS:" + str(int(fps))
            # with offset of 5 pixel on the x-axis and 25 pixels on the y-axis 
            # note that we are putting the fps after we have saved the frame so the 
            # fps wont be shown on the output video
            # setting text color to white by setting all values to 255
            cv2.putText(self.imgFinal, fps, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            
            # to show final output on the scree
            cv2.imshow("Output", self.imgFinal)

            # wait for a key press and if 'q' pressed break the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # try to read the next frame
            (success, self.img) = cap.read()
        
        # if the loop ends release the video
        cap.release()
        out.release()
        # close all cv2 windows
        cv2.destroyAllWindows()

    def processFrame(self):
        # first we swap blue and red channels 
        # bcause opencv reads the image as BGR while the model is trained on RGB channels
        # Normalize the image by dividing with 255 this will convert all values between 0 and 1
        imgNormalized = (self.img[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)

        # converting RGB colorspace to lab colorspace
        #which seperates color info and luminance (ab and l).
        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2Lab)
        #save channel 0 which is called L channel and contains the luminance info
        channelL = imgLab[:, :, 0]

                # Extracting “L”:

        #now we need to resize lab image as the model is trained on 224 cross 224 image resolution
        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized, (224, 224)), cv2.COLOR_RGB2Lab)
        channelLResized = imgLabResized[:, :, 0]
        # save updated L channel
        channelLResized -= 50
        
                #Predicting “a” and “b”:

        # call the model and set input to resized Lchannel that we just extracted in line 133-137
        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        # if we do feed forward pass we will get the output as first element of the resultant array
        result = self.colorModel.forward()[0, :, :, :].transpose((1, 2, 0))
        # the obrtained result was 256 cross 256 so we need to resize it original resolution
        resultResized = cv2.resize(result, (self.width, self.height))

                # Creating a colorized Lab photo (L + a + b):


        # now we need to concatenate the original Lchannel with this result this
        # is because the result has color info and Lchannel has luminance info
        # and all these channels combined will form a colored image

        self.imgOut = np.concatenate((channelL[:, :, np.newaxis], resultResized), axis=2)
        # next we need to click all values between 0 and 1

                #Converting to RGB
        
        self.imgOut = np.clip(cv2.cvtColor(self.imgOut, cv2.COLOR_LAB2BGR), 0, 1)
        # finally we will denomarlize the imgOut by multiplying it with 255 and typecasting it to unsignned intger 8
        self.imgOut = np.array((self.imgOut) * 255, dtype=np.uint8)
        # stack it with original image for side by side comparison 

        self.imgFinal = np.hstack((self.img, self.imgOut))
        # call processFrame() in processImage()