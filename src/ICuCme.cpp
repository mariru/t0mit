//============================================================================
// Name        : ICuCme.cpp
// Author      : Julian Straub
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened()){  // check if we succeeded
    	cout<<"No Cam!"<<endl;
        return -1;
    }

    CvHaarClassifierCascade *cascade_f = (CvHaarClassifierCascade*)cvLoad("../3rdparty/haarcascade_frontalface_alt.xml", 0, 0, 0);
    CvHaarClassifierCascade *cascade_e = (CvHaarClassifierCascade*)cvLoad("../3rdparty/haarcascade_eye.xml", 0, 0, 0);
    CvMemStorage* storage = cvCreateMemStorage(0);

    assert(cascade_f && cascade_e);

    Mat gray;
    namedWindow("normal",1);
    namedWindow("gray",2);
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        cout<<frame.cols<<"x"<<frame.rows<<"x"<<endl;
        cvtColor(frame, gray, CV_BGR2GRAY);
//        GaussianBlur(gray, gray, Size(7,7), 1.5, 1.5);
//        Canny(gray, gray, 0, 30, 3);
//        imshow("edges", gray);
        imshow("normal", frame);

        IplImage iplGray=gray;

        CvSeq *faces = cvHaarDetectObjects(&iplGray, cascade_f, storage,
        		1.1, 3, 0, cvSize( 40, 40 ) );

        /* return if not found */
        if (faces->total == 0){
        	cout<<"no face"<<endl;
        	continue;
        }

        /* draw a rectangle */
    	CvRect *r = (CvRect*)cvGetSeqElem(faces, 0);
    	cvRectangle(&iplGray,
    				cvPoint(r->x, r->y),
    				cvPoint(r->x + r->width, r->y + r->height),
    				CV_RGB(255, 0, 0), 1, 8, 0);

        /* reset buffer for the next object detection */
        cvClearMemStorage(storage);

        /* Set the Region of Interest: estimate the eyes' position */
        cvSetImageROI(&iplGray, cvRect(r->x, r->y + (r->height/5.5), r->width, r->height/3.0));

        /* detect eyes */
    	CvSeq* eyes = cvHaarDetectObjects(
            &iplGray, cascade_e, storage,
    		1.15, 3, 0, cvSize(25, 15));

        /* draw a rectangle for each eye found */
    	for(int i = 0; i < (eyes ? eyes->total : 0); i++ ) {
    		r = (CvRect*)cvGetSeqElem( eyes, i );
    		cvRectangle(&iplGray,
    					cvPoint(r->x, r->y),
    					cvPoint(r->x + r->width, r->y + r->height),
    					CV_RGB(255, 0, 0), 1, 8, 0);
    	}

        cvResetImageROI(&iplGray);

        Mat grayDisp(&iplGray);

        imshow("gray", grayDisp);
        waitKey(1);
        //if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
