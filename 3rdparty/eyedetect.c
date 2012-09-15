#include <stdio.h>
#include "cv.h"
#include "highgui.h"

CvHaarClassifierCascade *cascade_f;
CvHaarClassifierCascade *cascade_e;
CvMemStorage			*storage;

void detectEyes(IplImage *img);

int main(int argc, char** argv)
{
    IplImage *img;
	char *file1 = "haarcascade_frontalface_alt.xml";
    char *file2 = "haarcascade_eye.xml";

    /* usage: eyedetect <image> */
    assert(argc == 2);

    /* load the face classifier */
	cascade_f = (CvHaarClassifierCascade*)cvLoad(file1, 0, 0, 0);

    /* load the eye classifier */
    cascade_e = (CvHaarClassifierCascade*)cvLoad(file2, 0, 0, 0);

    /* setup memory storage, needed by the object detector */
    storage = cvCreateMemStorage(0);

    /* load image */
    img = cvLoadImage(argv[1], 1);

    /* always check */
    assert(cascade_f && cascade_e && storage && img);

    cvNamedWindow(argv[1], 1);

    /* detect eyes and display image */
    detectEyes(img);
    cvShowImage(argv[1], img);

    cvWaitKey(0);
    cvDestroyWindow(argv[1]);
    cvReleaseImage(&img);

    return 0;
}

void detectEyes(IplImage *img)
{
	int i;

    /* detect faces */
	CvSeq *faces = cvHaarDetectObjects(
		img, cascade_f, storage,
		1.1, 3, 0, cvSize( 40, 40 ) );

    /* return if not found */
    if (faces->total == 0) return;

    /* draw a rectangle */
	CvRect *r = (CvRect*)cvGetSeqElem(faces, 0);
	cvRectangle(img,
				cvPoint(r->x, r->y),
				cvPoint(r->x + r->width, r->y + r->height),
				CV_RGB(255, 0, 0), 1, 8, 0);

    /* reset buffer for the next object detection */
    cvClearMemStorage(storage);

    /* Set the Region of Interest: estimate the eyes' position */
    cvSetImageROI(img, cvRect(r->x, r->y + (r->height/5.5), r->width, r->height/3.0));

    /* detect eyes */
	CvSeq* eyes = cvHaarDetectObjects( 
        img, cascade_e, storage,
		1.15, 3, 0, cvSize(25, 15));

    /* draw a rectangle for each eye found */
	for( i = 0; i < (eyes ? eyes->total : 0); i++ ) {
		r = (CvRect*)cvGetSeqElem( eyes, i );
		cvRectangle(img, 
					cvPoint(r->x, r->y), 
					cvPoint(r->x + r->width, r->y + r->height),
					CV_RGB(255, 0, 0), 1, 8, 0);
	}

    cvResetImageROI(img);
}

