#include "mropencv.h"
#include "fstream"
string benchmarkpath = "E:\\Tracking\\Evaluation\\cvpr2013benchmark";
string videoname = "Basketball";

class MatchTemplateTracker
{
public:
	void init(Mat frame, Rect &box);
	void process(Mat frame, Rect & trackBox);
	Mat model;
};

void MatchTemplateTracker::init(Mat frame, Rect &box)
{
	Mat gray;
	cvtColor(frame, gray, CV_RGB2GRAY);
	model = gray(box);
}

void MatchTemplateTracker::process(Mat frame, Rect & trackBox)
{
	Mat gray;
	cvtColor(frame, gray, CV_RGB2GRAY);
	Rect searchWindow;
	searchWindow.width = trackBox.width * 3;
	searchWindow.height = trackBox.height * 3;
	searchWindow.x = trackBox.x + trackBox.width*0.5 - searchWindow.width*0.5;
	searchWindow.y = trackBox.y + trackBox.height * 0.5 - searchWindow.height * 0.5;
	searchWindow &= Rect(0, 0, frame.cols, frame.rows);
	Mat similarity;
	matchTemplate(gray(searchWindow), model, similarity, CV_TM_CCOEFF_NORMED);
	double mag, r;
	Point pt;
	minMaxLoc(similarity, 0, &mag, 0, &pt);
	trackBox.x = pt.x + searchWindow.x;
	trackBox.y = pt.y + searchWindow.y;
	model = gray(trackBox);
}

void tracking(Mat frame, Mat &model, Rect &trackBox)
{
	Mat gray;
	cvtColor(frame, gray, CV_RGB2GRAY);
	Rect searchWindow;
	searchWindow.width = trackBox.width * 3;
	searchWindow.height = trackBox.height * 3;
	searchWindow.x = trackBox.x + trackBox.width*0.5 - searchWindow.width*0.5;
	searchWindow.y = trackBox.y + trackBox.height * 0.5 - searchWindow.height * 0.5;
	searchWindow &= Rect(0, 0, frame.cols, frame.rows);
	Mat similarity;
	matchTemplate(gray(searchWindow), model, similarity, CV_TM_CCOEFF_NORMED);
	double mag, r;
	Point pt;
	minMaxLoc(similarity, 0, &mag, 0, &pt);
	trackBox.x = pt.x + searchWindow.x;
	trackBox.y = pt.y + searchWindow.y;
	model = gray(trackBox);
}

void index2string(int index, string &imgname)
{
	string fullpath = benchmarkpath + "\\";
	fullpath += videoname;
	fullpath += "\\img\\";
	char buffer[20];
	if (index < 10)
		sprintf_s(buffer, "000%d", index);
	else
	if (index < 100)
		sprintf_s(buffer, "00%d", index);
	else if (index < 1000)
		sprintf_s(buffer, "0%d", index);
	else
		sprintf_s(buffer, "%d", index);
	imgname = fullpath + buffer;
	imgname += ".jpg";
}

void readGroundth(Rect&box)
{
	string fullpath = benchmarkpath + "\\";
	fullpath += videoname;
	string gdpath = fullpath + "\\groundtruth_rect.txt";
	ifstream  gdfile(gdpath.c_str());
	char buffer[80];
	if(gdfile)
	{
		char dummy;
		gdfile >> box.y >> dummy;
		gdfile >> box.x>> dummy;
		gdfile >> box.width>> dummy;
		gdfile>>box.height;
	}
}

int main()
{
	Rect box;
	Mat frame, model, gray;
// 	VideoCapture capture;
// 	capture.open("E:\\Datasets\\bank1.avi");
// 	if (!capture.isOpened())
// 	{
// 		return -1;
// 	}
//	capture >> frame;
	int index = 1;
	string filename;
	index2string(index, filename);
	frame = imread(filename);
	index++;
	readGroundth(box);
	cvtColor(frame, gray,CV_RGB2GRAY);
	//model = gray(box);
	MatchTemplateTracker tracker;
	tracker.init(frame, box);
	while (1)
	{
//		capture >> frame;
		string filename;
		index2string(index, filename);
		frame = imread(filename);	
		if (frame.empty())
			return -1;
//		tracking(frame, model, box);
		tracker.process(frame, box);
		rectangle(frame, box, Scalar(0, 0, 255), 3);
		imshow("img", frame);
		if (cvWaitKey(20) ==27)
			break;
		index++;
	}
	return 0;
}