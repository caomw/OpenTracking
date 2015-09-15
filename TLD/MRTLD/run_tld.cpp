#include <opencv2/opencv.hpp>
#include "tld_utils.h"
#include "mropencv.h"
#include <iostream>
#include <sstream>
#include "TLD.h"
#include <stdio.h>
using namespace cv;
using namespace std;
//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;
//��ȡ��¼bounding box���ļ������bounding box���ĸ����������Ͻ�����x��y�Ϳ��  
//����\datasets\06_car\init.txt�У���¼�˳�ʼĿ���bounding box���������� 
//142,125,232,164     
void readBB(char* file){
  ifstream bb_file (file);//�����뷽ʽ���ļ� 
  string line;
  getline(bb_file,line); //��������is�ж������ַ�����str�У��ս��Ĭ��Ϊ '\n'�����з���   
  istringstream linestream(line);//istringstream������԰�һ���ַ�����Ȼ���Կո�Ϊ�ָ����Ѹ��зָ�������  
  string x1,y1,x2,y2;
  getline (linestream,x1, ',');//��������is�ж������ַ�����str�У�ֱ�������ս��delim�Ž�����  
  getline (linestream,y1, ',');
  getline (linestream,x2, ',');
  getline (linestream,y2, ',');
  int x = atoi(x1.c_str());// = (int)file["bb_x"];//atoi �� �ܣ� ���ַ���ת����������   
  int y = atoi(y1.c_str());// = (int)file["bb_y"];
  int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
  int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
  box = Rect(x,y,w,h);
}
//bounding box mouse callback//������Ӧ���ǵõ�Ŀ������ķ�Χ�������ѡ��bounding box��   
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;//�Ѿ����bounding box  
    break;
  }
}

void print_help(char** argv){
  printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
  printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}
//�������г���ʱ�������в���   
void read_options(int argc, char** argv,VideoCapture& capture,FileStorage &fs){
  for (int i=0;i<argc;i++){
      if (strcmp(argv[i],"-b")==0){
          if (argc>i){
              readBB(argv[i+1]);//�Ƿ�ָ����ʼ��bounding box  
              gotBB = true;
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-s")==0){//����Ƶ�ļ��ж�ȡ  
          if (argc>i){
              video = string(argv[i+1]);
              capture.open(video);
              fromfile = true;
          }
          else
            print_help(argv);

      }
      if (strcmp(argv[i],"-p")==0){//��ȡ�����ļ�parameters.yml  
          if (argc>i){
              fs.open(argv[i+1], FileStorage::READ); //FileStorage��Ķ�ȡ��ʽ�����ǣ�FileStorage fs(".\\parameters.yml", FileStorage::READ);  
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-no_tl")==0){
          tl = false;
      }
      if (strcmp(argv[i],"-r")==0){
          rep = true;
      }
  }
}
//����ʼ֡���г�ʼ��������Ȼ����֡����ͼƬ���У������㷨����
int main(int argc, char * argv[]){
  VideoCapture capture;
  capture.open(0);// open the default camera
  //OpenCV��C++�ӿ��У����ڱ���ͼ���imwriteֻ�ܱ����������ݣ�������Ϊͼ���ʽ������Ҫ���渡  
  //�����ݻ�XML/YML�ļ�ʱ��OpenCV��C���Խӿ��ṩ��cvSave����������һ������C++�ӿ����Ѿ���ɾ����  
  //ȡ����֮����FileStorage�ࡣ   
  FileStorage fs;
  //ֱ�ӹ̶�����
  argc = 3;
  argv[0] = "./run_tld";
  argv[1] = "-p";
  argv[2] = "parameters.yml";
  argv[3] = "-s";
  argv[4] = "../datasets/06_car/car.mpg";
  argv[5] = "-b";
  argv[6] = "../datasets/06_car/init.txt";

  //Read options
  read_options(argc,argv,capture,fs);//���������в���  
  //Init camera
  if (!capture.isOpened())
  {
	cout << "capture device failed to open!" << endl;
    return 1;
  }
  //Register mouse callback to draw the bounding box
  cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback( "TLD", mouseHandler, NULL );//�������ص�
  //TLD framework
  TLD tld;
  //Read parameters file
  tld.read(fs.getFirstTopLevelNode());
  Mat frame;
  Mat last_gray;
  Mat first;
  if (fromfile){//���ָ��Ϊ���ļ���ȡ  
      capture >> frame;//����ǰ֡  
      cvtColor(frame, last_gray, CV_RGB2GRAY);//ת��Ϊ�Ҷ�ͼ��  
      frame.copyTo(first);//������Ϊ��һ֡  
  }else{//���Ϊ��ȡ����ͷ�������û�ȡ��ͼ���СΪ320x240   
      capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
      capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
  }

  ///Initialization
GETBOUNDINGBOX://��ţ���ȡbounding box   
  while(!gotBB)
  {
    if (!fromfile){
      capture >> frame;
    }
    else
      first.copyTo(frame);
    cvtColor(frame, last_gray, CV_RGB2GRAY);
    drawBox(frame,box);//��bounding box ������  
    imshow("TLD", frame);
    if (cvWaitKey(33) == 'q')
	    return 0;
  }
  if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
      cout << "Bounding box too small, try again." << endl;
      gotBB = false;
      goto GETBOUNDINGBOX;
  }
  //Remove callback
  cvSetMouseCallback( "TLD", NULL, NULL ); //����Ѿ���õ�һ֡�û��򶨵�box�ˣ���ȡ��
  printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height); 
  //Output file
  FILE  *bb_file = fopen("bounding_boxes.txt","w");
  //-------------- TLD initialization -----------------------
  tld.init(last_gray,box,bb_file);// ��һ֡�Ҷ�ͼ��last_gray������İ�Χ���ʼĿ��λ�ô洢��box������洢�ļ�bb_file

  ///Run-time
  Mat current_gray;
  BoundingBox pbox;
  vector<Point2f> pts1; //
  vector<Point2f> pts2; // 
  bool status=true;//��¼���ٳɹ�����״̬ lastbox been found  
  int frames = 1;//��¼�ѹ�ȥ֡��
  int detections = 1;//��¼�ɹ���⵽��Ŀ��box��Ŀ 
REPEAT://
  while(capture.read(frame)){
    //get frame
    cvtColor(frame, current_gray, CV_RGB2GRAY);//�Ҷ�ͼ�����Ͼ�������ɫ��Ϣ��������
    //--------------------------Process Frame----------------------
	//last_gray��current_gray�ֱ�����һ֡�͵�ǰ֡�ĻҶ�ͼ��pts1��pts2�Ǹ��ٵ�����꣬
	//��ʼ��Ϊ�գ��ں����ڲ�ʵ�ָ�ֵ��Ӧ�ã�pbox����һ֡���ٵĽ����
	//status��boolֵ���ڱ�ʾ��һ֡�Ƿ��и��ٵ����ڣ�
	//tl�������t��Ӧtrack��l��Ӧlearn��Ҳ��tl�Ǳ�ʾ�Ƿ����track��learn��bool��ǣ�
	//bb_file�Ǵ�Ÿ��ٽ�����ļ���·����
	//�����������ִ�������ĸ�ģ�飺����ģ�顢���ģ�顢�ۺ�ģ�顢ѧϰģ�顣
    tld.processFrame(last_gray,current_gray,pts1,pts2,pbox,status,tl,bb_file);
    //Draw Points
    if (status){ //������ٳɹ�  
      drawPoints(frame,pts1);
      drawPoints(frame,pts2,Scalar(0,255,0)); //��ǰ������������ɫ���ʾ  
      drawBox(frame,pbox);
      detections++;
    }
    //Display
    imshow("TLD", frame);
    //swap points and images
    swap(last_gray,current_gray);//STL����swap()���������������ֵ���䷺�ͻ��汾����
    pts1.clear();
    pts2.clear();
    frames++;
    printf("Detection rate: %d/%d\n",detections,frames);
    if (cvWaitKey(33) == 'q')
      break;
  }
  if (rep){//rep ����ѭ��
    rep = false;
    tl = false;
    fclose(bb_file);
    bb_file = fopen("final_detector.txt","w");
    //capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
    capture.release();
    capture.open(video);
    goto REPEAT;
  }
  fclose(bb_file);
  return 0;
}
