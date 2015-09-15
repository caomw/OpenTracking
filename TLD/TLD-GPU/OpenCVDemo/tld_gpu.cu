#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <cuda_texture_types.h>////
#include<device_launch_parameters.h>///////�Ҽӵ�
#include<channel_descriptor.h>//
#include<texture_fetch_functions.h>//
#include<cuda_runtime.h>
#include<driver_types.h>
//#include<cutil.h>
using namespace cv;
using namespace std;
#define ele2D(BaseAddress, x, y,pitch) *((float*)((char*)(BaseAddress) + (y) * (pitch)) + (x));
texture<unsigned char, 2> imageData2D;//��ǰ֡��ͼ��ÿ֡�����
texture<float, 1> gridData1D;//���������ﶼ����
texture<unsigned char, 2> features2D;//���������ﶼ����
texture<float, 2> posteriors2D;//ÿһ֡ͼ����ܻ�䣬��Ϊ�и���
texture<float, 1> tf1D;//ÿһ֡ͼ���䣬����ͨ���˷������ʣ�µ�grid
texture<int,2> sumData2D;
texture<float,2>squmData2D;
texture<float,2>pEx2D;
texture<float,2>testData2D;
texture<float,2>nEx2D;
texture<float,2>example2D;
texture<float,2>batch2D;
//__constant__ char dev_features[6240];
//__constant__ float dev_lastb[4];
bool firstgetvar=true;
//bool first=true;
//bool firstlap=true;
////////////////////
//cudaArray* featuresArray;
//cudaArray* imageArray;
//cudaArray * posteriorsArray;
cudaArray* sumArray;
cudaArray* squmArray;
float *dev_grid;
float *dev_posteriors;
float *dev_pCounter;
float *dev_nCounter;
float *image;//���Ը���ΪcudaMallocPich��cudaMemcpy2D
float *dev_features;
float *dev_pEx;
float *dev_testData;
float *dev_nEx;
float *dev_example;
float *dev_batch;
float *dev_batchMatch;
float *dev_smatch;//���ģ��ƥ��Ķ���
float *dev_upPos;
int   *dev_upPosInd;
size_t img_pitch;
size_t pos_pitch;
size_t fea_pitch;
size_t pEx_pitch;
size_t testData_pitch;
size_t nEx_pitch;
size_t example_pitch;
size_t batch_pitch;
////////////////////
float* gridans;//getlappingker�˺������صĽ��
float *tfans;//varfilter�˺������ص����ݣ�ÿһ֡�ĳ��ȶ���64034�����Բ��ͷţ��ڶ��ε���ʱҲ�����ٷ����ڴ档
float *withVarans;//gpu���صĽ����12��64034��
float *swithVarans;//���ص�cpu�Ľ��
float *dev_varisPass;//������˵�ֵ���������˺���Ҫ�ã�������Ϊȫ�ֱ���
float *dev_filter1Ans;//һ���˺����Ľ������һ���˺���Ҫ��
float filter1_threshold;
///////////////////
//int img_data_size;
int img_w,img_h,gridl,gridw,maxThreads;
int sfeaturesw, sfeaturesh;
int postw,posth,patch_size;
int pEx_size=0,nEx_size=0;//��ס����ģ������
int ThreadNUM;
int dev_structSize;
int dev_nstructs;
__constant__ int patch[2];//patchSize�����ڴ棬�˺����������ã����Ұ�ƽ��Ҳ����

__constant__ float conthreshold;
__constant__ int congridw;
__constant__ int connstructs;
__constant__ int constructSize;

//�˺�������һ��grid������
__global__  void filter1ker(float *ans,int h,float threshold,int grid_w,int nstructs,int structSize)
{	
	//�߳�id
	//	int w=nstructs+2;
	int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=h) return;
	//����������ֱ���ڴ�ͼ�ϡ��õ����ͼ����ʵҲ���������ȶ����õ��ˡ�
	int index=tex1Dfetch(tf1D, tid);//�õ���ǰ�߳�Ҫ�����grid��ĵڼ���box.
	/*
	int box_x=tex1Dfetch(gridData1D,index*5);
	int box_y=tex1Dfetch(gridData1D,index*5+1);
	int box_w=tex1Dfetch(gridData1D,index*5+2);
	int box_h=tex1Dfetch(gridData1D,index*5+3);
	int scale_idx=tex1Dfetch(gridData1D,index*5+4);
	*/
	//�������ĸ��ĸ���ʱ��ȡ��ʽҲҪ��

	int box_x=tex1Dfetch(gridData1D,index);
	int box_y=tex1Dfetch(gridData1D,index+grid_w);
	int box_w=tex1Dfetch(gridData1D,index+grid_w*2);
	int box_h=tex1Dfetch(gridData1D,index+grid_w*3);
	int scale_idx=tex1Dfetch(gridData1D,index+grid_w*4);
	////////////	
	//����ferns��cpu���������classifier.getFeatures(patch,grid[i].sidx,ferns);
	int leaf;
	int x1,x2,y1,y2;
	//	int nstructs=10;
	//	int structSize =13;
	int imbig=0;
	float votes = 0;
	float point1,point2;
	for (int t=0;t<nstructs;t++){
		leaf=0;
		for (int f=0; f<structSize; f++){
			//ȡ�õ������
			//	x1=tex2D(features2D,t*structSize*4+f*4,scale_idx);
			//	y1=tex2D(features2D,t*structSize*4+f*4+1,scale_idx);
			//	x2=tex2D(features2D,t*structSize*4+f*4+2,scale_idx);
			//	y2=tex2D(features2D,t*structSize*4+f*4+3,scale_idx);
			x1=tex2D(features2D,t*structSize+f,scale_idx);
			y1=tex2D(features2D,t*structSize+structSize*nstructs+f,scale_idx);
			x2=tex2D(features2D,t*structSize+structSize*nstructs*2+f,scale_idx);
			y2=tex2D(features2D,t*structSize+structSize*nstructs*3+f,scale_idx);

			point1=tex2D(imageData2D,box_x+x1,box_y+y1);//��y1�У���x1�У�cpu�漴�����˼important
			point2=tex2D(imageData2D,box_x+x2,box_y+y2);
			//  if(patch[x1*box_w+y1]>patch[x2*box_w+y2])
			if(point1>point2)
				imbig=1;
			else
				imbig=0;
			leaf = (leaf<<1 )+ imbig;
		}
		// ferns[t]=leaf;
		ans[tid*(nstructs+2)+t]=leaf;
		votes += tex2D(posteriors2D,leaf,t);

	}
	float conf=votes;
	ans[tid*(nstructs+2)+nstructs]=conf;
	if(conf>threshold)
		ans[tid*(nstructs+2)+nstructs+1]=index;
	else
		ans[tid*(nstructs+2)+nstructs+1]=-1;

}
//*/
__global__ void varfilterker(float *tfans,int grid_w,float var){
	//ÿ���˺�������һ��box
	int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=grid_w) return;
	int index=tid;
	int box_x=tex1Dfetch(gridData1D,index);
	int box_y=tex1Dfetch(gridData1D,index+grid_w);
	int box_w=tex1Dfetch(gridData1D,index+grid_w*2);
	int box_h=tex1Dfetch(gridData1D,index+grid_w*3);
	int scale_idx=tex1Dfetch(gridData1D,index+grid_w*4);
	float brs =tex2D(sumData2D,box_x+box_w,box_y+box_h);//double brs = sum.at<int>(box.y+box.height,box.x+box.width);
	//sum.at<int>(x,y)ȡ���ǵ�x�е�y�У�tex2D(sumData2D,x,y)ȡ���ǵ�y�е�x��
	float bls =tex2D(sumData2D,box_x,box_y+box_h);	//double bls = sum.at<int>(box.y+box.height,box.x);
	float trs =tex2D(sumData2D,box_x+box_w,box_y);//double trs = sum.at<int>(box.y,box.x+box.width);
	float tls =tex2D(sumData2D,box_x,box_y);//double tls = sum.at<int>(box.y,box.x);
	float brsq =tex2D(squmData2D,box_x+box_w,box_y+box_h);//double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
	float blsq =tex2D(squmData2D,box_x,box_y+box_h);	//double blsq = sqsum.at<double>(box.y+box.height,box.x);
	float trsq =tex2D(squmData2D,box_x+box_w,box_y);//double trsq = sqsum.at<double>(box.y,box.x+box.width);
	float tlsq =tex2D(squmData2D,box_x,box_y);//double tlsq = sqsum.at<double>(box.y,box.x);
	float mean = (brs+tls-trs-bls)/((float)box_w*box_h);//double mean = (brs+tls-trs-bls)/((double)box.area());
	float sqmean = (brsq+tlsq-trsq-blsq)/((float)box_w*box_h);//double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
	float temp=sqmean-mean*mean;	//return sqmean-mean*mean;
	if(temp>=var)
		tfans[tid]=tid;
	else
		tfans[tid]=-1;
}
float * Allvarfiltercu(const int *ssum,float *ssqum,int w,int h,float var){
	//�󶨵�����
	int ssum_data_size = sizeof(float) * w*h;
	int ssqum_data_size = sizeof(float) * w*h;
	if(firstgetvar)
	{
		cudaChannelFormatDesc chDesc6 = cudaCreateChannelDesc<int>();	
		cudaChannelFormatDesc chDesc7 = cudaCreateChannelDesc<float>();	
		cudaMallocArray(&sumArray, &chDesc6, w, h);
		cudaMallocArray(&squmArray, &chDesc7, w, h);
	}
	cudaMemcpyToArray( sumArray, 0, 0, ssum, ssum_data_size, cudaMemcpyHostToDevice);	
	cudaMemcpyToArray( squmArray, 0, 0, ssqum, ssqum_data_size, cudaMemcpyHostToDevice);
	if(firstgetvar){
		cudaBindTextureToArray( sumData2D, sumArray);	
		cudaBindTextureToArray( squmData2D, squmArray);
	}
	//�ú˺���ȥ����ÿ���߳���һ��box
	dim3 blocks((gridw+255)/256);
	dim3 threads(256);
	varfilterker<<<blocks,threads>>>(tfans,gridw,var);
	cudaThreadSynchronize();	
	float *stfans=new float[gridw];//����˭��˭�ͷš�
	cudaMemcpy( stfans, tfans, gridw*sizeof(float), cudaMemcpyDeviceToHost);
	firstgetvar=false;
	return stfans;
}
void filter1cucha(const unsigned char *simg, float *varisPass,float * filter1Ans,int varis_index)
{
	cudaMemcpy2D(image, img_pitch, simg, sizeof(unsigned char) * img_w, sizeof(unsigned char) * img_w, img_h, cudaMemcpyHostToDevice);
	//��varisPass���ƹ�ȥ 
	cudaMemcpy( dev_varisPass, varisPass, varis_index * sizeof( float ), cudaMemcpyHostToDevice );	
	///////////////////////
	//cudaMemcpy2D(dev_posteriors, pos_pitch, sposteriors, sizeof(float) * postw, sizeof(float) * postw, posth, cudaMemcpyHostToDevice);
	//float *ans;//filter1ker�˺������ص����ݣ�ÿһ֡ͼ��Ҫ�ͷţ���Ϊÿһ֡ʱvaris_indexֵ��һ��������������Ҫ��ans�Ĵ�С��һ��
	//12*varis_index��С��ǰ10��ferns,��11����measure_forest��ֵconf����12��,���conf������ֵ���������i�����-1,varis_index��tf�ĳ���
	//�������ƻ�ÿ���̴߳���һ������	,�˺���Ҫ���ferns,conf,i
	dim3 blocks((varis_index+255)/256);
	dim3 threads(256);		
	//���������Ĵ�С��(nstructs+2)*varis_index��С
	filter1ker<<< blocks, threads>>>( dev_filter1Ans, varis_index,filter1_threshold,gridw,dev_nstructs,dev_structSize);
	cudaThreadSynchronize();
	cudaMemcpy(filter1Ans, dev_filter1Ans, (dev_nstructs+2)*varis_index*sizeof(float) , cudaMemcpyDeviceToHost);
	//cudaMemcpy(out,dev_posteriors,8192*10*4,cudaMemcpyDeviceToHost);����ʱ���������Դ����ݲ鿴�õ�
}

__global__  void filter1ker1(float *ans,int varis_index)
{	
	extern __shared__ float shared[];
	//	int w=nstructs+2;
	int tidx=threadIdx.x;
	int tidy=threadIdx.y;//����Ҫ��features2D��ȡ��tidx�飬tidy��
	int tid=threadIdx.x*blockDim.y+threadIdx.y;//int nstructs,int structSize���������������ٴ��ˣ����Դ���ȡ
	//int bid=blockIdx.x;
	int bid=blockIdx.x*gridDim.y+blockIdx.y;
	if(bid>=varis_index) return;
	int index=tex1Dfetch(tf1D, bid);//�õ���ǰ�߳�Ҫ�����grid��ĵڼ���box.
	//�������ĸ��ĸ���ʱ��ȡ��ʽҲҪ��
	int box_x=tex1Dfetch(gridData1D,index);
	int box_y=tex1Dfetch(gridData1D,index+congridw);
	int box_w=tex1Dfetch(gridData1D,index+congridw*2);
	int box_h=tex1Dfetch(gridData1D,index+congridw*3);
	int scale_idx=tex1Dfetch(gridData1D,index+congridw*4);
	int i;
	////////////	
	//����ferns
	shared[tid]=0;
	int x1,x2,y1,y2;
	float point1,point2;	
	x1=tex2D(features2D,tidx*constructSize+tidy,scale_idx);
	y1=tex2D(features2D,tidx*constructSize+constructSize*connstructs+tidy,scale_idx);
	x2=tex2D(features2D,tidx*constructSize+constructSize*connstructs*2+tidy,scale_idx);
	y2=tex2D(features2D,tidx*constructSize+constructSize*connstructs*3+tidy,scale_idx);
	point1=tex2D(imageData2D,box_x+x1,box_y+y1);//��y1�У���x1�У�cpu�漴�����˼important
	point2=tex2D(imageData2D,box_x+x2,box_y+y2);
	if(point1>point2)
		shared[tid]=1<<(constructSize-1-tidy);//Ϊ��leaf׼��

	__syncthreads();
	////////////////////////////////////////////�Ż�////////////////////////////////////////////////////////
	///*
	for(i=blockDim.y;i>1;){
		if(tidy<i/2){
			i=(i+1)/2;
			shared[tid]+=shared[tid+i];
		}
		else
			break;
		__syncthreads();
	}
	if(tidy==0)
	{
		ans[bid*(connstructs+2)+tidx]=shared[tid];//��leafд�롣**********************************
	}
	if(tidy==1)
	{
		shared[tid]= tex2D(posteriors2D,shared[tid-1],tidx);//��ÿ��votes���ܹ�10������ÿ�еĵڶ����̼߳��㣬�����Ժ�浽�ڶ��̵߳�shared�����浽��һ���̵߳�shared�𣬷���Ҫͬ��
	}
	__syncthreads();
	//leaf��һ���߳��㣬conf��һ��block�㣬�Ƚϳ����Ҳ��һ��block�㣬����tid=0���㣬ǧ����ö���߳��غϵ���

	if(tid==1)
	{
		for(i=1;i<connstructs;i++)
			shared[1]+=shared[i*constructSize+1];

	}	
	/*��������Ż�������ԣ���֪��ʲô����
	if(tidy==1)//�ղŰ�votes���洢��ÿ�еĵڶ���shared��
	{
	for(i=blockDim.x;i>1;)
	{
	if(tidx<i/2)
	{
	i=(i+1)/2;
	shared[tid]+=shared[tid+i*blockDim.y];
	}
	else
	break;
	__syncthreads();
	}
	}*/
	__syncthreads();
	if(tid==0)
		ans[bid*(connstructs+2)+connstructs]=shared[1];//
	if(tid==1)
	{
		if(shared[1]>conthreshold)
			ans[bid*(connstructs+2)+connstructs+1]=index;//*********************************
		else
			ans[bid*(connstructs+2)+connstructs+1]=-1;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//������ԭ���ģ��������Ż���
	/*
	if(tidy==0)
	{
	for(i=1;i<constructSize;i++)
	shared[tid]+=shared[tid+i];//��leaf�����߳̿飨10�У��飩13�У���ÿ��13����ÿ���߳���һ�������ӵ���һ���߳�	
	ans[bid*(connstructs+2)+tidx]=shared[tid];//��leafд�롣**********************************
	shared[tid]= tex2D(posteriors2D,shared[tid],tidx);//��ÿ��votes���ܹ�10����
	}
	__syncthreads();
	//leaf��һ���߳��㣬conf��һ��block�㣬�Ƚϳ����Ҳ��һ��block�㣬����tid=0���㣬ǧ����ö���߳��غϵ���
	if(tid==0)
	{
	for(i=1;i<connstructs;i++)
	shared[0]+=shared[i*constructSize];
	ans[bid*(connstructs+2)+connstructs]=shared[0];//*********************************conf
	if(shared[0]>conthreshold)
	ans[bid*(connstructs+2)+connstructs+1]=index;//*********************************
	else
	ans[bid*(connstructs+2)+connstructs+1]=-1;
	}
	*/
	///////////////////////////////////////////////////////////////////////////////
}

void filter1cu(unsigned char *simg, float *varisPass,float * filter1Ans,int varis_index,double &f1datacost)
{   //�Ż�filter1cu�����Գ���������������һ���ģ�ֻ�����̵߳���֯�ṹ����
	//cudaHostRegister((void*)simg,sizeof(unsigned char) * img_w*img_h,1);
	cudaMemcpy2D(image, img_pitch, simg, sizeof(unsigned char) * img_w, sizeof(unsigned char) * img_w, img_h, cudaMemcpyHostToDevice);
	//��varisPass���ƹ�ȥ 
	cudaMemcpy( dev_varisPass, varisPass, varis_index * sizeof( float ), cudaMemcpyHostToDevice );	
	///////////////////////
	//cudaMemcpy2D(dev_posteriors, pos_pitch, sposteriors, sizeof(float) * postw, sizeof(float) * postw, posth, cudaMemcpyHostToDevice);
	//float *ans;//filter1ker�˺������ص����ݣ�ÿһ֡ͼ��Ҫ�ͷţ���Ϊÿһ֡ʱvaris_indexֵ��һ��������������Ҫ��ans�Ĵ�С��һ��
	//(nstructs+2)*varis_index��С��ǰ10��ferns,��11����measure_forest��ֵconf����12��,���conf������ֵ���������i�����-1,varis_index��tf�ĳ���
	//�������ƻ�ÿ���̴߳���һ������	,�˺���Ҫ���ferns,conf,i
	//dim3 blocks(varis_index);//ÿ������һ��ͼ
	double fangx= sqrt((double)varis_index);
	int fang= int(fangx);
	//dim3 blocks(varis_index/65535+1,65535);//ÿ������һ��ͼ
	dim3 blocks(varis_index/fang+1,fang);
	dim3 threads(dev_nstructs,dev_structSize);//ÿ���߳���һ���Ե㣬ÿһ���߳���һ�飬ÿ����13�������ɡ�//��Ϊdev_nstructs�е��̣߳�dev_structSize�е��߳�		
	//���������Ĵ�С��(nstructs+2)*varis_index��С
	filter1ker1<<< blocks, threads,dev_nstructs*dev_structSize*sizeof(float)>>>( dev_filter1Ans,varis_index);
	cudaThreadSynchronize();
	cudaMemcpy(filter1Ans, dev_filter1Ans, (dev_nstructs+2)*varis_index*sizeof(float) , cudaMemcpyDeviceToHost);
	//cudaMemcpy(out,dev_posteriors,8192*10*4,cudaMemcpyDeviceToHost);����ʱ���������Դ����ݲ鿴�õ�
	f1datacost=0;
}

__global__ void fiterWithVarker(float *ans,int grid_w,float var,int w,float threshold)
{
	int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=grid_w) return;
	int index=tid;
	int box_x=tex1Dfetch(gridData1D,index);
	int box_y=tex1Dfetch(gridData1D,index+grid_w);
	int box_w=tex1Dfetch(gridData1D,index+grid_w*2);
	int box_h=tex1Dfetch(gridData1D,index+grid_w*3);
	int scale_idx=tex1Dfetch(gridData1D,index+grid_w*4);
	float brs =tex2D(sumData2D,box_x+box_w,box_y+box_h);//double brs = sum.at<int>(box.y+box.height,box.x+box.width);
	//sum.at<int>(x,y)ȡ���ǵ�x�е�y�У�tex2D(sumData2D,x,y)ȡ���ǵ�y�е�x��
	float bls =tex2D(sumData2D,box_x,box_y+box_h);	//double bls = sum.at<int>(box.y+box.height,box.x);
	float trs =tex2D(sumData2D,box_x+box_w,box_y);//double trs = sum.at<int>(box.y,box.x+box.width);
	float tls =tex2D(sumData2D,box_x,box_y);//double tls = sum.at<int>(box.y,box.x);
	float brsq =tex2D(squmData2D,box_x+box_w,box_y+box_h);//double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
	float blsq =tex2D(squmData2D,box_x,box_y+box_h);	//double blsq = sqsum.at<double>(box.y+box.height,box.x);
	float trsq =tex2D(squmData2D,box_x+box_w,box_y);//double trsq = sqsum.at<double>(box.y,box.x+box.width);
	float tlsq =tex2D(squmData2D,box_x,box_y);//double tlsq = sqsum.at<double>(box.y,box.x);
	float mean = (brs+tls-trs-bls)/((float)box_w*box_h);//double mean = (brs+tls-trs-bls)/((double)box.area());
	float sqmean = (brsq+tlsq-trsq-blsq)/((float)box_w*box_h);//double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
	//float temp=sqmean-mean*mean;	//return sqmean-mean*mean;
	if((sqmean-mean*mean)<var)
	{
		ans[tid*w+11]=-2;
		return;
	}
	////////////	
	//����ferns��cpu���������classifier.getFeatures(patch,grid[i].sidx,ferns);
	int leaf;
	int x1,x2,y1,y2;
	int nstructs=10;
	int structSize =13;
	int imbig=0;
	float votes = 0;
	float point1,point2;
	for (int t=0;t<nstructs;t++){
		leaf=0;
		for (int f=0; f<structSize; f++){
			//ȡ�õ������
			x1=tex2D(features2D,t*structSize*4+f*4,scale_idx);
			y1=tex2D(features2D,t*structSize*4+f*4+1,scale_idx);
			x2=tex2D(features2D,t*structSize*4+f*4+2,scale_idx);
			y2=tex2D(features2D,t*structSize*4+f*4+3,scale_idx);
			point1=tex2D(imageData2D,box_x+x1,box_y+y1);//��y1�У���x1�У�cpu�漴�����˼important
			point2=tex2D(imageData2D,box_x+x2,box_y+y2);
			//  if(patch[x1*box_w+y1]>patch[x2*box_w+y2])
			if(point1>point2)
				imbig=1;
			else
				imbig=0;
			leaf = (leaf<<1 )+ imbig;
		}
		// ferns[t]=leaf;
		ans[tid*w+t]=leaf;
		votes += tex2D(posteriors2D,leaf,t);
	}	
	ans[tid*w+10]=votes;
	if(votes>threshold)
		ans[tid*w+11]=index;
	else
		ans[tid*w+11]=-1;
	return;		
}
float *fiterWithVarcu(const int *ssum,float *ssqum,int w,int h,float var,const unsigned char *simg, float threshold, float *sposteriors)
{
	int ssum_data_size = sizeof(float) * w*h;
	int ssqum_data_size = sizeof(float) * w*h;
	if(firstgetvar)
	{
		cudaChannelFormatDesc chDesc6 = cudaCreateChannelDesc<int>();	
		cudaChannelFormatDesc chDesc7 = cudaCreateChannelDesc<float>();	
		cudaMallocArray(&sumArray, &chDesc6, w, h);
		cudaMallocArray(&squmArray, &chDesc7, w, h);
	}
	cudaMemcpyToArray( sumArray, 0, 0, ssum, ssum_data_size, cudaMemcpyHostToDevice);	
	cudaMemcpyToArray( squmArray, 0, 0, ssqum, ssqum_data_size, cudaMemcpyHostToDevice);
	if(firstgetvar){
		cudaBindTextureToArray( sumData2D, sumArray);	
		cudaBindTextureToArray( squmData2D, squmArray);
	}
	//�ú˺���ȥ����ÿ���߳���һ��box
	cudaMemcpy2D(image, img_pitch, simg, sizeof(unsigned char) * img_w, sizeof(unsigned char) * img_w, img_h, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dev_posteriors, pos_pitch, sposteriors, sizeof(float) * postw, sizeof(float) * postw, posth, cudaMemcpyHostToDevice);

	dim3 blocks((gridw+255)/256);
	dim3 threads(256);
	fiterWithVarker<<<blocks,threads>>>(withVarans,gridw,var,12,threshold);		
	cudaThreadSynchronize();
	cudaMemcpy(swithVarans, withVarans, 12*gridw*sizeof(float) , cudaMemcpyDeviceToHost);//����������ͷ�
	firstgetvar=false;
	return swithVarans;
}
__global__ void getlappingker(float box1_x,float box1_y,float box1_w,float box1_h, int grid_w,float *gridans,float * dev_grid)
{	
	int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=grid_w)
		return;
	float box2_x=tex1Dfetch(gridData1D,tid);
	float box2_y=tex1Dfetch(gridData1D,tid+grid_w);
	float box2_w=tex1Dfetch(gridData1D,tid+grid_w*2);
	float box2_h=tex1Dfetch(gridData1D,tid+grid_w*3);
	if((box1_x > box2_x+box2_w)||(box1_y > box2_y+box2_h)||(box1_x+box1_w < box2_x)||(box1_y+box1_h < box2_y)){
		dev_grid[tid+grid_w*5]=0;		
		gridans[tid]=0;
		return;
	}
	float colInt =  min(box1_x+box1_w,box2_x+box2_w) - max(box1_x, box2_x);
	float rowInt =  min(box1_y+box1_h,box2_y+box2_h) - max(box1_y,box2_y);
	float intersection = colInt * rowInt;
	float area1 = box1_w*box1_h;
	float area2 = box2_w*box2_h;
	float answ=intersection / (area1 + area2 - intersection);
	dev_grid[tid+grid_w*5]=answ;
	gridans[tid]=answ;
	return;
}

//���һ��10��С��goodbox,��bad_box
void getlappingcu(float * lastb,float *sgridans)
{
	dim3 blocks((gridw+255)/256);
	dim3 threads(256);
	getlappingker<<<blocks,threads>>>(lastb[0],lastb[1],lastb[2],lastb[3],gridw,gridans,dev_grid);
	cudaMemcpy(sgridans, gridans, sizeof(float)*gridw, cudaMemcpyDeviceToHost);//�Ѻ˺����ļ��������ƻ�cpu
}
//ר�ż���һ��init����������Щһ���Դ���Ķ���,�����Դ�
//����Ҫ����������learn�����Ȱ�getOverlapping,ͬʱ����bad_box�ĸ��£��������ø��˺�������good_box�ĸ���
void gpuParam(int nstructs ,int structSize){
	//���structSize����23ʱ��pow(2.0,structSize)������������ܻᳬ��init���ܱ�ʾ�ķ�Χ������cpu��ĳ���һ��structSizeһ������23���ͻᱨ����
	postw=pow(2.0,structSize);
	dev_nstructs=nstructs;
	dev_structSize=structSize;
	posth=nstructs;
	cudaMemcpyToSymbol((char *) &connstructs,(void *)&dev_nstructs,sizeof(int));//���ص������ڴ���
	cudaMemcpyToSymbol((char *) &constructSize,(void *)&dev_structSize,sizeof(int));//���ص������ڴ���
}
void initNccData(int patch_s) //���뵥����һ��Ncc�����ݳ�ʼ��������������������Դ棬��Ϊ��tld.init()��Ϳ�ʼ����pEx��nEx����ʹ��NCConf��
{  //���������tld�Ĺ��캯����tld.init֮������
	int patchs[2];
	patchs[0]=patch_s;
	patchs[1]=patch_s*patch_s;
	patch_size=patch_s;
	cudaMemcpyToSymbol((char *) patch,(void *)patchs,sizeof(int)*2);//���ص������ڴ���
	//������������С����patch_size*patch_size*100��pEx��nEx
	cudaMallocPitch((void**)(&dev_pEx), &pEx_pitch, sizeof(float) * patch_size, patch_size*100);//��������float����������Ҳֻ����float���͵ġ�
	cudaChannelFormatDesc pExDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, pEx2D, dev_pEx, pExDesc, patch_size, patch_size*100, pEx_pitch);
	//nEx
	cudaMallocPitch((void**)(&dev_nEx), &nEx_pitch, sizeof(float) * patch_size, patch_size*100);
	cudaChannelFormatDesc nExDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, nEx2D, dev_nEx, nExDesc, patch_size, patch_size*100, nEx_pitch);
	//
	//batch��������example,���100��
	cudaMallocPitch((void**)(&dev_batch), &batch_pitch, sizeof(float) * patch_size, patch_size*100);
	cudaChannelFormatDesc batchDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, batch2D, dev_batch, batchDesc, patch_size, patch_size*100, batch_pitch);
	//posteriors�ĳ�ʼ��������cpuÿ�θ��¶��к������µ�gpu�����Բ����ٴ���
	cudaMallocPitch((void**)(&dev_posteriors), &pos_pitch, sizeof(float) * postw, posth);
	cudaChannelFormatDesc posDesc = cudaCreateChannelDesc<float>();
	cudaMemset(dev_posteriors,0,pos_pitch*posth);//��ʼ��ֵ,���������ÿ��posteriors�ĸ��¶����£���Ҫ�����ʼ��0
	//cudaMemset2D(dev_posteriors,pos_pitch,0,sizeof(float) * postw,posth);//��ʼ��ֵ,���������ÿ��posteriors�ĸ��¶����£���Ҫ�����ʼ��0	
	cudaBindTexture2D(NULL, posteriors2D, dev_posteriors, posDesc, postw, posth, pos_pitch);
	//example,������Ҫ�����ģ��
	cudaMallocPitch((void**)(&dev_example), &example_pitch, sizeof(float) * patch_size, patch_size);
	cudaChannelFormatDesc exampleDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, example2D, dev_example, exampleDesc, patch_size, patch_size, example_pitch);
	cudaMalloc((void **) &dev_smatch,200*sizeof(float));//��ģ��ƥ��gpu�ĺ˺�������һ���洢������Դ棬һ�η����һ���Դ棬��������ÿ�ζ������Դ棬��ʡʱ��
	cudaMalloc((void **) &dev_batchMatch,200*100*sizeof(float));//��������ͼ���ker����һ���洢�����ȫ���Դ棬����Ҫ(pEx.size()+nEx.size())*detections��С��һ�η��䣬�ظ����ã�����ÿ�ζ����䣬��ʡʱ��
	//����update��posterios�����õĴ�СΪnstructs*2��float����
	cudaMalloc((void **) &dev_upPos,dev_nstructs*sizeof(float));
	cudaMalloc((void **) &dev_upPosInd,dev_nstructs*sizeof(float));
}
void initGpuData(float filter1threshold,int img_ww,int img_hh,float *sgrid ,int gridww,unsigned char *sfeatures,int sfeaturesww, int sfeatureshh)
{	
	filter1_threshold=filter1threshold;
	img_w=img_ww;
	img_h=img_hh;
	gridw=gridww;
	gridl=gridww*6;
	sfeaturesw=sfeaturesww;
	sfeaturesh=sfeatureshh;
	cudaMemcpyToSymbol((char *) &conthreshold,(void *)&filter1_threshold,sizeof(float));//���ص������ڴ���
	cudaMemcpyToSymbol((char *) &congridw,(void *)&gridw,sizeof(int));//���ص������ڴ���

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);	
	maxThreads=prop.maxThreadsPerBlock;


	//��img������������1��Ҫ��
	cudaMallocPitch((void**)(&image), &img_pitch, sizeof(unsigned char) * img_w, img_h);
	cudaChannelFormatDesc imgDesc = cudaCreateChannelDesc<unsigned char>();
	cudaBindTexture2D(NULL, imageData2D, image, imgDesc, img_w, img_h, img_pitch);
	///////////////////////////////////////
	//grid
	int grid_data_size = sizeof(float) * gridl;
	cudaMalloc((void**)&dev_grid,grid_data_size);
	cudaMemcpy(dev_grid,sgrid,grid_data_size,cudaMemcpyHostToDevice);
	cudaBindTexture(0,gridData1D,dev_grid);
	////////////////////////////////////
	//feature
	cudaMallocPitch((void**)(&dev_features), &fea_pitch, sizeof(unsigned char) * sfeaturesw, sfeaturesh);
	cudaChannelFormatDesc feaDesc = cudaCreateChannelDesc<unsigned char>();
	cudaMemcpy2D(dev_features, fea_pitch, sfeatures, sizeof(unsigned char) * sfeaturesw, sizeof(unsigned char) * sfeaturesw, sfeaturesh, cudaMemcpyHostToDevice);
	cudaBindTexture2D(NULL, features2D, dev_features, feaDesc, sfeaturesw, sfeaturesh, fea_pitch);
	/////////////////////////////////////
	/*
	//posteriors
	cudaMallocPitch((void**)(&dev_posteriors), &pos_pitch, sizeof(float) * postw, posth);
	cudaChannelFormatDesc posDesc = cudaCreateChannelDesc<float>();
	for(int i=0;i<posth;i++)
	cudaMemcpy2D((char*)dev_posteriors+i*pos_pitch, pos_pitch, posteriorsP[i], sizeof(float) * postw, sizeof(float) * postw, 1, cudaMemcpyHostToDevice);
	cudaBindTexture2D(NULL, posteriors2D, dev_posteriors, posDesc, postw, posth, pos_pitch);
	//////////////////////////////////
	*/
	cudaMalloc( (void**)&dev_varisPass, gridw * sizeof( float ) );//��ŷ�����˺�����ݣ�������ݴ�cpu����������Ҫ�洢Ϊ������kerʹ�ã���󲻳���gridw������һ�η��䣬��ʡʱ��
	cudaBindTexture(0, tf1D, dev_varisPass);
	/*
	cudaMallocPitch((void**)(&dev_pCounter), &pCo_pitch, sizeof(float) * postw, posth);
	cudaChannelFormatDesc pCoDesc = cudaCreateChannelDesc<float>();
	cudaMemcpy2D(dev_pCounter, pCo_pitch, spCounter, sizeof(float) * postw, sizeof(float) * postw, posth, cudaMemcpyHostToDevice);	
	cudaBindTexture2D(NULL, pCounter2D, dev_pCounter, pCoDesc, postw, posth, pCo_pitch);

	/////////////////////////////////
	cudaMallocPitch((void**)(&dev_nCounter), &nCo_pitch, sizeof(float) * postw, posth);
	cudaChannelFormatDesc nCoDesc = cudaCreateChannelDesc<float>();
	cudaMemcpy2D(dev_nCounter, nCo_pitch, snCounter, sizeof(float) * postw, sizeof(float) * postw, posth, cudaMemcpyHostToDevice);	
	cudaBindTexture2D(NULL, nCounter2D, dev_nCounter, nCoDesc, postw, posth, nCo_pitch);
	/////////////////////////////////
	*/
	//gridans
	cudaMalloc( (void**)&gridans, sizeof(float)*gridw);	//���˺����Ľ������ռ�洢
	cudaMalloc( (void**)&tfans, sizeof(float)*gridw );	//��varfilterker�˺����Ľ������ռ�洢    
	cudaMalloc( (void**)&withVarans, 12*gridw*sizeof(float) );
	swithVarans=new float[12*gridw*sizeof(float) ];//ÿ��ͼ��һ��������ڴ�Ҳ��һ�������Բ�����Ϊȫ��

	/////////////////////////////////
	cudaMalloc( (void**)&dev_filter1Ans, (dev_nstructs+2)*gridw*sizeof(float) );//����varis_index�Ĵ�С�Ǳ�ģ������ǿ�����������ڴ档

}
//�ͷ��Դ�
void endGpuData()
{
	cudaUnbindTexture( imageData2D );
	cudaUnbindTexture( gridData1D );
	cudaUnbindTexture(features2D );
	cudaUnbindTexture( posteriors2D );	
	cudaUnbindTexture( tf1D );
	cudaUnbindTexture(pEx2D);
	cudaUnbindTexture(nEx2D);
	cudaUnbindTexture(example2D);
	cudaUnbindTexture(batch2D);
	cudaFree( dev_features );
	cudaFree( dev_posteriors);
	cudaFree(dev_pEx);
	cudaFree(dev_nEx);
	cudaFree(dev_example);
	cudaFree(dev_batch);
	cudaFree(image);
	cudaFree(dev_grid);
	cudaFree(gridans);
	cudaFree(tfans);
	cudaFree(withVarans);
	cudaFree(dev_smatch);
	cudaFree(dev_batchMatch);	
	cudaFree(dev_varisPass);
	cudaFree(dev_filter1Ans);
	cudaFree(dev_upPos);
	cudaFree(dev_upPosInd);
	pEx_size=0,nEx_size=0;
	delete [] swithVarans;
}

__global__ void updatePoker(float *dev_posteriors,int *dev_upPosInd,float* dev_upPos,int pos_pitch )
{
	//int threadNum=blockDim.x;
	int tid=threadIdx.x;
	int idx=dev_upPosInd[tid];
	float var=dev_upPos[tid];
	*((float*)((char *)dev_posteriors+pos_pitch*tid)+idx)=var;
}

void updatePoscu(float *upPos,int *upPosInd)
{
	cudaMemcpy(dev_upPos,upPos,dev_nstructs*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_upPosInd,upPosInd,dev_nstructs*sizeof(int),cudaMemcpyHostToDevice);
	updatePoker<<<1,dev_nstructs,0>>>(dev_posteriors,dev_upPosInd,dev_upPos,pos_pitch);
	//	*(float*)((char *)dev_posteriors+pos_pitch*2+143)=0.9;
}
void addpExcu(const float *spEx)//�Ѹ��µ�pEx���ݼ��ص�gpu
{
	cudaMemcpy2D((float*)((char*)dev_pEx+patch_size*pEx_size*pEx_pitch), pEx_pitch, spEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	pEx_size++;
}

void addpExcu(const float *spEx,int position,bool full)
{
	if(!full)
	{
		cudaMemcpy2D((float*)((char*)dev_pEx+patch_size*pEx_size*pEx_pitch), pEx_pitch, spEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	pEx_size++;	
	}
	else
	{
		cudaMemcpy2D((float*)((char*)dev_pEx+patch_size*position*pEx_pitch), pEx_pitch, spEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	

	}
}


void addnExcu1(const float *snEx)//�Ѹ��µ�nEx���ݼ��ص�gpu
{	
	cudaMemcpy2D((float*)((char*)dev_nEx+patch_size*nEx_size*nEx_pitch), nEx_pitch, snEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	nEx_size++;	

}
void addnExcu(const float *snEx)//�Ѹ��µ�nEx���ݼ��ص�gpu
{	
	cudaMemcpy2D((float*)((char*)dev_nEx+patch_size*nEx_size*nEx_pitch), nEx_pitch, snEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	nEx_size++;	

}
void addnExcu(const float *snEx,int position,bool full)
{
	if(!full)
	{
		cudaMemcpy2D((float*)((char*)dev_nEx+patch_size*nEx_size*nEx_pitch), nEx_pitch, snEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	nEx_size++;	
	}
	else
	{
		cudaMemcpy2D((float*)((char*)dev_nEx+patch_size*position*nEx_pitch), nEx_pitch, snEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	

	}
}
__global__  void NNConfker(float *dev_smatch,int pEx_s)//ע��˺���ֻ�ܷ����Դ���Ķ�������������Ҫôͨ���������ݻ���ȫ���ڴ�
{   

	extern __shared__ float shared[];
	int tid=threadIdx.x;//blockDim.x=patch_size
	int bid=blockIdx.x;
	int threadNum=blockDim.x;
	int photox;;//��photox��
	int photoy;;//��photo��
	int  offset=(threadNum+1)/2;;
	//int offset =(patch_size+1)/2;
	float pi,ei;
	int i;
	shared[tid*3]=0;
	shared[tid*3+1]=0;
	shared[tid*3+2]=0;
	if(bid<pEx_s){//˵�����block�ô�����ģ����example�ĶԱ�
		for(i=tid;i<patch[1];i+=threadNum)//һ���߳̿���Ҫ����ü���ͼ���
		{
			photox=i/patch[0];
			photoy=i%patch[0];
			ei=tex2D(example2D,photoy,photox);
			pi=tex2D(pEx2D,photoy,bid*patch[0]+photox);
			shared[tid*3]+=ei*pi;
			shared[tid*3+1]+=ei*ei;
			shared[tid*3+2]+=pi*pi;
		}
		__syncthreads();
		////////////////////////////////////////////////////////////////
		for(i=blockDim.x;i>1;)
		{
			if(tid<i/2)
			{
				i=(i+1)/2;
				shared[tid*3] += shared[(tid+i)*3];
				shared[tid*3+1] += shared[(tid+i)*3+1];
				shared[tid*3+2] += shared[(tid+i)*3+2];
			}
			else 
				break;
			__syncthreads();
		}
		__syncthreads();
		/*
		while(offset>0) {
		if(tid<offset) {
		if(tid+offset<threadNum)
		{
		shared[tid*3] += shared[(tid+offset)*3];
		shared[tid*3+1] += shared[(tid+offset)*3+1];
		shared[tid*3+2] += shared[(tid+offset)*3+2];
		}
		}
		threadNum=offset;
		if(offset==1)
		offset=0;
		else
		offset =(offset+1)/2;

		__syncthreads();
		}
		*/
		if(tid<2)
			shared[tid+1]=sqrt((float) shared[tid+1]);
		__syncthreads();
		if(tid == 0) {
			shared[1]=shared[1]*shared[2];
			dev_smatch[bid] = (shared[0]/shared[1]+1)/2;  
		}
	}
	else
	{

		if(bid-pEx_s<0) return;//����nex��СΪ0
		for(i=tid;i<patch[1];i+=threadNum)//һ���߳̿���Ҫ����ü���ͼ���
		{
			photox=i/patch[0];
			photoy=i%patch[0];
			ei=tex2D(example2D,photoy,photox);
			pi=tex2D(nEx2D,photoy,(bid-pEx_s)*patch[0]+photox);//�����������ȥpEx
			shared[tid*3]+=ei*pi;
			shared[tid*3+1]+=ei*ei;
			shared[tid*3+2]+=pi*pi;
		}
		__syncthreads();   
		////////////////////////////////////////////
		for(i=blockDim.x;i>1;)
		{
			if(tid<i/2)
			{
				i=(i+1)/2;
				shared[tid*3] += shared[(tid+i)*3];
				shared[tid*3+1] += shared[(tid+i)*3+1];
				shared[tid*3+2] += shared[(tid+i)*3+2];
			}
			else
				break;
			__syncthreads();
		}
		__syncthreads();
		//���Ż�
		/*
		while(offset>0) {
		if(tid<offset) {
		if(tid+offset<threadNum)
		{
		shared[tid*3] += shared[(tid+offset)*3];
		shared[tid*3+1] += shared[(tid+offset)*3+1];
		shared[tid*3+2] += shared[(tid+offset)*3+2];
		}
		}
		threadNum=offset;
		if(offset==1)
		offset=0;
		else
		offset =(offset+1)/2;


		__syncthreads();
		}
		*/
		if(tid<2)
			shared[tid+1]=sqrt((float) shared[tid+1]);
		__syncthreads();
		if(tid == 0) {
			shared[1]=shared[1]*shared[2];
			dev_smatch[bid] = (shared[0]/shared[1]+1)/2;  
		}
	}
}
void NNConfcu(const float *sexample,float *smatch)
{
	cudaMemcpy2D(dev_example, example_pitch, sexample, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	//��pEx��nEx�Ĵ�С����Դ�ĳ��������ﻹ�ǲ����������ǰ�pEx_size���ɲ��������˺���
	dim3 blocks(pEx_size+nEx_size);//�����������Ƕ�����һ���߳̿鴦��һ��Сͼ��
	//dim3 threads(patch_size,patch_size);//Ҫ��gpu����,��˲��ܴ���512����������Ҳ��patch_size���ܴ���24�����Զ��������ÿ��С�̶�������һ���鴦�������ص�
	if((patch_size*patch_size)>maxThreads)
		ThreadNUM=maxThreads;
	else
		ThreadNUM=patch_size*patch_size;//�������Բ�����ֶ�����߳���
	dim3 threads(ThreadNUM);
	NNConfker<<< blocks, threads,ThreadNUM*sizeof(float)*3>>>(dev_smatch,pEx_size);
	cudaThreadSynchronize();
	cudaMemcpy(smatch,dev_smatch,(pEx_size+nEx_size)*sizeof(float),cudaMemcpyDeviceToHost);
}
__global__ void NNConfBatchker(float *dev_batchMatch,int pEx_s)
{
	//�͵���example����һ���ģ���ͬ���ǣ�ȡexampleʱȡ����batch2D�ĵ�blockIDx.x��ͼ��
	//����ԭ��blockIDx��һά������ֻ��x���꣬���ڰ�x��y���껥��һ�¾�����
	//д��ʱ����д��һ����blockIdx��ص�dev_batchMatch��


	extern __shared__ float shared[];
	int tid=threadIdx.x;//blockDim.x=patch_size
	int bid=blockIdx.y;
	int bidx=blockIdx.x;
	int threadNum=blockDim.x;
	int photox;;//��photox��
	int photoy;;//��photo��
	int  offset=(threadNum+1)/2;;
	//int offset =(patch_size+1)/2;
	float pi,ei;
	int i;
	shared[tid*3]=0;
	shared[tid*3+1]=0;
	shared[tid*3+2]=0;
	if(bid<pEx_s){//˵�����block�ô�����ģ����example�ĶԱ�
		for(i=tid;i<patch[1];i+=threadNum)//һ���߳̿���Ҫ����ü���ͼ���
		{
			photox=i/patch[0];
			photoy=i%patch[0];
			ei=tex2D(batch2D,photoy,photox+bidx*patch[0]);//ȡbatch2D��bidx��ͼ����������bidx*patch[0]
			pi=tex2D(pEx2D,photoy,bid*patch[0]+photox);
			shared[tid*3]+=ei*pi;
			shared[tid*3+1]+=ei*ei;
			shared[tid*3+2]+=pi*pi;
		}
		__syncthreads();
		////////////////////////////////////////////////////////////////////
		///*
		for(i=blockDim.x;i>1;)
		{
			if(tid<i/2)
			{
				i=(i+1)/2;
				shared[tid*3] += shared[(tid+i)*3];
				shared[tid*3+1] += shared[(tid+i)*3+1];
				shared[tid*3+2] += shared[(tid+i)*3+2];
			}
			else
				break;
			__syncthreads();
		}
		__syncthreads();
		//*/
		/*
		while(offset>0) {
		if(tid<offset) {
		if(tid+offset<threadNum)
		{
		shared[tid*3] += shared[(tid+offset)*3];
		shared[tid*3+1] += shared[(tid+offset)*3+1];
		shared[tid*3+2] += shared[(tid+offset)*3+2];
		}
		}
		threadNum=offset;
		if(offset==1)
		offset=0;
		else
		offset =(offset+1)/2;


		__syncthreads();
		}
		*/
		if(tid<2)
			shared[tid+1]=sqrt((float) shared[tid+1]);
		__syncthreads();
		if(tid == 0) {
			shared[1]=shared[1]*shared[2];
			dev_batchMatch[bidx*gridDim.y+bid] = (shared[0]/shared[1]+1)/2;  //д����        
		}
	}
	else
	{

		if(bid-pEx_s<0) return;//����nex��СΪ0
		for(i=tid;i<patch[1];i+=threadNum)//һ���߳̿���Ҫ����ü���ͼ���
		{
			photox=i/patch[0];
			photoy=i%patch[0];
			ei=tex2D(batch2D,photoy,photox+bidx*patch[0]);
			pi=tex2D(nEx2D,photoy,(bid-pEx_s)*patch[0]+photox);//�����������ȥpEx
			shared[tid*3]+=ei*pi;
			shared[tid*3+1]+=ei*ei;
			shared[tid*3+2]+=pi*pi;
		}
		__syncthreads();   
		///////////////////////////////////////////////////////
		///*
		for(i=blockDim.x;i>1;)
		{
			if(tid<i/2)
			{
				i=(i+1)/2;
				shared[tid*3] += shared[(tid+i)*3];
				shared[tid*3+1] += shared[(tid+i)*3+1];
				shared[tid*3+2] += shared[(tid+i)*3+2];
			}
			else
				break;
			__syncthreads();
		}
		__syncthreads();
		//*/
		/*
		while(offset>0) {
		if(tid<offset) {
		if(tid+offset<threadNum)
		{
		shared[tid*3] += shared[(tid+offset)*3];
		shared[tid*3+1] += shared[(tid+offset)*3+1];
		shared[tid*3+2] += shared[(tid+offset)*3+2];
		}
		}
		threadNum=offset;
		if(offset==1)
		offset=0;
		else
		offset =(offset+1)/2;


		__syncthreads();
		}
		*/
		if(tid<2)
			shared[tid+1]=sqrt((float) shared[tid+1]);
		__syncthreads();
		if(tid == 0) {
			shared[1]=shared[1]*shared[2];
			dev_batchMatch[bidx*gridDim.y+bid] = (shared[0]/shared[1]+1)/2;  //д����        
		}
	}	
}



void NNConfBatchcu(void ** batch,float *nccBatAnscu,int count,double &f2datacost)
{  	//����ʱָ�����飬���ʱһ��ָ�룬ָ��һ���ڴ�
	//����Ҫ��ָ����������ָ���Mat����һ�������ڴ�batch2D��,�ø�forѭ��
	for(int i=0;i<count;i++)
		cudaMemcpy2D((char*)dev_batch+i*batch_pitch*patch_size, batch_pitch, batch[i], sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	//�������gpu����,���߳̿��ά��
	//���ǿ����������������趨һ����ά��dim3����x,y),x�е��߳̿鴦��batch�ĵ�x*patch_size�е�(x+1)*patch_size�е����ݣ�����x��Сͼ
	//���ں˺�����ÿ���������һ����������������ȵ����Դ�������һ���ڴ������洢���,�Ѿ���nccinit��������dev_batchMatch
	dim3 blocks(count,pEx_size+nEx_size);
	if((patch_size*patch_size)>maxThreads)
		ThreadNUM=maxThreads;
	else
		ThreadNUM=patch_size*patch_size;//�������Բ�����ֶ�����߳���
	dim3 threads(ThreadNUM);
	NNConfBatchker<<< blocks, threads,ThreadNUM*sizeof(float)*3>>>(dev_batchMatch,pEx_size);//һֱ���Ҵ���ԭ��������������ˣ����ƴ�������״�
	cudaThreadSynchronize();
	cudaMemcpy(nccBatAnscu,dev_batchMatch,(pEx_size+nEx_size)*count*sizeof(float),cudaMemcpyDeviceToHost);	
    //��ôӿ�ʼ��ʱ��ֹ֮ͣ���ʱ��
  	f2datacost = 0;
}
__global__ void testDataker(float *dev_out)
{
	//for(int i=0;i<15;i++)
	//	dev_out[i]=tex2D(testData2D,i,0);
	int tid=threadIdx.x;//blockDim.x=patch_size
	int bid=blockIdx.y;
	int bidx=blockIdx.x;
	int threadNum=blockDim.x;
	int photox;;//��photox��
	int photoy;;//��photo��
	photox=tid/patch[0];
	photoy=tid%patch[0];
	if(blockIdx.x==9)
		if(tid==8)
			for(int i=0;i<6;i++)
				dev_out[i]=tex2D(batch2D,photoy+i,photox+bidx*patch[0]);
}
void testDatacu(const float *in,float *pout,float *nout)
{
	//in�Ǵ�cpu������ָ�룬pout,nout������cpu������
	//����dev_testData�����Ұ�Ϊ����
	cudaMallocPitch((void**)(&dev_testData), &testData_pitch, sizeof(float) * patch_size, patch_size*100);//��������float����������Ҳֻ����float���͵ġ�
	cudaChannelFormatDesc testDataDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, testData2D, dev_testData, testDataDesc, patch_size, patch_size*100, testData_pitch);
	cudaMemcpy2D((float*)((char*)dev_testData), testData_pitch, in, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	//����dev_out�Դ�
	float *dev_out;
	cudaMalloc(&dev_out,15*sizeof(float));
	dim3 blocks(12,pEx_size+nEx_size);
	if((patch_size*patch_size)>512)
		ThreadNUM=512;
	else
		ThreadNUM=patch_size*patch_size;//�������Բ�����ֶ�����߳���
	dim3 threads(ThreadNUM);
	testDataker<<<blocks, threads,ThreadNUM*sizeof(float)*3>>>(dev_out);
	cudaMemcpy(pout,dev_out,15*sizeof(float),cudaMemcpyDeviceToHost);
	//cudaMemcpy2D((float*)pout, 60, dev_pEx, pEx_pitch, sizeof(float) * patch_size, patch_size, cudaMemcpyDeviceToHost);	
	cudaMemcpy(nout,dev_batchMatch,(pEx_size+nEx_size)*12*sizeof(float),cudaMemcpyDeviceToHost);
	//cudaMemcpy2D((float*)nout, 60, dev_batch, batch_pitch, sizeof(float) * patch_size, patch_size*4, cudaMemcpyDeviceToHost);		
}
