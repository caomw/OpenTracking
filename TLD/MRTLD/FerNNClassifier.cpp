/*
 * FerNNClassifier.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */

#include "FerNNClassifier.h"

using namespace cv;
using namespace std;
//������Щ����ͨ������ʼ����ʱ����parameters.yml�ļ����г�ʼ��  
void FerNNClassifier::read(const FileNode& file){
  ///Classifier Parameters
  valid = (float)file["valid"];
  ncc_thesame = (float)file["ncc_thesame"];
  nstructs = (int)file["num_trees"];//��ľ����һ�������鹹����ÿ����������ͼ���Ĳ�ͬ��ͼ��ʾ���ĸ���  
  structSize = (int)file["num_features"];//ÿ����������������Ҳ��ÿ�����Ľڵ����������ÿһ����������Ϊһ�����߽ڵ�  
  thr_fern = (float)file["thr_fern"];
  thr_nn = (float)file["thr_nn"];
  thr_nn_valid = (float)file["thr_nn_valid"];
}
/*
* In
*	-scales:		�����ܼ��ĳ߶Ȳ�ĳ߶�
*	-nstructs��	��10��
* Out
*	-features:   Ferns features (one std::vector for each scale) ????
*	-thrN:		 0.5*nstructs ???
* Init
*   -nCounter:	����ȡֵ�ķֲ����� 2^13��bin,�� FerNNClassifier::update
*   -pCounter:
*/
void FerNNClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  // 1. Initialize test locations for features
  //   ���������Ҫ����ԣ�x1f��y1f��x2f��y2f��ע�ⷶΧ[0,1)����
  //   ��ȷ����ÿһ������������Щ��Խ��ж��õ�����Щλ��һ��ȷ���Ͳ���ı䣬
  //   ��������Ҫ���ж�߶ȼ�⣬����ͬһ�����ڲ�ͬ�߶�scales��ʵ�ʶ�Ӧ������Ҫ����ʵ�ʵ�width��height�� 
  int totalFeatures = nstructs*structSize;//nstructs 10 structSize 13
  //��ά����  ����ȫ���߶ȣ�scales����ɨ�贰�ڣ�ÿ���߶Ȱ���totalFeatures������  
  features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));
  RNG& rng = theRNG();
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  //���ާ����������n��������������ÿ�����������ǻ���һ��pixel comparisons�����رȽϼ����ģ�  
  //pixel comparisons�Ĳ�������������һ����һ����patchȥ��ɢ�����ؿռ䣬�������п��ܵĴ�ֱ��ˮƽ��pixel comparisons  
  //Ȼ�����ǰ���Щpixel comparisons��������n����������ÿ���������õ���ȫ��ͬ��pixel comparisons���������ϣ���  
  //���������з�������������ͳһ�����Ϳ��Ը�������patch��   
  //�������ȥ���ÿһ���߶�ɨ�贰�ڵ�����   
  for (int i=0;i<totalFeatures;i++){
      x1f = (float)rng; //����[0,1)ֱ�ӵĸ�����
      y1f = (float)rng;
      x2f = (float)rng;
      y2f = (float)rng;
      for (int s=0;s<scales.size();s++){
          x1 = x1f * scales[s].width;
          y1 = y1f * scales[s].height;
          x2 = x2f * scales[s].width;
          y2 = y2f * scales[s].height;
          features[s][i] = Feature(x1, y1, x2, y2); //��s�ֳ߶ȵĵ�i������  ���������������ص�����  
      }
  }
  // 2. Thresholds�� ����������ֵ
  thrN = 0.5*nstructs;
  // 3. Initialize Posteriors�� ��ʼ���������   
  //�������ָÿһ���������Դ����ͼ��Ƭ�������ضԱȣ�ÿһ�����ضԱȵõ�0����1�����е�����13��comparison�Աȣ�  
  //����һ��13λ�Ķ����ƴ���x��Ȼ��������һ����¼�˺�����ʵ�����P(y|x)��yΪ0����1�������ࣩ��Ҳ���ǳ���x��  
  //�����ϣ���ͼ��ƬΪy�ĸ����Ƕ��ٶ�n�������������ĺ��������ƽ��������0.5���ж��京��Ŀ�� 
  for (int i = 0; i<nstructs; i++) {
	  //ÿһ��ÿ����ά��һ��������ʵķֲ�������ֲ���2^d����Ŀ��entries��������d�����رȽ�pixel comparisons  
	  //�ĸ�����������structSize����13��comparison�����Ի����2^13��8,192�����ܵ�code��ÿһ��code��Ӧһ���������  
	  //�������P(y|x)= #p/(#p+#n) ,#p��#n�ֱ������͸�ͼ��Ƭ����Ŀ��Ҳ���������pCounter��nCounter  
	  //��ʼ��ʱ��ÿ��������ʶ��ó�ʼ��Ϊ0������ʱ�������淽ʽ���£���֪����ǩ��������ѵ��������ͨ��n��������  
	  //���з��࣬���������������ô��Ӧ��#p��#n�ͻ���£�����P(y|x)Ҳ��Ӧ������
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  }
}
/*
* In
*	-image:		path
*	-scale_idx:	
*   -features   
* Out
*	-fern:	nstructs��������ɵ�������ÿ����������structSize��13��λ����ô�������أ�13�Ե������ֵ�ĶԱȽ����0/1��
*			�Աȵ㶼��¼��features���棬��FerNNClassifier::prepare�׶��������
*/
//�ú����õ������image���������Ľڵ㣬Ҳ�����������������13λ�Ķ����ƴ��룩  
void FerNNClassifier::getFeatures(const cv::Mat& image,const int& scale_idx, vector<int>& fern){
  int leaf;
  //ÿһ��ÿ����ά��һ��������ʵķֲ�������ֲ���2^d����Ŀ��entries��������d�����رȽ�pixel comparisons  
  //�ĸ�����������structSize����13��comparison�����Ի����2^13��8,192�����ܵ�code��ÿһ��code��Ӧһ���������  
  for (int t=0;t<nstructs;t++){//nstructs ��ʾ���ĸ��� 10
      leaf=0;//Ҷ��  �������սڵ�
      for (int f=0; f<structSize; f++){
		  //���εõ�ÿһλ
		  //���ص�patchͼ��Ƭ��(y1,x1)��(y2, x2)������رȽ�ֵ������0����1  
		  //Ȼ��leaf�ͼ�¼����13λ�Ķ����ƴ��룬��Ϊ����   
          leaf = (leaf << 1) + features[scale_idx][t*nstructs+f](image);//��������أ���һ������ڵڶ����㷵��1������Ϊ0
      }
      fern[t]=leaf;
  }
}
// ���ʺ�
float FerNNClassifier::measure_forest(vector<int> fern) {
  float votes = 0;
   // �������posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);  
  for (int i = 0; i < nstructs; i++) {
      votes += posteriors[i][fern[i]];//ÿ������ÿ������ֵ��Ӧ�ĺ�������ۼ�ֵ ��ͶƱ
  }
  return votes;
}
// ��������������ֱ��ͼ�ֲ���ע�⣺posteriorsֻ�����������ĸ���
//����������������ͬʱ���º������   
void FerNNClassifier::update(const vector<int>& fern, int C, int N) {
  int idx;
  for (int i = 0; i < nstructs; i++) {//10
      idx = fern[i];//13λ������
	  //ֱ����idx��Ϊһ��bin��sizeof(int)���ݳ�ѽ
	  // ���⣺ ������������ر����ô�����ֲ��ͻ��ر�ϡ�裬��ˣ����ܱ�֤ͳ�Ƶ���Ч�ԣ���
	  // ���⣺ ���ϵ�ֱ��ͼͳ�ƣ���ƽ����Ҳ̫���˰ɣ�����
	  //C=1����������C=0��������
      (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
      if (pCounter[i][idx]==0) {//��Ȼ�������ʣ��������������ĿΪ0���������ĸ�����ȻҲΪ0
          posteriors[i][idx] = 0;
      } else {
          posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      }
  }
}
/*
* Fun
*	bootstrapѵ�����ɭ�ַ�����,���������ֲ��͸��ʣ����׷ִ����������֧���ʣ�measure_forest��<=thrP������)�����׷ִ�ĸ����� ֧����>=���
*	-nCounter ->posteriors
*   -pCounter ->posteriors
* In
*	- ferns��	���ҹ�˳�����������������
*	- resample��	bootstrap������ʵ���ϳ�������û��bootstrap����
*/
//���⣺��������ȷ������������Ͳ����·ֲ�����ô������ԭʼ�ֲ�ѽ���ѵ�posteriors�Ѿ���ʼ��������
//��ʵ�������������������ʼû���������Ļ�����ôposteriorsӦ�ö���0�����������������
//������measure_forest(ferns[i].first)>thrP�Ͳ�����
// ��ô��ѵ��ʱ������˳��Ͳ���ǰ�涼�������������涼�Ǹ���������֮��Ȼ���������˳��
//ѵ�����Ϸ�������n���������������ϣ�
//��Ҫ�����ǣ���ÿһ������ferns_data[i] �������������������ǩ������measure_forest������
//�ҳ�������box������������������ֵ����Ӧ�ĺ�������ۼ�ֵ�����ۼ�ֵ���С����������ֵ��0.6* nstructs��
//0.6�������ֵ��Fern����������ֵ������ʼ��ʱ��myParam.yml�ж�ȡ��������ò��Լ��������޸ģ�
//�ҵ����ţ���Ҳ�������������������ȴ������ɸ������ˣ������˷���������ԾͰѸ�������ӵ���������ʹpNum=pNum+1��
//ͬʱ��update�������º�����ʡ����ڸ�������ͬ����������ָ�����������󣬾���ӵ���������ʹnNum=nNum+1��
//update������������������һ�������Ǹ�box��Ӧ��10����fern[]���ڶ�������Ҫ���и��µ����������⻹�Ǹ������⣬1��ʾ���������������Ŀ��
//0��ʾ���¸����������Ŀ��������������ʾҪ���µ���Ŀ�����������������еĵ��ø�ֵ����ȡ1��Ҳ��ÿ�ζ��Ƕ���������Ŀ����1(Ϊ����һ�ߵĲ�����Ӧ�ؼ�С1��)��
//��Ҳ�������ᵽ�ķִ�������ŵ���Ӧ�Ŀ���ͬ������˼����Ϊÿ��ֻ���ж�һ�������Ƿ��зִ����Ը��µ���ĿҲֻ����1��
//���ڸ�����Ŀ��ͬʱ��Ҳ�����˺������ֵ������post=pNum/nNum��ʽ��������
void FerNNClassifier::trainF(const vector<std::pair<vector<int>,int> >& ferns,int resample){
  thrP = thr_fern*nstructs; //0.6*10                           // int step = numX / 10;
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index
                                                            //       double *x = X+nTREES*I; //tree index
          if(ferns[i].second==1){ //������  //Ϊ1��ʾ������   //       if (Y[I] == 1) {
			  //measure_forest������������������������ֵ��Ӧ�ĺ�������ۼ�ֵ  
			  //���ۼ�ֵ���С����������ֵ��Ҳ���������������������ȴ������ɸ�������  
			  //���ַ���������ԾͰѸ�������ӵ��������⣬ͬʱ���º������  
			  if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
                update(ferns[i].first,1,1);                 //             update(x,1,1);
			  ////��������������ͬʱ���º������
          }else{//������                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
                update(ferns[i].first,0,1);                 //             update(x,0,1);
          }
      }
	
  //}
}
/*
* Fun: ѵ�����ʱ�ģ�NN������
* In:
*	- nn_examples:[+][-][-].... ֻ�е�һ����������
* Out:
*	nEx <-- nn_examples�ĸ����������� Relative similarity  >0.5
*	pEx <-- nn_examples�������������� Relative similarity  <thr_nn(0.65)
* ע�⣺ �ǲ��ϵ����� nEx �� pEx
*/
//���⣺���������ĳ���˳���Ӱ�쵽���յķ���������ϲ�������������
//ѵ������ڷ���������ÿһ������nn_data�������ǩ����������ͨ��NNConf(nn_examples[i], isin, conf, dummy)��������ͼ��Ƭ������ģ��֮���������ƶ�conf��
//���������ƶ�С��0.65 ������Ϊ�䲻����ǰ��Ŀ�꣬Ҳ���Ƿ�������ˣ���ʱ��Ͱ����ӵ��������⡣
//Ȼ���ͨ��pEx.push_back(nn_examples[i]);����������ӵ�pEx���������У�
//ͬ����������ָ�����������󣬾���ӵ���������
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples){
  float conf,dummy;
  vector<int> y(nn_examples.size(),0);
  y[0]=1;//ֻ�е�һ������������������ԭʼ��Ŀ�����򣬶���best_box��//����˵������trainNN������������nn_data��������ֻ��һ��pEx����nn_data[0]  
  vector<int> isin;
  for (int i=0;i<nn_examples.size();i++){//  For each example
      NNConf(nn_examples[i],isin,conf,dummy);//  Measure Relative similarity//��������ͼ��Ƭ������ģ��֮���������ƶ�conf  
	  //thr_nn: 0.65 ��ֵ   
	  //��ǩ�������������������ƶ�С��0.65 ������Ϊ�䲻����ǰ��Ŀ�꣬Ҳ���Ƿ�������ˣ���ʱ��Ͱ����ӵ���������  
	  if (y[i]==1 && conf<=thr_nn){//    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65
          if (isin[1]<0){ //ע�⣺���pExΪ�գ�NNConf ֱ�ӷ��� thr_nn=0��isin��Ϊ-1��                                         //      if isnan(isin(2))
              pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
              continue;                                            //        continue;
          }                                                        //      end
          //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
          pEx.push_back(nn_examples[i]);//֮ǰ������������׷��
      }                                                            //    end
      if(y[i]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
        nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];

  }                                                            
  acum++;
  printf("%d. Trained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                               

/*Inputs:
* -NN Patch
* Outputs:
* -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
* isin[3]��isin[0]:is pos ? isin[1]����+�ģ���ô��¼��ӽ�����������Id��isin[2]��is neg?
//isin�д������int��ֵ����ʼ��ȫΪ-1����һ�����ȡֵΪ1�����ʾNNConf()�ڼ�������ͼ��Ƭpatch������ģ��pEx�е�boxʱ��������ģ������һ���������ƶȳ�����ֵncc_thesame (�̶�ֵ0.95����myParam.yml�ж�ȡ)��
//��ʱ������patchҲ�ŵ�����ģ�͵�pEx�У����Ե�һ��ȡֵΪ1�ͱ�ʾ�Ѿ��ѵ�ǰ����ͼ��Ƭpatch�ŵ�pEx�С�
//�ڶ�����ȡֵ�����ڵ�1����ȡֵ�������һ��ȡֵΪ-1����ô�ڶ�����ȡֵ����-1�������һ����ȡֵ��1��
//��ô�ڶ�����ȡֵ�����ڱ�������ģ��ʱ�ҵ��ĵ�һ��������ͼ��Ƭpatch���ƶȳ���ncc_the same��box��������
//�������������һ���ӽ�����ͬ�ĵط�ֻ���ڵ�һ���Ƕ�Ӧ����ģ�͵��������������ݼ�pEx���������Ƕ�Ӧ����ģ�͵ĸ������������ݼ�nEx��
*/
//���������ͼ��Ƭpatch���ȱ�������ģ���е��������������ݼ�pEx�е�box(��һ����ʵ����best_box����������߸���)��
//����matchTemplate()����ƥ���ncc������ncc�õ����ƶ�nccP�����ҳ�ncc�е������maxP��
//ͬ���ķ�ʽҲ��������ģ���еĸ������������ݼ�nEx�е�box���ҳ�maxN
void FerNNClassifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf){
  isin=vector<int>(3,-1);
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
      return;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return;
  }
  Mat ncc(1,1,CV_32F);
  float nccP,csmaxP,maxP=0;
  bool anyP=false;
  int maxPidx,validatedPart = ceil(pEx.size()*valid);//��������ǰ 50%�����ڼ���Conservative similarit��5.2 5��
  float nccN, maxN=0;
  bool anyN=false;
  //�Ƚ�ͼ��Ƭp������ģ��M�ľ��루���ƶȣ���������������������ƶȣ�Ҳ���ǽ������ͼ��Ƭ��  
  //����ģ�������е�ͼ��Ƭ����ƥ�䣬�ҳ������Ƶ��Ǹ�ͼ��Ƭ��Ҳ�������ƶȵ����ֵ  
  for (int i=0;i<pEx.size();i++){
      matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);// measure NCC to positive examples
	  //���ϵ����ȡֵ��Χ��[-1,1]������1���[0,2]���ٽ���Χ��СΪ[0,1]
      nccP=(((float*)ncc.data)[0]+1)*0.5;//����ƥ�����ƶ� 
      if (nccP>ncc_thesame)//0.95
        anyP=true;
      if(nccP > maxP){
          maxP=nccP;//Relative similarity //��¼�������ƶ��Լ���Ӧ��ͼ��Ƭindex����ֵ 
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;//Conservative similari
      }
  }
  //���㸺������������ƶ�   
  for (int i=0;i<nEx.size();i++){
      matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);//measure NCC to negative examples
      nccN=(((float*)ncc.data)[0]+1)*0.5;
      if (nccN>ncc_thesame)
        anyN=true;
      if(nccN > maxN)
        maxN=nccN;
  }
  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  //Measure Relative Similarity
  //������ƶ� = ��������������ƶ� / ����������������ƶ� + ��������������ƶȣ�  
  //�����������1-maxP��1-maxN���������Ϊ�������������Ƶ���С�̶ȡ������������Ƶ���С�̶ȡ�
  //Ȼ��ͼ���������ƶ�rsconf=(1-maxN)/[(1-maxP)+(1-maxN)]��Ҳ���������������Ƶ���С����£���������������ռ�����Ͷ���Ϊ������ƶȣ�
  //�����������������ռ����Խ����ô��patchpatch�븺���������ƵĿ�����Խ�󣬴Ӷ�������ƶ�Խ��(�����е��ֿڣ������ǴӶ�ż����ĽǶ�����)
  float dN=1-maxN;
  float dP=1-maxP;
  // ���ڱ������ƶ�csconf�������õ��ģ���pEx��ǰ�벿�������У�����ж�maxP���и��£���ô�Ѵ�ʱ��maxP�ŵ�csmaxP�У�
  //Ҳ��csmaxP��¼�������������ݼ�pEx��ǰ�벿��������������ͼ��Ƭ��������ƶȣ�
  //Ȼ��csconf=(1-maxN)/[(1-maxN)+(1-csmaxP)]������csmaxP������maxP������csconf������rsconf��
  //Ҳ������Ϊ��ǰ����ͼ��Ƭpatch���������������ݼ��������Ƶ������ϣ���ͬ��patch��
  //rsconf�Ķ�����Ϊ�����ء����ڵ�һ������ѵ����ʱ���Ƿ���û�ж�����壬
  //������init()�е�һ�ν���trainNN()ʱ���Ὣrsconf���õ���ֵ����
  rsconf = (float)dN/(dN+dP);//��ԭ�ġ�5.2���г��룬����Ҳ�ǿ�������
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
}
//  evaluateTh
//	���룺
// 	���������ާ���Լ���nXT
// 	������ģ����Լ���nExT
// 	�����
// 	���º�����ާ��ֵ��thr_fern
// 	���º��ģ��NCC��ֵ��thr_nn
// 	���º����Ч���ж�(P/Nѧϰ)��ֵ��thr_nn_valid
// 	������
// 	�����и��������ާ���Լ������������ާ���Ŷȣ������ֵ����Ԥ����ֵ
// 	thr_fern�����ø����ֵ�滻���£������и�����ģ����Լ���������NCCϵ����
// 	�����ֵ����Ԥ����ֵthr_nn�����ø����ֵ�滻���£��������µõ���thr_nn
// 	����Ԥ��thr_nn_valid�����滻����thr_nn_valid
void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT,const vector<cv::Mat>& nExT){
float fconf;
// ����thr_fern����nXT�е�ÿ����������measure_forest()�����ҳ�10���������box��01�룬��01���ҵ���Ӧ�ĺ�����ʣ���10����������ۼӺ���ƽ������ƽ��ֵ��thr_fern(��ʼֵ0.6����myParam.yml�л�ȡ)�Ƚϣ�
//�������thr_fern��thr_fern����Ϊ���ƽ��ֵ��Ҳ������nXT�����е�box��10�����������ͶƱ�õ�һ��������ʣ�
//�����к������ȡƽ���󣬱Ƚ����е�box��ȡ������ʾ�ֵ�����Ǹ�box����Ӧ��ƽ��ֵ�ŵ�thr_fern��
  for (int i=0;i<nXT.size();i++){
	  //���л����������ĺ�����ʵ�ƽ��ֵ�������thr_fern������Ϊ����ǰ��Ŀ��  
	  //measure_forest���ص������к�����ʵ��ۼӺͣ�nstructs Ϊ���ĸ�����Ҳ���ǻ�������������Ŀ 
    fconf = (float) measure_forest(nXT[i].first)/nstructs;//ƽ��
    if (fconf>thr_fern)//0.6thrP����ΪPositive thershold
      thr_fern=fconf;//������ʼֵ//ȡ���ƽ��ֵ��Ϊ �ü��Ϸ������� �µ���ֵ�������ѵ������ 
}
  //����thr_nn�����ڸ�������������nExT���Լ��е�ÿ����������NNConf()����������������ģ��pEx��nEx�����ݵ����ƶ����õ�������ƶ�conf
  //(���һ��ѵ��NN������һ��������õ��ı������ƶ�Ҳ��������)�����conf������ֵthr_nn(��ʼֵ0.65����myParam.yml�л�ȡ)��
  //�����thr_nnΪconf
  vector <int> isin;
  float conf,dummy;
  for (int i=0;i<nExT.size();i++){
      NNConf(nExT[i],isin,conf,dummy);
      if (conf>thr_nn)
        thr_nn=conf; //ȡ������������ƶ���Ϊ ������ڷ������� �µ���ֵ�������ѵ������  
  }
  //����thr_nn_valid��������º��thr_nn����thr_nn_valid(��ʼֵ0.7����myParam.yml�л�ȡ)����ô����thr_nn_validֵΪthr_nn
  if (thr_nn>thr_nn_valid)//���ǵĳ�ʼֵ���ܲ����ϸ����Ǽ���Ƿ�Ҫ�������ڷ���������ֵ
    thr_nn_valid = thr_nn;
}
//���������⣨����ģ�ͣ�������������������ʾ�ڴ�����   
void FerNNClassifier::show(){
  Mat examples((int)pEx.size()*pEx[0].rows,pEx[0].cols,CV_8U);
  double minval;
  Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
  for (int i=0;i<pEx.size();i++){
    minMaxLoc(pEx[i],&minval);//Ѱ��pEx[i]����Сֵ 
    pEx[i].copyTo(ex);
    ex = ex-minval; //������������С����������Ϊ0���������ذ�������  
	//Mat Mat::rowRange(int startrow, int endrow) const Ϊָ������span����һ���µľ���ͷ��  
	//Mat Mat::rowRange(const Range& r) const   //Range �ṹ��������ʼ����ֹ������ֵ��  
    Mat tmp = examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
    ex.convertTo(tmp,CV_8U);
  }
  imshow("Examples",examples);
}
