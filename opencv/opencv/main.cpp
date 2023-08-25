#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <fstream>
using namespace cv;
using namespace std;

void drawHist(Mat& hist, int type,string name)
{
	// ����ֱ��ͼ���ƴ���
	int histWidth = 512;
	int histHeight = 400;
	cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));
	// ��ֱ��ͼ���й�һ��
	cv::normalize(hist, hist, 1, 0, type, -1, cv::Mat());
	// ����ֱ��ͼ
	int binWidth = 2;
	for (int i = 1; i <= hist.rows; i++)
	{
		cv::line(histImage, cv::Point(binWidth * (i-1), histHeight-1), cv::Point(binWidth * i-1, histHeight - cvRound(20*histHeight*hist.at<float>(i-1))-1), cv::Scalar(255, 255, 255));
	}
	// ��ʾֱ��ͼ
	cv::imshow(name, histImage);
}

void addsalt(Mat& image, int n) {
	for (int k = 0; k < n; k++)
	{
		int j = rand() % image.rows;
		int i = rand() % image.cols;
		int color = rand() % 2;
		if (color == 0)  //waite
		{
			if (image.type() == CV_8U)
			{
				image.at<uchar>(j, i) = 255;
			}
			else if(image.type() == CV_8UC3)
			{
				image.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
			}
		}
		else
		{
			if (image.type() == CV_8U)
			{
				image.at<uchar>(j, i) = 0;
			}
			else if (image.type() == CV_8UC3)
			{
				image.at<Vec3b>(j, i) = Vec3b(0, 0, 0);
			}
		}
	}
}

void ord_feature(Mat& img,vector<KeyPoint>& kps,Mat& des) {
	Ptr<ORB>orb = ORB::create();
	orb->detect(img, kps);
	orb->compute(img, kps, des);
}

void match_min(vector<DMatch>& dm, vector<DMatch>& gdm) {
	double maxdis = 0, mindis = 10000;
	for (int i = 0; i < dm.size(); i++)
	{
		double dis = dm[i].distance;
		if (dis > maxdis) maxdis = dis;
		if (dis < mindis) mindis = dis;
	}
	cout << maxdis << mindis << endl;
	for (int i = 0; i < dm.size(); i++)
	{
		double dis = dm[i].distance;
		if (dis <= max(2 * mindis, 20.0)) {
			gdm.push_back(dm[i]);
		}
	}
}

void ransac(vector<DMatch> gdm, vector<KeyPoint> kps1, vector<KeyPoint> kps2, vector<DMatch>& gsc) {
	vector<Point2f> srcp(gdm.size()), dstp(gdm.size());
	for (int i = 0; i < gdm.size(); i++)
	{
		srcp[i] = kps1[gdm[i].queryIdx].pt;
		dstp[i] = kps2[gdm[i].trainIdx].pt;
		vector<int> inliersMask(srcp.size());
		findHomography(srcp, dstp,RANSAC,5,inliersMask);
		for (int i = 0; i < inliersMask.size(); i++)
		{
			if (inliersMask[i])
				gsc.push_back(gdm[i]);
		}
	}
}

int main(int argc, char** agrv) {
	//Mat imgg = imread("chess0.jpg");
	Mat img = imread("C:/Users/wfs/Pictures/�����з��Ӯ�����������.png");
	Mat socer = imread("C:/Users/wfs/Pictures/socear.png");
	Mat socear = imread("C:\\Users\\wfs\\Desktop\\opencv\\opencv\\opencv\\socear.png",IMREAD_GRAYSCALE);
	//Mat book = imread("C:/Users/wfs/Pictures/����.png");
	//Mat mask = imread("C:/Users/wfs/Desktop/opencv/opencv/opencv/OR_MASK.jpg");
	Mat lena = imread("lena.png");
	Mat lenahead = imread("lenahead.png");
	Mat lenagray = imread("lenagray.jpg");
	Mat lenagraysalt = imread("lenagraysalt.png");
	Mat lenagraynoise = imread("lenagraynoise.png");
	cvtColor(lenahead, lenahead, COLOR_BGR2GRAY);
	cvtColor(lenagray, lenagray, COLOR_BGR2GRAY);
	cvtColor(img, img, COLOR_BGR2GRAY);
	equalizeHist(socear, socear); //ֱ��ͼ���⻯ ��ֻ�ܵ�ͨ��

	//Mat socear = socer(Range(400,1200), Range(242,1035));
	//imwrite("socear.png", socear);

	//img.convertTo(img,CV_8U,(0,255));
	//convertScaleAbs(img, img, 1.0 / 256.0, 0.0);

	//��������

	//CV_8U: 8 λ�޷���������ռ�� 1 ���ֽڡ�
	//CV_8S: 8 λ�з���������ռ�� 1 ���ֽڡ�
	//CV_16U : 16 λ�޷���������ռ�� 2 ���ֽڡ�
	//CV_16S : 16 λ�з���������ռ�� 2 ���ֽڡ�
	//CV_32S : 32 λ�з���������ռ�� 4 ���ֽڡ�
	//CV_32F : 32 λ��������ռ�� 4 ���ֽڡ�
	//CV_64F : 64 λ��������ռ�� 8 ���ֽڡ�

	//���ݶ�ȡ

	//Mat A = (Mat_<int>(3, 3) << 1, 2, 5, 4, 5, 6, 7, 8, 9);
	//Mat B1 = Mat(5, 5, CV_8UC1,Scalar(4,5,6));
	//Mat B2 = Mat(5, 5, CV_8UC2, Scalar(4, 5, 6));
	//Mat B3 = Mat(5, 5, CV_8UC3, Scalar(4, 5, 6));
	//cout << A.at<int>(0, 0) << endl;
	//Vec2b vc = B2.at<Vec2b>(0, 0);
	//cout << B2 << endl;
	//cout << vc << endl;
	//cout << (int)vc.val[0] << endl;
	//cout << B3 << endl;
	//���ص���ÿһ�У��е��ֽ���
	//cout <<  (int)(*(B3.data + B3.step[0] * 1 + B3.step[1] * 2 + 2)) << endl;
	//cout << (int)(*(B3.data + B3.step[0] * 1 + B3.step[1] * 2 + 1)) << endl;
	//cout << (int)(*(B3.data + B3.step[0] * 1 + B3.step[1] * 2 + 0)) << endl;

	//Mat img;
	//cout << A << endl;
	//picture read and show
	//img = imread("c:/users/wfs/pictures/screenshot 2022-08-07 225529.png");
	//if (img.empty()) {
	//	cout << "no img" << endl;
	//	return -1;
	//}
	//imshow("sdz", A);
	// im.write();
	//waitKey(0);

	//mat����
	//*���  ��dot�ڻ� ��mul��Ӧλ���

	//����ͷ
	//Mat img;
	//VideoCapture vc(0);
	////vc >> img;
	////bool iscolor = (img.type() == CV_8UC3);
	////VideoWriter vw;
	////int codeff = VideoWriter::fourcc('m', 'p', '4', 'v');
	////vw.open("live.mp4", codeff, 30, img.size(), iscolor);
	//while (1)
	//{
	//	vc.read(img);
	//	imshow("live", img);
	//	int fps = vc.get(CAP_PROP_FPS);
	//	cout << fps << endl;
	//	int width = vc.get(CAP_PROP_FRAME_WIDTH);
	//	int hight = vc.get(CAP_PROP_FRAME_HEIGHT);
	//	putText(img, to_string(fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 3);
	//	waitKey(50);
	//}

	////ͼ����������ת��  RGB,HSV��ɫ�ȣ����Ͷȣ����� �� GRAY=R*0.3+G*0.59+B*0.11
	
	////8U :0-255,32F:0-1,d64f:0-1      >1Ϊ��ɫ ��<1Ϊ��ɫ
	//Mat img32,HSV,HSV_32,gray,gray_32;
	//img.convertTo(img32, CV_32F, 1 / 255.0, 0);
	//cvtColor(img, HSV, COLOR_BGR2HSV);
	//cvtColor(img32, HSV_32, COLOR_BGR2HSV);
	//cvtColor(img, gray, COLOR_BGR2GRAY);
	//cvtColor(img32, gray_32, COLOR_BGR2GRAY);
	////split
	//Mat imgs[3],img1,img2,img3;
	//split(img, imgs);
	//img1 = imgs[0];
	//img2 = imgs[1];
	//img3 = imgs[2];
	////merge
	//Mat zero = Mat::zeros(img.size(), CV_8UC1);
	//Mat imgsb[3];
	//imgsb[0] = img1;
	//imgsb[1] = zero;
	//imgsb[2] = zero;
	//Mat blue;
	//merge(imgsb, 3,blue);
	//vector<Mat> imgsv;
	//imgsv.push_back(zero);
	//imgsv.push_back(img2);
	//imgsv.push_back(zero);
	//Mat green;
	//merge(imgsv, green);

	////ͼ�����رȽ� ��Ҫ��С��ͨ��������������һ��
	
	////min max��src1��src2��out��
	//Mat Min, Max;
	//min(socer, book, Min);
	////minMaxLoc  ���������С����ֵ�������꣬�������������

	////�� �� �� ���  0-255��Ϊ�����ƣ���ÿλ��������   0-11111111
	
	//Mat AND, OR, NOT, XOR;
	//bitwise_not(img, NOT, mask);
	//bitwise_and(img, NOT, AND, mask);
	//bitwise_or(img, NOT, OR, mask);
	//bitwise_xor(img, NOT, XOR, mask);
	////for (int i = 0; i < img.rows / 2; i++)
	////{
	////	for (int j = 0; j < 20; j++)
	////	{
	////		OR.at<Vec3b>(i, i+j) = Vec3b(0, 0, 0);
	////	}
	////}
	////for (int i = 0; i < img.rows / 2; i++)
	////{
	////	for (int j = img.cols; j >= img.cols - 20; j--) {
	////		if (j - i >= 0 && j - i < img.cols) {
	////			OR.at<cv::Vec3b>(i, j - i) = cv::Vec3b(0, 0, 0);
	////		}
	////	}
	////}
	////Mat ORG;
	////cvtColor(OR, ORG, COLOR_BGR2GRAY);
	////imwrite("OR_MASK.png", ORG);

	////��ֵ��ֵ�� 5������
	
	//Mat gimg;
	//threshold(img, gimg, 127, 255, THRESH_BINARY);
	//cvtColor(img, img, COLOR_BGR2GRAY);
	//adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 51, 0);

//	//LUT���ұ�

//uchar Lutfirst[256];
//for (int i = 0; i < 256; i++)
//{
//	if (i < 100)
//		Lutfirst[i] = 0;
//	if (i >= 100 && i <= 200)
//		Lutfirst[i] = 100;
//	if (i > 200)
//		Lutfirst[i] = 255;
//}
//Mat lutone(1, 256, CV_8UC1, Lutfirst);
//LUT(img, lutone, img);

////ͼ�����ţ���ת��ƴ��

//resize(img, img, Size(512, 512), 0, 0, INTER_AREA); //��size�Ͳ����������
//resize(img, img, Size(1682, 1392), 0, 0, INTER_NEAREST);
//resize(img, img, Size(1682, 1392), 0, 0, INTER_LINEAR);
//resize(img, img, Size(1682, 1392), 0, 0, INTER_CUBIC);
//flip(img, img, 0);
//hconcat(img, img, img);
//vconcat(img, img, img);

////ͼƬ��ת������任

//Mat rotation_matrix1 = getRotationMatrix2D(Point2f(800.0, 700.0), 60, 1);
//warpAffine(img, img, rotation_matrix1, img.size());
//Point2f src[3];
//Point2f dst[3];
//src[0] = Point2f(0, 0);
//src[1] = Point2f(0, img.cols);
//src[2] = Point2f(img.rows, 0);
//dst[0] = Point2f((img.rows)*0.1, (img.cols)*0.2);
//dst[1] = Point2f((img.rows)*0.3, (img.cols)*0.8);
//dst[2] = Point2f((img.rows)*0.9, (img.cols)*0.1);
//Mat rotation_matrix2 = getAffineTransform(src,dst);
//warpAffine(img, img, rotation_matrix2, img.size());

//͸�ӱ任���ĵ�任��
//Mat persp_matrix = getPerspectiveTransform(Point2f src, Point2f dst);
//warpPerspective(img, img, persp_matrix, img.size());

//��ͼ
//circle line rectangle fillPolv putText

////ROI��ȡ

//Mat imgr = img(Range(100, 1000), Range(200, 1500));
//Rect rec(100,100,500,500);
//Mat imgrec = img(rec);
//Mat copy;
//img.copyTo(copy,mask);

////��˹������ �²�������С�ߴ磩�����Ų����ԣ�����ֵ��

////pyrDown(img, img);
//vector<Mat> Guass;
//Guass.push_back(img);
//for (int i = 0; i < 3; i++)
//{
//	Mat guass;
//	pyrDown(Guass[i], guass);
//	Guass.push_back(guass);
//}
//for (int i = 0; i < 3; i++)
//{
//	imshow(to_string(i), Guass[i]);
//}
//waitKey(0);

////������˹������ ���²������

//vector<Mat> Guass;
//Guass.push_back(socer);
//for (int i = 0; i < 3; i++)
//{
//	Mat guass;
//	pyrDown(Guass[i], guass);
//	Guass.push_back(guass);
//}
//vector<Mat> Lap;
//for (int i = Guass.size()-1; i > 0 ; i--)
//{
//	Mat lap, upsample;
//	if (i==Guass.size()-1)
//	{
//		Mat down;
//		pyrDown(Guass[i], down);
//		pyrUp(down, upsample);
//		cout<<Guass[i].size()<<endl;
//		cout << upsample.size() << endl;
//		lap = Guass[i] - upsample;
//		Lap.push_back(lap);
//	}
//	pyrUp(Guass[i], upsample);
//	lap = Guass[i-1] - upsample;
//	Lap.push_back(lap);
//}
//for (int i = 0; i < Lap.size(); i++)
//{
//	imshow(to_string(i), Lap[i]);
//}
//waitKey(0);

////ͳ��ֱ��ͼ

//// ����ֱ��ͼ����
//int histSize = 256;    // ֱ��ͼ��bin��Ŀ
//float range[] = { 0, 256 };    // ����ֵ��Χ
//const float* histRange = { range };
//// ����ֱ��ͼ
//cv::Mat hist;
//cv::calcHist(&socear, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
//// ����ֱ��ͼ���ƴ���
//int histWidth = 512;
//int histHeight = 400;
//cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));
//// ��ֱ��ͼ���й�һ��
//cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
//// ����ֱ��ͼ
//int binWidth = cvRound((double)histWidth / histSize);
//for (int i = 0; i < histSize; i++)
//{
//	cv::line(histImage, cv::Point(binWidth * i, histHeight), cv::Point(binWidth * i, histHeight - cvRound(hist.at<float>(i))), cv::Scalar(255, 255, 255));
//}
//// ��ʾֱ��ͼ
//imshow("scoear", socear);
//cv::imshow("Histogram", histImage);
//cv::waitKey(0);

////ֱ��ͼƥ��     ͨ��ԭ��Ŀ��ֱ��ͼ֮�������֮����ۻ�����֮�����Сֵ��ȷ������ӳ���ϵ

////������ֵ������ÿһ����Сֵȷ��Lutӳ���ϵ��Lutƥ��ֱ��ͼ��
//int histSize = 256;    // ֱ��ͼ��bin��Ŀ
//float range[] = { 0, 256 };    // ����ֵ��Χ
//const float* histRange = { range };
//// ����ֱ��ͼ
//cv::Mat hist1,hist2;
//cv::calcHist(&img, 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange);
//cv::calcHist(&socear, 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange);
//drawHist(hist1, NORM_L1, "img");
//drawHist(hist2, NORM_L1, "soc");
//float hist1_cdf[256] = { hist1.at<float>(0) };
//float hist2_cdf[256] = { hist2.at<float>(0) };
//for (int i = 1; i < 256; i++)
//{
//	hist1_cdf[i] = hist1_cdf[i - 1] + hist1.at<float>(i);
//	hist2_cdf[i] = hist2_cdf[i - 1] + hist2.at<float>(i);
//}
//float diffcdfmatrix[256][256];
//for (int i = 0; i < 256; i++)
//{
//	for (int j = 0; j < 256; j++) {
//		diffcdfmatrix[i][j] = fabs(hist1_cdf[i] - hist2_cdf[j]);
//	}
//}
//Mat lut(1, 256, CV_8U);
//for (int i = 0; i < 256; i++)
//{
//	float min = diffcdfmatrix[i][0];
//	int index = 0;
//	for (int j = 1; j < 256; j++)
//	{
//		if (min> diffcdfmatrix[i][j])
//		{
//			min = diffcdfmatrix[i][j];
//			index = j;
//		}
//	}
//	lut.at<uchar>(i) = (uchar)index;
//}
//Mat res;
//LUT(socear, lut, res);
//imshow("socear", res);
//waitKey(0);

//ͼ��ģ��ƥ��

//Mat result;
//matchTemplate(lena,lenahead,result, TM_CCOEFF_NORMED);
//double maxVal, minVal;
//Point minLoc, maxLoc;
////Ѱ��ƥ�����е����ֵ����Сֵ�Լ�����λ��
//minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
////�������ƥ������
//rectangle(lena, cv::Rect(maxLoc.x, maxLoc.y, lenahead.cols, lenahead.rows), Scalar(0, 0, 255), 2);
//imshow("lena", lena);
//imshow("res", result);
//waitKey(0);

//ͼ����

//Mat kernel = (Mat_<float>(3, 3) << 1, 2, 1, 2, 0, 2, 1, 2, 1);
//Mat kernel_norm = kernel / 12;
//Mat filter;
//filter2D(lena, filter, -1,kernel_norm);
//imshow("fil",filter);

//ͼ������

//addsalt(lena, 10000);
//addsalt(lenagray, 10000);
//imshow("lena", lena);
//imshow("lenagray", lenagray);
//imwrite("lenasalt.png", lena);
//imwrite("lenagray.png",lenagray);
//Mat lena_noise = Mat::zeros(lena.rows, lena.cols, lena.type());
//Mat equalLena_noise = Mat::zeros(lena.rows, lena.cols, lenagray.type());
//imshow("lenaԭͼ", lena);
//imshow("equalLenaԭͼ", lenagray);
//RNG rng; //����һ��RNG��
//rng.fill(lena_noise, RNG::NORMAL, 10, 20); //������ͨ���ĸ�˹�ֲ������
//rng.fill(equalLena_noise, RNG::NORMAL, 15, 30); //������ͨ���ĸ�˹�ֲ������
//imshow("��ͨ����˹����", lena_noise);
//imshow("��ͨ����˹����", equalLena_noise);
//lena = lena + lena_noise; //�ڲ�ɫͼ������Ӹ�˹����
//lenagray = lenagray + equalLena_noise; //�ڻҶ�ͼ������Ӹ�˹����
////��ʾ��Ӹ�˹�������ͼ��
//imshow("lena�������", lena);
//imshow("equalLena�������", lenagray);
//imwrite("lenanoise.png", lena);
//imwrite("lenagraynoise.png", lenagray);


//�����˲�    ��˹�����ø�˹�˲�����

////��ֵ�˲�
//Mat res,res1;
//blur(lenagraysalt, res, Size(3, 3));
//blur(lenagraysalt, res1, Size(9, 9));
//imshow("res", res);
//imshow("res1", res1);
////�����˲�
//Mat equalLena_32F;
//lenagraysalt.convertTo(equalLena_32F, CV_32F, 1.0 / 255);
//Mat resultNorm, result, dataSqrNorm, dataSqr, equalLena_32FSqr;
////�����˲�boxFilter()��sqrBoxFilter()
//boxFilter(lenagraysalt, resultNorm, -1, Size(3, 3), Point(-1, -1), true);  //���й�һ��
//boxFilter(lenagraysalt, result, -1, Size(3, 3), Point(-1, -1), false);  //�����й�һ��
//sqrBoxFilter(equalLena_32F, equalLena_32FSqr, -1, Size(3, 3), Point(-1, -1),
//	true, BORDER_CONSTANT);
////��ʾ������
//imshow("resultNorm", resultNorm);
//imshow("result", result);
//imshow("equalLena_32FSqr", equalLena_32FSqr);
////��˹�˲�
//Mat result_5gauss, result_9gauss;  //��ź��и�˹�����˲�������������ִ����˲����ߴ�
//Mat result_5salt, result_9salt;  ////��ź��н��������˲�������������ִ����˲����ߴ�
////���þ�ֵ�˲�����blur()�����˲�
//GaussianBlur(lenagraynoise, result_5gauss, Size(5, 5), 10, 20);
//GaussianBlur(lenagraynoise, result_9gauss, Size(9, 9), 10, 20);
//GaussianBlur(lenagraysalt, result_5salt, Size(5, 5), 10, 20);
//GaussianBlur(lenagraysalt, result_9salt, Size(9, 9), 10, 20);
////��ʾ���и�˹����ͼ��
//imshow("equalLena_gauss", lenagraynoise);
//imshow("result_5gauss", result_5gauss);
//imshow("result_9gauss", result_9gauss);
////��ʾ���н�������ͼ��
//imshow("equalLena_salt", lenagraysalt);
//imshow("result_5salt", result_5salt);
//imshow("result_9salt", result_9salt);

//�������˲�  ��ֵ�˲�����������

//Mat imgResult3, grayResult3, imgResult9, grayResult9;
////�ֱ�Ժ��н��������Ĳ�ɫ�ͻҶ�ͼ������˲����˲�ģ��Ϊ3��3
//medianBlur(lenasalt, imgResult3, 3);
//medianBlur(lenagraysalt, grayResult3, 3);
////�Ӵ��˲�ģ�壬ͼ���˲�������ģ��
//medianBlur(lenasalt, imgResult9, 9);
//medianBlur(lenagraysalt, grayResult9, 9);
////��ʾ�˲�������
//imshow("img", lenasalt);
//imshow("gray", lenagraysalt);
//imshow("imgResult3", imgResult3);
//imshow("grayResult3", grayResult3);
//imshow("imgResult9", imgResult9);
//imshow("grayResult9", grayResult9);

//�ɷ����˲�
//X,Y��������ֱ�

//��Ե��� sobel scharr����

//Mat sobelX,sobelY,sobelXY;
//Sobel(lenagray, sobelX, CV_16S,1,0,3);
//Sobel(lenagray, sobelY, CV_16S, 0,1,3);
//convertScaleAbs(sobelX, sobelX);
//convertScaleAbs(sobelY, sobelY);
//sobelXY = sobelX + sobelY;
//Mat scX, scY, scXY;
//Scharr(lenagray, scX, CV_16S, 1, 0);
//Scharr(lenagray, scY, CV_16S, 0, 1);
//convertScaleAbs(scX, scX);
//convertScaleAbs(scY, scY);
//scXY = scX + scY;
//Mat BX, BY, CX, CY;
//getDerivKernels(BX,BY,1,0,3);
//BX = BX.reshape(CV_8U, 1);
//Mat BXY = BY * BX;
//getDerivKernels(CX, CY, 1, 0, FILTER_SCHARR);
//CX = CX.reshape(CV_8U, 1);
//Mat CXY = CY * CX;
//cout << BXY << endl;
//cout << CXY << endl;

//�Ƚ��и�˹ģ��ȥ������   ������˹����(�����޹أ���������Ӱ��)   canny����

//GaussianBlur(lenagray, lenagray,Size(3,3),5);
//Mat lap, can;
//Laplacian(lenagray, lap, CV_16S, 3, 1);
//convertScaleAbs(lap, lap);
//Canny(lenagray, can, 80, 120,3);

//��ͨ�����


//ͼ�����任     ŷʽ�����������̾���

//ͼ����̬ѧ  ��ʴ��ȥ��΢С�����룩�ṹԪ���жϱ�������ȥ��      ����

//////�������ڸ�ʴ��ԭͼ��
//Mat src = (Mat_<uchar>(6, 6) << 0, 0, 0, 0, 255, 0,
//0, 255, 255, 255, 255, 255,
//0, 255, 255, 255, 255, 0,
//0, 255, 255, 255, 255, 0,
//0, 255, 255, 255, 255, 0,
// 0, 0, 0, 0, 0, 0);
//Mat struct1 = getStructuringElement(0, Size(3, 3));
//Mat struct2 = getStructuringElement(1, Size(3, 3));
//Mat src_erode;
//erode(src, src_erode, struct2);
////����  ͨ���ṹԪ�������Ԫ��
//Mat src_dilate;
//dilate(src, src_dilate, struct2);

//��̬ѧӦ��  �����ȸ�ʴ�����ͣ�ȥ��ͼ���е�����,������С��ͨ��,�����ϴ���ͨ�򣩣��գ��������ٸ�ʴ��ȥ����ͨ���ڵ�С�Ϳն�,ƽ����������,���������ٽ�����ͨ��
//��̬ѧ�ݶ�   ��ñ  ��ñ   ���л���������

////������֤��̬ѧӦ�õĶ�ֵ������
//Mat src = (Mat_<uchar>(9, 12) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
//0, 255, 255, 255, 0, 255, 255, 255, 0, 0, 0, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0,
//0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
//0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
////namedWindow("src", WINDOW_NORMAL);  //�������ɵ�����ʾͼ��ĳߴ�
//Mat kernel = getStructuringElement(0, Size(3, 3));
//Mat open, close, gradient,tophat, blackhat, hitmiss,lenaopen;
//morphologyEx(src, open,MORPH_OPEN,kernel);					//�ȸ�ʴ������
//morphologyEx(src, close, MORPH_CLOSE, kernel);				//�������ٸ�ʴ
//morphologyEx(src, gradient, MORPH_GRADIENT, kernel);	//���ͼ�ȥ��ʴ
//morphologyEx(src, tophat, MORPH_TOPHAT, kernel);			//ԭͼ��������
//morphologyEx(src, blackhat, MORPH_BLACKHAT, kernel);	//�������ԭͼ
//morphologyEx(src, hitmiss, MORPH_HITMISS, kernel);			//��ȫ��ͬ����
//morphologyEx(lenagray, lenaopen, MORPH_OPEN, kernel);

//ͼ��ϸ�� �Ǽܻ� include ximgproc


//�������


//���������������


//͹�����



//ֱ�߼�� ����任��ͼ�οռ�������ռ��б任���ü������ʾ��



//�㼯��� fitline   minencodingcircle/triangle


//QR��ά��ʶ��  QRcodedetector


//����ͼ�����Ϸ�������ͣ���ֹ�ظ�����

//Mat lenaint, lenaintsqr, lenainttr;
//integral(lenagray,lenaint);
//normalize(lenaint, lenaint, 0, 255, NORM_MINMAX);
//integral(lenagray, lenaint, lenaintsqr);
//normalize(lenaintsqr, lenaintsqr, 0, 255, NORM_MINMAX);
//integral(lenagray, lenaint, lenaintsqr,lenainttr);
//normalize(lenainttr, lenainttr, 0, 255, NORM_MINMAX);
//normalize(lenaint, lenaint, 0, 255, NORM_MINMAX);

//ͼ��ָ� ��ˮ������ˮ�Ϳգ�

//RNG rng(10086);
//int connect = 4;
//int maskval = 255;
//int flag = connect | (maskval << 8) | FLOODFILL_FIXED_RANGE;
//Scalar updiff = Scalar(20, 20, 20);
//Scalar lowdiff = Scalar(20, 20, 20);
//Mat mask = Mat::zeros(lena.rows+2, lena.cols+2, CV_8UC1);
//while (true)
//{
//	int py = rng.uniform(0, lena.rows - 1);
//	int px = rng.uniform(0, lena.cols - 1);
//	Point point = Point(px, py);
//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//	int area = floodFill(lena, mask,point, color,nullptr, lowdiff, updiff, flag);
//	imshow("lena", lena);
//	imshow("mask", mask);
//	int k = waitKey(0);
//	if ((k & 255)==27)
//	{
//		break;
//	}
//}

//��ˮ�뷨��������עˮ�㣬עˮ�ҵ���ˮ�룩

//Mat lenashed;
//watershed(lena,lenashed);

//Harris�ǵ���

////ͼ���ݶȼ��㣺���ȣ��������ͼ��Ӧ�� Sobel ���ӻ������ݶ����ӣ�����ͼ���� x �� y �����ϵ��ݶȡ��⽫���������ݶ�ͼ��dx �� dy������ʾͼ����ÿ�����ص���ݶȴ�С�ͷ���
////����Э������󣺶���ÿ�����ص㣬ͨ������������Χ�����е��ݶ�ֵ��Э��������������ǵ�Ŀ����ԡ�Э������������Ӧ���ص�� x �� y �����ݶȵ�ƽ�����Լ����ǵĳ˻���
////����ǵ���Ӧ����������Э������������ֵ������ǵ���Ӧ������ͨ����ʹ��������Ӧ���� R �������ǵ����Ҫ�ԣ�
////R = ��1 * ��2 - k * (��1 + ��2) ^ 2
////���У���1 �� ��2 ��Э������������ֵ��k ��һ�����鳣����
////�Ǽ���ֵ���ƣ��ڼ���ǵ���Ӧ�����󣬸�����Ӧ������ֵ����ÿ�����ص���зǼ���ֵ���ơ�����ζ��ֻ�е���ǰ���ص����Ӧֵ������Χ���ص�������ʱ�򣬲Ž������ص���Ϊ�ǵ㡣
////������ֵ�����Ը���Ӧ�õ���������һ����ֵ��ȥ����Ӧ����������ֵ�Ľǵ㡣
//Mat harris;
//cornerHarris(lena, harris, 2, 3,0.04);
//Mat harris_norm;
//normalize(harris, harris_norm,0,255,NORM_MINMAX);
//convertScaleAbs(harris_norm, harris_norm);
//vector<KeyPoint>points;
//for (int row = 0; row < harris_norm.rows; row++)
//{
//	for (int col = 0; col < harris_norm.cols; col++)
//	{
//		int R = harris_norm.at<uchar>(row, col);
//		if (R>125)
//		{
//			KeyPoint kp;
//			kp.pt.x = col;
//			kp.pt.y = row;
//			points.push_back(kp);
//		}
//	}
//}
//drawKeypoints(lena, points, lena);
//imshow("cornermap", harris_norm);
//imshow("corner", lena);

//hsi-Tomas�ǵ��� �ж��Ƿ�Ϊ�ǵ㲻ͬ

//vector<Point2f> corners;
//goodFeaturesToTrack(lenagray, corners,100,0.01,0.04,Mat());
//vector<KeyPoint>kp;
//for (int i = 0; i < corners.size(); i++)
//{
//	KeyPoint Kp ;
//	Kp.pt = corners[i];
//	kp.push_back(Kp);
//}
//drawKeypoints(lena, kp, lena);

//�ǵ����������Ż�

//vector<Point2f> corners;
//goodFeaturesToTrack(lenagray, corners, 100, 0.01, 0.04, Mat());
//vector<Point2f>kpba = corners;
//TermCriteria cri = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.01);
//cornerSubPix(lenagray, corners,Size(3,3),Size(-1,-1),cri);
//for (int i = 0; i < corners.size(); i++)
//{
//	cout << kpba[i]<< endl;
//	cout << corners[i] << endl;
//}
//vector<KeyPoint>kp;
//for (int i = 0; i < corners.size(); i++)
//{
//	KeyPoint Kp;
//	Kp.pt = corners[i];
//	kp.push_back(Kp);
//}
//drawKeypoints(lena, kp, lena);

//ORB������

//Ptr<ORB> orb =  ORB::create();
//vector<KeyPoint> kps;
//orb->detect(lena, kps);
//Mat desc;
//orb->compute(lena, kps, desc);
//drawKeypoints(lena, kps, lena);
//drawKeypoints(lena, kps, lena,Scalar(0,0,0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//namedWindow("lena", WINDOW_FREERATIO);
//imshow("lena", lena);

//������ƥ��

//vector<KeyPoint>kps1, kps2;
//Mat des1, des2;
//ord_feature(lenagray, kps1, des1);
//ord_feature(lenahead, kps2, des2);
//vector<DMatch>dm;
//BFMatcher matcher(NORM_HAMMING);
//matcher.match(des1, des2, dm);
//double maxdis = 0, mindis = 10000;
//for (int i = 0; i < dm.size(); i++)
//{
//	double dis = dm[i].distance;
//	if (dis > maxdis) maxdis = dis;
//	if (dis < mindis) mindis = dis;
//}
//cout << maxdis << mindis << endl;
//vector<DMatch>gdm;
//for (int i = 0; i < dm.size(); i++)
//{
//	double dis = dm[i].distance;
//	if (dis <= max(2 * mindis, 20.0)) {
//		gdm.push_back(dm[i]);
//	}
//}
//Mat out;
//drawMatches(lenagray, kps1, lenahead, kps2, gdm, out);
//namedWindow("out", WINDOW_FREERATIO);
//imshow("out", out);

//RANSAC�������Ż�

//vector<KeyPoint>kps1, kps2;
//Mat des1, des2;
//ord_feature(lenagray, kps1, des1);
//ord_feature(lenahead, kps2, des2);
//BFMatcher matcher(NORM_HAMMING);
//vector<DMatch>dm, gdm,grs;
//matcher.match(des1, des2, dm);
//match_min(dm, gdm);
//ransac(gdm, kps1, kps2, grs);
//cout << gdm.size() << grs.size() << endl;
//Mat out, sc,org;
//drawMatches(lenagray, kps1, lenahead, kps2, dm, org);
//namedWindow("org", WINDOW_FREERATIO);
//imshow("org", org);
//drawMatches(lenagray, kps1, lenahead, kps2, gdm, out);
//namedWindow("out", WINDOW_FREERATIO);
//imshow("out", out);
//drawMatches(lenagray, kps1, lenahead, kps2, grs, sc);
//namedWindow("grc", WINDOW_FREERATIO);
//imshow("grc", sc);

//��Ŀ���ģ�ͣ��궨

//vector<Mat>imgs;
//string imgname;
//ifstream fin("pic.txt");
//while (getline(fin,imgname))
//{
//	Mat pic = imread(imgname, IMREAD_GRAYSCALE);
//	imgs.push_back(pic);
//}
//vector<vector<Point2f>> imgpts;
//for (int i = 0; i < imgs.size(); i++)
//{
//	Mat img1 = imgs[i];
//	vector<Point2f> cor;
//	findChessboardCorners(img1, Size(6, 8),cor);
//	find4QuadCornerSubpix(img1, cor,Size(11,11));
//	//drawChessboardCorners(img1, Size(6, 8), cor,true);
//	//imshow("img1", img1);
//	//waitKey(0);
//	imgpts.push_back(cor);
//}
//Size rect = Size(1, 1);
//vector<vector<Point3f>> objpts;
//for (int i = 0; i < imgpts.size(); i++)
//{
//	vector<Point3f>pt;
//	for (int x = 1; x < 7; x++)
//	{
//		for (int y = 1; y < 9;y++)
//		{
//			Point3f realpt;
//			realpt.x = rect.width * x;
//			realpt.y = rect.height * y;
//			realpt.z = 0;
//			pt.push_back(realpt);
//		}
//	}
//	objpts.push_back(pt);
//}
//Size picsize;
//picsize.width = imgs[0].cols;
//picsize.height = imgs[0].rows;
//Mat cameramatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
//Mat diffmatrix = Mat(1, 5, CV_32FC1, Scalar::all(0));
//vector<Mat> rotation;
//vector<Mat> mov;
//calibrateCamera(objpts, imgpts, picsize, cameramatrix, diffmatrix, rotation, mov);
//cout << cameramatrix << endl;
//cout << diffmatrix << endl;
//for (int i = 0; i < rotation.size(); i++)
//{
//	cout << rotation[i] << endl;
//	cout << mov[i] << endl;
//}



waitKey(0);
return 0;
}