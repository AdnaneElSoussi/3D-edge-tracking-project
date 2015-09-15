#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace cv;
int const numberofvert=8;
int const numberofedges=12;
float const number_of_control_points=20;
/// The following three functions perform the thinning step. Coding of the three functions was obtained from:
/// "http://answers.opencv.org/question/3207/what-is-a-good-thinning-algorithm-for-getting-the-skeleton-of-characters-for-ocr/"
/// ------------------------------------------------------------------------------------------ ///

void ThinSubiteration1(Mat & pSrc, Mat & pDst)
{
    int rows = pSrc.rows;
    int cols = pSrc.cols;
    pSrc.copyTo(pDst);
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if(pSrc.at<float>(i, j) == 1.0f)
            {
                /// get 8 neighbors
                /// calculate C(p)
                int neighbor0 = (int) pSrc.at<float>( i-1, j-1);
                int neighbor1 = (int) pSrc.at<float>( i-1, j);
                int neighbor2 = (int) pSrc.at<float>( i-1, j+1);
                int neighbor3 = (int) pSrc.at<float>( i, j+1);
                int neighbor4 = (int) pSrc.at<float>( i+1, j+1);
                int neighbor5 = (int) pSrc.at<float>( i+1, j);
                int neighbor6 = (int) pSrc.at<float>( i+1, j-1);
                int neighbor7 = (int) pSrc.at<float>( i, j-1);
                int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
                        int(~neighbor3 & ( neighbor4 | neighbor5)) +
                        int(~neighbor5 & ( neighbor6 | neighbor7)) +
                        int(~neighbor7 & ( neighbor0 | neighbor1));
                if(C == 1)
                {
                    /// calculate N
                    int N1 = int(neighbor0 | neighbor1) +
                             int(neighbor2 | neighbor3) +
                             int(neighbor4 | neighbor5) +
                             int(neighbor6 | neighbor7);
                    int N2 = int(neighbor1 | neighbor2) +
                             int(neighbor3 | neighbor4) +
                             int(neighbor5 | neighbor6) +
                             int(neighbor7 | neighbor0);
                    int N = min(N1,N2);
                    if ((N == 2) || (N == 3))
                    {
                        /// calculate criteria 3
                        int c3 = ( neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
                        if(c3 == 0)
                        {
                            pDst.at<float>( i, j) = 0.0f;
                        }
                    }
                }
            }
        }
    }
}


void ThinSubiteration2(Mat & pSrc, Mat & pDst)
{
    int rows = pSrc.rows;
    int cols = pSrc.cols;
    pSrc.copyTo( pDst);
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if (pSrc.at<float>( i, j) == 1.0f)
            {
                /// get 8 neighbors
                /// calculate C(p)
                int neighbor0 = (int) pSrc.at<float>( i-1, j-1);
                int neighbor1 = (int) pSrc.at<float>( i-1, j);
                int neighbor2 = (int) pSrc.at<float>( i-1, j+1);
                int neighbor3 = (int) pSrc.at<float>( i, j+1);
                int neighbor4 = (int) pSrc.at<float>( i+1, j+1);
                int neighbor5 = (int) pSrc.at<float>( i+1, j);
                int neighbor6 = (int) pSrc.at<float>( i+1, j-1);
                int neighbor7 = (int) pSrc.at<float>( i, j-1);
                int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
                        int(~neighbor3 & ( neighbor4 | neighbor5)) +
                        int(~neighbor5 & ( neighbor6 | neighbor7)) +
                        int(~neighbor7 & ( neighbor0 | neighbor1));
                if(C == 1)
                {
                    /// calculate N
                    int N1 = int(neighbor0 | neighbor1) +
                             int(neighbor2 | neighbor3) +
                             int(neighbor4 | neighbor5) +
                             int(neighbor6 | neighbor7);
                    int N2 = int(neighbor1 | neighbor2) +
                             int(neighbor3 | neighbor4) +
                             int(neighbor5 | neighbor6) +
                             int(neighbor7 | neighbor0);
                    int N = min(N1,N2);
                    if((N == 2) || (N == 3))
                    {
                        int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
                        if(E == 0)
                        {
                            pDst.at<float>(i, j) = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void normalizeLetter(Mat & inputarray, Mat & outputarray)
{
    bool bDone = false;
    int rows = inputarray.rows;
    int cols = inputarray.cols;

    inputarray.convertTo(inputarray,CV_32FC1);

    inputarray.copyTo(outputarray);

    outputarray.convertTo(outputarray,CV_32FC1);

    /// pad source
    Mat p_enlarged_src = Mat(rows + 2, cols + 2, CV_32FC1);
    for(int i = 0; i < (rows+2); i++)
    {
        p_enlarged_src.at<float>(i, 0) = 0.0f;
        p_enlarged_src.at<float>( i, cols+1) = 0.0f;
    }
    for(int j = 0; j < (cols+2); j++)
    {
        p_enlarged_src.at<float>(0, j) = 0.0f;
        p_enlarged_src.at<float>(rows+1, j) = 0.0f;
    }
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if (inputarray.at<float>(i, j) >= 20.0f)
            {
                p_enlarged_src.at<float>( i+1, j+1) = 1.0f;
            }
            else
                p_enlarged_src.at<float>( i+1, j+1) = 0.0f;
        }
    }

    /// start to thin
    Mat p_thinMat1 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
    Mat p_thinMat2 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
    Mat p_cmp = Mat::zeros(rows + 2, cols + 2, CV_8UC1);

    while (bDone != true)
    {
        /// sub-iteration 1
        ThinSubiteration1(p_enlarged_src, p_thinMat1);
        /// sub-iteration 2
        ThinSubiteration2(p_thinMat1, p_thinMat2);
        /// compare
        compare(p_enlarged_src, p_thinMat2, p_cmp, CV_CMP_EQ);
        /// check
        int num_non_zero = countNonZero(p_cmp);
        if(num_non_zero == (rows + 2) * (cols + 2))
        {
            bDone = true;
        }
        /// copy
        p_thinMat2.copyTo(p_enlarged_src);
    }
    // copy result
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            outputarray.at<float>( i, j) = p_enlarged_src.at<float>( i+1, j+1);
        }
    }
}
/// ------------------------------------------------------------------------------------------ ///
/// This function calculates the 4x4 homogeneous transformation matrix between the object frame of reference (w) and the camera frame of reference (c) ///
void euler2ext(float A, float B, float C ,float X, float Y, float Z,Mat rotmat,float theta);
/// This function creates the 3x3 camera intrinsic matrix ///
void IntrnMatCreate(Mat IntrnMat);
/// This function converts the vertex coordinates from the object frame of reference (w) to the camera frame of reference (c) ///
void tocameraspace(float vertices[][numberofvert],Mat rotmat,Mat newvertices);
/// This function applies the pinhole projection model and creates the 2-D projection of the tracked structure ///
void topixels(Mat newvertices,Mat pixelcoords,Mat IntrnMat);
/// This function evaluates the likelihood of the particle ///
float comp(Mat image,Mat pixelcoords, Mat frame, int sides[][2]);
/// This function applies the sobel kernel and finds the edges in the image ///
void Calaculate_edges(Mat frame, Mat edges,float tol);

/// This function is for displaying the particles' projections///
void draw(Mat image,Mat pixelcoords, float score, int sides[][2]);
int main()
{
/// Video capture ///
    VideoCapture cap;
    cap.open(1); /// Change number "1" to use a different camera or replace with file name for an offline video
    Mat frame; /// Matrix containg current video frame
    cap >> frame;
/// The model of tracked structure as vertices and sides ///
    float vertices[3][8]= {{0,0,5,5,0,0,5,5},{0,5,5,0,0,5,5,0},{2.5,2.5,2.5,2.5,0,0,0,0}};
    int sides[12][2]= {{0,1},{1,2},{2,3},{3,0},{0,4},{1,5},{2,6},{3,7},{4,5} ,{5,6} ,{6,7} ,{4,7}};
///////////////////////////////////////////////////////////////////////////////////
    int sizeofINT[2]= {3,3};
    Mat IntrnMat(2,sizeofINT, CV_32FC1, cv::Scalar(0)); /// initializing matrix containg the inrinsic camera parameters
    int numberofvertices= sizeof(vertices)/sizeof(vertices[0][0])/3; /// calculates number of vertices in the object model
    Mat pixelcoords=Mat::zeros(2,numberofvertices, CV_32FC1); /// initializing matrix which contains the inrinsic camera parameters
    int sizeofrot[2]= {4,4};
    Mat rotmat(2,sizeofrot, CV_32FC1, cv::Scalar(0)); /// initializing matrix which contains 4x4 homogeneous transformation
    ///  matrix between the object frame of reference (w) and the camera frame of reference (c)
    int sizeofvert[2]= {3,numberofvertices};
    Mat newvertices(2,sizeofvert, CV_32FC1, cv::Scalar(0)); /// initializing matrix containg the vertices in the camera frame of reference
    clock_t tstart, tend; /// Used to calculated time of processing per frame
    double timer; /// Used to calculated time of processing per frame
    Mat image=Mat::zeros(240,320, CV_8UC1);
    float likelyhood,my_rand,my_randx,my_randy,my_randz; /// initializing dummy variables which will contain particle likelihood and a random number respectively
////// initialize displays ///////////
    namedWindow("frame", 1); /// create display window
    namedWindow("all parts", 1); /// create display window

    int film_count=0; /// over all video counter
////// reading camera input //////////
    cap >> frame;
    resize(frame, frame, Size(320,240), 0, 0, INTER_LINEAR ); /// resizing camera input
    cv::Size s = frame.size(); /// extracting frame size

    /// ///////// intialize video creation /////// ///
    VideoWriter writer;
    int fourcc=CV_FOURCC('D','I','V','3');
    writer=VideoWriter("movie9.avi",fourcc, 30, s, 1);
    VideoWriter writer2;
    writer2=VideoWriter("movie29.avi",fourcc, 30, s, 1);
    /////////////////////////////////////////////////

    /// Extracting frame size to initialize other variables ///
    int height = s.height;
    int width = s.width;
    int sizeofspace[2] = { height, width };
    //////////////////////////////////////////////////////////

    Mat edges(2,sizeofspace, CV_8UC1, cv::Scalar(0)); /// This matrix will contain the edge image
    Mat direction(2,sizeofspace, CV_8UC1, cv::Scalar(0));  /// This matrix will contain the color encoded edge image
    Mat cleared(2,sizeofspace, CV_8UC1, cv::Scalar(0)); /// This empty matrix will be used for clearing purposes
    //////////////////////////////////////
    int number_of_particles=800; /// Number of particles

    /// ----------- STATE MATRIX ------------ ///
    int sizeofstates[2]= {number_of_particles,9}; /// States matrix size.
    Mat States(2,sizeofstates, CV_32FC1, cv::Scalar(0));/// States matrix
    /// 1st, 2nd and 3rd columns contain x,y and z position of the particle respectively.
    /// 3rd, 4th and 5th columns contain the x,y and postion of the particle's look at point respectively.
    /// 7th column contains the weight of every particle
    /// 8th column contains the angle of the camera's up vector with the z-axis in degrees
    ///-------------------------------------///

    /// -------DUMMY VARIABLES TO BE EXPLAINED LATER-------- ///
    int intialize_counter=0;
    int z1;
    Mat a=Mat::zeros(1,1, CV_32FC1);
    Mat weights=Mat::zeros(number_of_particles,1, CV_32FC1);
    cv::Mat sum_vect(1, States.cols, States.type());
    /// gets the weights column
    cv::Mat one = States.col(6);
    /// sorts the weights column and save indices in idx
    cv::Mat1i idx;
    cv::Mat New_States(States.rows, States.cols, States.type());
    int number_of_resampled_parts=0;
    int num_to_sample,resamp,part_num=0;
    cv::Mat sum_vect1(1, States.cols, States.type());
    Mat image1=Mat::zeros(240,320, CV_8UC1);
    Mat to_display=Mat::zeros(240,320, CV_8UC3);
    to_display=frame.clone();
    Mat to_display2(240,320, CV_8UC3, cv::Scalar(150,100,100));
    Mat to_display2c(240,320, CV_8UC3, cv::Scalar(150,150,150));
    float score;
    cv::Mat max_vect(1, States.cols, States.type());
    cv::Mat avg_vect(1, States.cols, States.type());
    float lenght_of_vect;
    Size  ksize; ksize.width = 7; ksize.height = 7;

    ///-------------------------------------///


/////////////////////////////////////////////////////////////////////////////

    for (film_count=0; film_count<5000; film_count++)
    {


        tstart = clock();
/// //////////// initialize states vector //////////////////////// ///
        if (film_count==0)
        {
            for(intialize_counter=0; intialize_counter<number_of_particles; intialize_counter++)
            {
                /// camera look at ///
                lenght_of_vect=sqrt(15*15*3);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 0) = (float)(0+my_rand*lenght_of_vect*0.05*5);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 1) = (float)(0+my_rand*lenght_of_vect*0.05*5);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 2) = (float)(0+my_rand*lenght_of_vect*0.05*5);
                States.at<float>(intialize_counter, 8) = (float)(0);
                ///////////////////////

                /// camera position ///
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 3) = (float)(-15+my_rand*5);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 4) = (float)(-15+my_rand*5);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 5) = (float)(15+my_rand*5);
                ////////////////////

                /// weight ///
                States.at<float>(intialize_counter, 6) = (float)(1.0/number_of_particles);
                ////////////////////

            }
        }
        else
        {
            for(intialize_counter=0; intialize_counter<number_of_particles; intialize_counter++)
            {

                /// camera look at ///
                lenght_of_vect=(float) (
                                   (States.at<float>(intialize_counter, 0)-States.at<float>(intialize_counter, 3))*(States.at<float>(intialize_counter, 0)-States.at<float>(intialize_counter, 3))+
                                   (States.at<float>(intialize_counter, 1)-States.at<float>(intialize_counter, 4))*(States.at<float>(intialize_counter, 1)-States.at<float>(intialize_counter, 4))+
                                   (States.at<float>(intialize_counter, 2)-States.at<float>(intialize_counter, 5))*(States.at<float>(intialize_counter, 2)-States.at<float>(intialize_counter, 5)) );
                lenght_of_vect=sqrt(lenght_of_vect);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 0) = (float)(States.at<float>(intialize_counter, 0)+my_rand*lenght_of_vect*0.05*2);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 1) = (float)(States.at<float>(intialize_counter, 1)+my_rand*lenght_of_vect*0.05*2);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 2) = (float)(States.at<float>(intialize_counter, 2)+my_rand*lenght_of_vect*0.05*2);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 8) = (float)(States.at<float>(intialize_counter, 8)+my_rand);

                /// camera position ///
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 3) = (float)(States.at<float>(intialize_counter, 3)+my_rand*5);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 4) = (float)(States.at<float>(intialize_counter, 4)+my_rand*5);
                my_rand=(rand()%10000)/10000.0-0.5;
                States.at<float>(intialize_counter, 5) = (float)(States.at<float>(intialize_counter, 5)+my_rand*5);


            }


        }
////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////


///////////////// Frame Processing    /////////////////////
        if (film_count>0)
        {
            cap >> frame;
            resize(frame, frame, Size(320,240), 0, 0, INTER_LINEAR );
            to_display=frame.clone();
            to_display2=to_display2c.clone();
            edges=cleared.clone();
            direction=cleared.clone();
            image=cleared.clone();
            image1=cleared.clone();
        }
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        GaussianBlur(frame, frame, ksize, 0, 0, BORDER_DEFAULT);
        Calaculate_edges(frame, edges,10);

       // dilate(edges, edges, Mat(), Point(-1, -1), 1, 1, 1);
        //normalizeLetter(edges, edges);
      //  Calaculate_dir_and_spread(direction, edges);
        dilate(edges, direction, Mat(), Point(-1, -1), 2, 1, 1);
        imshow("frame",direction);
      ///  imwrite("direction2.jpg",direction);
/// DRAW maxima ///
/*
        for(intialize_counter=0; intialize_counter<number_of_particles; intialize_counter++)
        {
            euler2ext(States.at<float>(intialize_counter, 0),States.at<float>(intialize_counter, 1),States.at<float>(intialize_counter, 2),States.at<float>(intialize_counter,3)
                      ,States.at<float>(intialize_counter, 4),States.at<float>(intialize_counter, 5), rotmat,States.at<float>(intialize_counter, 8));
            IntrnMatCreate( IntrnMat);
            tocameraspace( vertices, rotmat, newvertices);
            topixels(newvertices,pixelcoords,IntrnMat);
            score=(float)(New_States.at<float>(intialize_counter, 6)/max_vect.at<float>(0,6) );
            draw(to_display2,pixelcoords,score,sides);
        }

*/


///////////////// Likelyhood function /////////////////////
        for(intialize_counter=0; intialize_counter<number_of_particles; intialize_counter++)
        {
            euler2ext(States.at<float>(intialize_counter, 0),States.at<float>(intialize_counter, 1),States.at<float>(intialize_counter, 2),States.at<float>(intialize_counter,3)
                      ,States.at<float>(intialize_counter, 4),States.at<float>(intialize_counter, 5), rotmat,States.at<float>(intialize_counter, 8));
            IntrnMatCreate( IntrnMat);
            tocameraspace( vertices, rotmat, newvertices);
            topixels(newvertices,pixelcoords,IntrnMat);
            ///////////// Update Weights /////////////////////////
            likelyhood=comp(image,pixelcoords,direction,sides);
            States.at<float>(intialize_counter, 6) =exp(2*likelyhood)*States.at<float>(intialize_counter, 6)+pow(10,-4);
            States.at<float>(intialize_counter, 7) =likelyhood;
        }
//////////// Normalize Weights //////////////////////

        reduce(States, sum_vect, 0, CV_REDUCE_SUM, -1);
        States.col(6)=States.col(6)/sum_vect.at<float>(0,6);

/////////// ReSampling Particles /////////////////

/// Sorting ///

   one = States.col(6);
        cv::sortIdx(one, idx, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);

        for(int y = 0; y < States.rows; y++)
        {
            States.row(idx(0,y)).copyTo(New_States.row(y));
        }
        States=New_States.clone();
//cout << New_States.col(6) <<endl;
        number_of_resampled_parts=0;
        part_num=0;
        while(number_of_resampled_parts<number_of_particles)
        {
            num_to_sample=ceil((float)States.at<float>(part_num, 6)*number_of_particles);
            if(part_num<number_of_particles && (number_of_resampled_parts+resamp)<number_of_particles )
            {
                for (resamp=0; resamp<num_to_sample; resamp++)
                {
                    States.row(part_num).copyTo(New_States.row(number_of_resampled_parts+resamp));
                    /// camera look at  ///
                    my_rand=(rand()%10000)/10000.0-0.5;
                    New_States.at<float>(number_of_resampled_parts+resamp, 0) = (float)(New_States.at<float>(number_of_resampled_parts+resamp, 0)+my_rand/(States.at<float>(part_num, 7)*2+1));
                    my_rand=(rand()%10000)/10000.0-0.5;
                    New_States.at<float>(number_of_resampled_parts+resamp, 1) = (float)(New_States.at<float>(number_of_resampled_parts+resamp, 1)+my_rand/(States.at<float>(part_num, 7)*2+1));
                    my_rand=(rand()%10000)/10000.0-0.5;
                    New_States.at<float>(number_of_resampled_parts+resamp, 2) = (float)(New_States.at<float>(number_of_resampled_parts+resamp, 2)+my_rand/(States.at<float>(part_num, 7)*2+1));
                    my_rand=(rand()%10000)/10000.0-0.5;
                    New_States.at<float>(number_of_resampled_parts+resamp, 8) = (float)(New_States.at<float>(number_of_resampled_parts+resamp, 8)+my_rand);

                    /// camera position ///
                    my_rand=(rand()%10000)/10000.0-0.5;
                    New_States.at<float>(number_of_resampled_parts+resamp, 3) = (float)(New_States.at<float>(number_of_resampled_parts+resamp, 3)+my_rand/(States.at<float>(part_num, 7)*2+1));
                    my_rand=(rand()%10000)/10000.0-0.5;
                    New_States.at<float>(number_of_resampled_parts+resamp, 4) = (float)(New_States.at<float>(number_of_resampled_parts+resamp, 4)+my_rand/(States.at<float>(part_num, 7)*2+1));
                    my_rand=(rand()%10000)/10000.0-0.5;
                    New_States.at<float>(number_of_resampled_parts+resamp, 5) = (float)(New_States.at<float>(number_of_resampled_parts+resamp, 5)+my_rand/(States.at<float>(part_num, 7)*2+1));
                }
            }
            number_of_resampled_parts=num_to_sample+number_of_resampled_parts;
            part_num=part_num+1;
        }
        States=New_States.clone();
//////////// Normalize Weights //////////////////////

        reduce(States, sum_vect1, 0, CV_REDUCE_SUM, -1);
        States.col(6)=States.col(6)/sum_vect1.at<float>(0,6);

////////////////////////////////display//////////////////////////////////
/////////////////////////
//get weights column
        one = States.col(6);
// sort the weights column and save indices in dst
        cv::sortIdx(one, idx, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
// now build your final matrix
        for(int y = 0; y < States.rows; y++)
        {
            States.row(idx(0,y)).copyTo(New_States.row(y));
        }

/////////////////////////

        reduce(States, max_vect, 0, CV_REDUCE_MAX, -1);
        reduce(States, avg_vect, 0, CV_REDUCE_AVG, -1);


/// DRAW AVG///
        euler2ext(avg_vect.at<float>(0, 0),avg_vect.at<float>(0, 1),avg_vect.at<float>(0, 2),avg_vect.at<float>(0,3)
                  ,avg_vect.at<float>(0, 4),avg_vect.at<float>(0, 5), rotmat,avg_vect.at<float>(0, 8));
        IntrnMatCreate( IntrnMat);
        tocameraspace( vertices, rotmat, newvertices);
        topixels(newvertices,pixelcoords,IntrnMat);
        draw(to_display,pixelcoords,-1,sides);
/// DRAW AVG///
       /* euler2ext(New_States.at<float>(number_of_particles-1, 0),New_States.at<float>(number_of_particles-1, 1),New_States.at<float>(number_of_particles-1, 2),New_States.at<float>(number_of_particles-1,3)
                  ,New_States.at<float>(number_of_particles-1, 4),New_States.at<float>(number_of_particles-1, 5), rotmat,New_States.at<float>(number_of_particles-1, 8));
        IntrnMatCreate( IntrnMat);
        tocameraspace( vertices, rotmat, newvertices);
        topixels(newvertices,pixelcoords,IntrnMat);
        draw(to_display,pixelcoords,1,sides);*/


        imshow("all parts",to_display);
       writer.write(to_display);
       // writer2.write(to_display2);
        tend = clock();
        timer=(tend- tstart);
std::cout << "It took "<< difftime(tend, tstart)/CLOCKS_PER_SEC<<" second(s)."<<std::endl;
        waitKey(1);
    }
    waitKey(0);

    return 0;
}
void euler2ext(float A, float B, float C ,float X, float Y, float Z,Mat rotmat,float theta)
{

    theta=theta*M_PI/180;
    Mat lookat=Mat::zeros(1,3, CV_32FC1);
    lookat.at<float>(0,0)=A;
    lookat.at<float>(0,1)=B;
    lookat.at<float>(0,2)=C;
    Mat pos=Mat::zeros(1,3, CV_32FC1);
    pos.at<float>(0,0)=X;
    pos.at<float>(0,1)=Y;
    pos.at<float>(0,2)=Z;
    Mat zc=Mat::zeros(1,3, CV_32FC1);
    zc=lookat-pos;
    zc=zc/norm(zc, NORM_L2, noArray());
    float w = cos(theta/2);
    float x = sin(theta/2)*zc.at<float>(0,0);
    float y = sin(theta/2)*zc.at<float>(0,1);
    float z = sin(theta/2)*zc.at<float>(0,2);
    float mag=sqrt(w*w+x*x+y*y+z*z);
    w=w/mag;
    x=x/mag;
    y=y/mag;
    z=z/mag;
    Mat R=Mat::zeros(3,3, CV_32FC1);
    Mat R2=Mat::zeros(3,3, CV_32FC1);

    R.at<float>(0,0) = 1 - 2*(y*y + z*z);
    R.at<float>(0,1) = 2*(x*y - z*w);
    R.at<float>(0,2) = 2*(x*z + y*w);

    R.at<float>(1,0) = 2*(x*y + z*w);
    R.at<float>(1,1) = 1 - 2*(x*x + z*z);
    R.at<float>(1,2) = 2*(y*z - x*w );

    R.at<float>(2,0) = 2*(x*z - y*w );
    R.at<float>(2,1) = 2*(y*z + x*w );
    R.at<float>(2,2) = 1 - 2 *(x*x + y*y);


    float D= zc.dot(pos);
    Mat yc=Mat::zeros(1,3, CV_32FC1);
    yc.at<float>(0,0)=(float)(pos.at<float>(0,0)-zc.at<float>(0,0) );
    yc.at<float>(0,1)=(float)(pos.at<float>(0,1)-zc.at<float>(0,1) );
    yc.at<float>(0,2)=(float) ( (D-zc.at<float>(0,0)*yc.at<float>(0,0)-zc.at<float>(0,1)*yc.at<float>(0,1))/(zc.at<float>(0,2)) );
    yc=yc-pos;
    yc=yc/norm(yc, NORM_L2, noArray());
    Mat xc=Mat::zeros(1,3, CV_32FC1);
    xc=yc.cross(zc);

    R2.at<float>( 0,0)=xc.at<float>(0,0);
    R2.at<float>( 1,0)=yc.at<float>(0,0);
    R2.at<float>( 2,0)=zc.at<float>(0,0);
    R2.at<float>( 0,1)=xc.at<float>(0,1);
    R2.at<float>( 1,1)=yc.at<float>(0,1);
    R2.at<float>( 2,1)=zc.at<float>(0,1);
    R2.at<float>( 0,2)=xc.at<float>(0,2);
    R2.at<float>( 1,2)=yc.at<float>(0,2);
    R2.at<float>( 2,2)=zc.at<float>(0,2);

    R=R2*R;



    rotmat.at<float>( 0,0)=R.at<float>( 0,0);
    rotmat.at<float>( 1,0)=R.at<float>( 1,0);
    rotmat.at<float>( 2,0)=R.at<float>( 2,0);
    rotmat.at<float>( 0,1)=R.at<float>( 0,1);
    rotmat.at<float>( 1,1)=R.at<float>( 1,1);
    rotmat.at<float>( 2,1)=R.at<float>( 2,1);
    rotmat.at<float>( 0,2)=R.at<float>( 0,2);
    rotmat.at<float>( 1,2)=R.at<float>( 1,2);
    rotmat.at<float>( 2,2)=R.at<float>( 2,2);
    rotmat.at<float>( 0,3)=-rotmat.at<float>( 0,0)*X-rotmat.at<float>( 0,1)*Y-rotmat.at<float>( 0,2)*Z;
    rotmat.at<float>( 1,3)=-rotmat.at<float>( 1,0)*X-rotmat.at<float>( 1,1)*Y-rotmat.at<float>( 1,2)*Z;
    rotmat.at<float>( 2,3)=-rotmat.at<float>( 2,0)*X-rotmat.at<float>( 2,1)*Y-rotmat.at<float>( 2,2)*Z;
    rotmat.at<float>( 3,3)=1;


}
void IntrnMatCreate(Mat IntrnMat)
{
    IntrnMat.at<float>( 0,0)= 362.6081;
    IntrnMat.at<float>( 0,2)= 171.2292;
    IntrnMat.at<float>( 1,1)= 361.3750;
    IntrnMat.at<float>( 1,2)= 129.7031;
    IntrnMat.at<float>( 2,2)= 1;
}

void tocameraspace(float vertices[][numberofvert],Mat rotmat,Mat newvertices)
{

    int sizeofINT[2]= {4,1};
    int i;
    float x,y,z;
    Mat vertex(2,sizeofINT, CV_32FC1, cv::Scalar(0));
    for  (i=0; i<numberofvert; i++)
    {
        vertex.at<float>( 0,0)=(float)vertices[0][i];
        vertex.at<float>( 1,0)=(float)vertices[1][i];
        vertex.at<float>( 2,0)=(float)vertices[2][i];
        vertex.at<float>( 3,0)=(float)1.0;
        vertex=rotmat*vertex;
        newvertices.at<float>( 0,i)= (float)vertex.at<float>( 0,0)/(float)vertex.at<float>( 3,0);
        newvertices.at<float>( 1,i)= (float)vertex.at<float>( 1,0)/(float)vertex.at<float>( 3,0);
        newvertices.at<float>( 2,i)= (float)vertex.at<float>( 2,0)/(float)vertex.at<float>( 3,0);

    }

}
void topixels(Mat newvertices,Mat pixelcoords,Mat IntrnMat)
{
    int sizeofINT[2]= {3,1};
    int i;
    float x,y,z;
    Mat vertex(2,sizeofINT, CV_32FC1, cv::Scalar(0));
    for  (i=0; i<numberofvert; i++)
    {
        vertex.at<float>( 0,0)=(float)newvertices.at<float>( 0,i);
        vertex.at<float>( 1,0)=(float)newvertices.at<float>( 1,i);
        vertex.at<float>( 2,0)=(float)newvertices.at<float>( 2,i);
        vertex=IntrnMat*vertex;
        pixelcoords.at<float>( 0,i)= (float)vertex.at<float>( 0,0)/(float)vertex.at<float>( 2,0);
        pixelcoords.at<float>( 1,i)= (float)vertex.at<float>( 1,0)/(float)vertex.at<float>( 2,0);
    }


}
float comp(Mat image,Mat pixelcoords, Mat frame, int sides[][2])
{
    int i=0,j,counter=0;
    float lengthofedge;
    int flag;
    cv::Size s = frame.size();
    int height = s.height;
    int width = s.width;
    float x1,y1,x2,y2,color,t,xn,yn,difference,score=0;
    for( i=0; i<numberofedges; i++)
    {

        x1=pixelcoords.at<float>( 1,sides[i][0]);
        y1=pixelcoords.at<float>( 0,sides[i][0]);
        x2=pixelcoords.at<float>( 1,sides[i][1]);
        y2=pixelcoords.at<float>( 0,sides[i][1]);
        t=0;
        color=(atan((y2-y1)/(x2-x1))+M_PI/2)/M_PI*254+1;
        if ((x2-x1)==0 && (y2-y1)!=0)
        {
            color=(M_PI/2+M_PI/2)/M_PI*254+1;
        }
        counter=0;
        while (t<=1)
        {
            xn=cvRound(x1+t*(x2-x1));
            yn=cvRound(y1+t*(y2-y1));
            if (xn>=0 && xn<height && yn>=0 && yn<width)
            {
                difference=abs((abs((float)frame.at<uchar>(xn, yn)-color)-1)/254*180);
                if (difference>90)
                {
                    difference=180-difference;
                }
                if (difference<40 && (float)frame.at<uchar>(xn, yn)!=0  )
                {
                    counter=counter+1;
                    image.at<uchar>(xn, yn) = (uchar)(255);
                }


            }
            t=t+1/number_of_control_points;
        }
        lengthofedge=( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) );
        flag=0;
        if (lengthofedge>(50*50))
        {
            flag=1;
        }
        score=counter/number_of_control_points+score;


    }
    return score;
}
void Calaculate_edges(Mat frame, Mat edges,float tol)
{
    cv::Size s = frame.size();
    int height = s.height;
    int width = s.width;
    int i, j;
    // xderivative kernel defined as sobelx
    float sobelx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
    float sobely[3][3] = { { -1,-2,-1 }, { 0,0,0}, { 1,2,1 } };

    float currentcolorx,currentcolory,currentcolor;
    for (i = (1); i<(height - (1)); i++)
    {
        for (j = (1); j<(width - (1)); j++)
        {

            currentcolorx = (float)(
                                frame.at<uchar>(i +1, j +1)*sobelx[0][0]
                                +frame.at<uchar>(i +1, j   )*sobelx[0][1]
                                +frame.at<uchar>(i +1, j -1)*sobelx[0][2]
                                +frame.at<uchar>(i   , j +1)*sobelx[1][0]
                                +frame.at<uchar>(i   , j   )*sobelx[1][1]
                                +frame.at<uchar>(i   , j -1)*sobelx[1][2]
                                +frame.at<uchar>(i -1, j +1)*sobelx[2][0]
                                +frame.at<uchar>(i -1, j   )*sobelx[2][1]
                                +frame.at<uchar>(i -1, j -1)*sobelx[2][2]
                            );
            currentcolory = (float)(
                                frame.at<uchar>(i +1, j +1)*sobely[0][0]
                                +frame.at<uchar>(i +1, j   )*sobely[0][1]
                                +frame.at<uchar>(i +1, j -1)*sobely[0][2]
                                +frame.at<uchar>(i   , j +1)*sobely[1][0]
                                +frame.at<uchar>(i   , j   )*sobely[1][1]
                                +frame.at<uchar>(i   , j -1)*sobely[1][2]
                                +frame.at<uchar>(i -1, j +1)*sobely[2][0]
                                +frame.at<uchar>(i -1, j   )*sobely[2][1]
                                +frame.at<uchar>(i -1, j -1)*sobely[2][2]
                            );
            currentcolor= sqrt(currentcolorx*currentcolorx+currentcolory*currentcolory)/sqrt(2080800)*255;

            if (currentcolor>tol)
            {

                    if ((currentcolory)==0 && (currentcolorx)!=0)
                    {
                        currentcolor=(M_PI/2+M_PI/2)/M_PI*254+1;
                    }
                    else
                    {
                    currentcolor= (atan(currentcolorx/currentcolory))/M_PI*254+1;
                    }
                edges.at<uchar>(i, j)=  (uchar)(currentcolor);
            }
            else
            {
                edges.at<uchar>(i, j)=  (uchar)(0);
            }


        }

    }
}

void draw(Mat image,Mat pixelcoords, float score, int sides[][2])
{
    int i=0,j,counter=0;
    cv::Size s = image.size();
    int height = s.height;
    int width = s.width;
    float x1,y1,x2,y2,color,t,xn,yn,difference;
    for( i=0; i<numberofedges; i++)
    {
        x1=pixelcoords.at<float>( 1,sides[i][0]);
        y1=pixelcoords.at<float>( 0,sides[i][0]);
        x2=pixelcoords.at<float>( 1,sides[i][1]);
        y2=pixelcoords.at<float>( 0,sides[i][1]);

        t=0;
        color=(atan((y2-y1)/(x2-x1))+M_PI/2)/M_PI*254+1;
        if ((x2-x1)==0 && (y2-y1)!=0)
        {
            color=(M_PI/2+M_PI/2)/M_PI*254+1;
        }
        counter=0;
        Point pt1,pt2;
        pt1.y=x1;
        pt1.x=y1;
        pt2.y=x2;
        pt2.x=y2;
        if (score>0)
        {
            line(image, pt1, pt2, Scalar(0,255*score,0), (1*score), CV_AA,0);
        }
        if (score==-1)
        {
            line(image, pt1, pt2, Scalar(0,0,255), (4), CV_AA,0);
        }

    }
}

