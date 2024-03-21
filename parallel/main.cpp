#include "iostream"
#include "opencv2/opencv.hpp"
#include "chrono"
#include "immintrin.h"

using namespace std;
using namespace cv;

void mosaic(Mat &image){
    double sizeBlock = 7;
    double width = image.cols;
    double height = image.rows;


    for (int i = 0; i < image.cols; i += sizeBlock){
        for (int j = 0; j < image.rows; j += sizeBlock){
            int summRed = 0;
            int summGreen = 0;
            int summBlue = 0;
            int count = 0;
            for (int x = i; x < min(i + sizeBlock, height); x++){
                for (int y = j; y < min(j + sizeBlock, width); y++){
                    Vec3b intensity = image.at<Vec3b>(x,y);
                    summRed += intensity[2];
                    summGreen += intensity[1];
                    summBlue += intensity[0];
                    count++;
                }
            }
            int averageRed = summRed / count;
            int averageGreen = summGreen / count;
            int averageBlue = summBlue / count;
            for (int x = i; x < min(i + sizeBlock, height); x++) {
                for (int y = j; y < min(j + sizeBlock, width); y++) {
                    image.at<Vec3b>(x, y) = Vec3b(averageBlue, averageGreen, averageRed);
                }
            }

        }
    }
}

void contrast(Mat &image){

    double factor = (100.0 + (-100.0)) / 100.0;
    factor *= factor;

    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            for (int x = 0; x < 3; x++){
                double intensity = image.at<Vec3b>(i, j)[x] / 255.0;
                intensity -= 0.5;
                intensity *= factor;
                intensity += 0.5;
                intensity *= 255;
                if (intensity < 0){
                    intensity = 0;
                }
                if (intensity > 255){
                    intensity = 255;
                }
                image.at<Vec3b>(i,j)[x] = saturate_cast<unsigned char >(intensity);
            }
        }
    }

}

void mosaicOmp(Mat &image){
    double sizeBlock = 7;
    double width = image.cols;
    double height = image.rows;


#pragma omp parallel for
    for (int i = 0; i < image.cols; i += sizeBlock){
        for (int j = 0; j < image.rows; j += sizeBlock){
            int summRed = 0;
            int summGreen = 0;
            int summBlue = 0;
            int count = 0;
            for (int x = i; x < min(i + sizeBlock, height); x++){
                for (int y = j; y < min(j + sizeBlock, width); y++){
                    Vec3b intensity = image.at<Vec3b>(x,y);
                    summRed += intensity[2];
                    summGreen += intensity[1];
                    summBlue += intensity[0];
                    count++;
                }
            }
            int averageRed = summRed / count;
            int averageGreen = summGreen / count;
            int averageBlue = summBlue / count;
            for (int x = i; x < min(i + sizeBlock, height); x++) {
                for (int y = j; y < min(j + sizeBlock, width); y++) {
                    image.at<Vec3b>(x, y) = Vec3b(averageBlue, averageGreen, averageRed);
                }
            }

        }
    }
}

void contrastOmp(Mat &image){

    double factor = (100.0 + (-100.0)) / 100.0;
    factor *= factor;

#pragma omp parallel for
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            for (int x = 0; x < 3; x++){
                double intensity = image.at<Vec3b>(i, j)[x] / 255.0;
                intensity -= 0.5;
                intensity *= factor;
                intensity += 0.5;
                intensity *= 255;
                if (intensity < 0){
                    intensity = 0;
                }
                if (intensity > 255){
                    intensity = 255;
                }
                image.at<Vec3b>(i,j)[x] = saturate_cast<unsigned char >(intensity);
            }
        }
    }

}


struct Pixeles{
    unsigned char b;
    unsigned char g;
    unsigned char r;
};

void vectorizationMosaic(Mat &image){
    int sizeBlock = 7;
    int width = image.cols;
    int height = image.rows;

    for (int i = 0; i < image.rows; i += sizeBlock){
        for (int j = 0; j < image.cols; j += sizeBlock){
            __m128i colors = _mm_setzero_si128();
            int count = 0;
            for (int y = i; y < min(i + sizeBlock, height); y++){
                for (int x = j; x < min(j + sizeBlock, width); x++){
                    Pixeles arr = image.at<Pixeles>(y,x);
                    colors = _mm_add_epi32(colors, _mm_set_epi32(arr.b, arr.g, arr.r, 0));
                    count++;
                }
            }
            __m128i averageColor = _mm_div_epi32(colors, _mm_set1_epi32(count));
            unsigned char rgb[4];
            _mm_storeu_epi32(rgb, averageColor);
            Pixeles newColor = {rgb[3], rgb[2], rgb[1]};
            for (int y = i; y < min(i + sizeBlock, height); y++){
                for (int x = j; x < min(j + sizeBlock, width); x++) {
                    Pixeles arr = image.at<Pixeles>(y,x) = newColor;
                }
            }
        }
    }
}

void vectorizationContrast(Mat &image){
    double factor = (259.0 * (-100.0 + 255.0)) / (255.0 * (259.0 - (-100.0)));

    for (int y = 0; y < image.rows; y++){
        for (int x = 0; x < image.cols; x++){
            __m128i colors = _mm_setr_epi32(image.ptr<int>(y,x)[0],image.ptr<int>(y,x)[1],image.ptr<int>(y,x)[2],image.ptr<int>(y,x)[3]);
            colors = _mm_cvtepu8_epi32(colors);
            __m256d pixelData = _mm256_cvtepi32_pd(colors);
            pixelData = _mm256_div_pd(pixelData, _mm256_set1_pd(255.0));
            pixelData = _mm256_sub_pd(pixelData, _mm256_set1_pd(0.5));
            pixelData = _mm256_mul_pd(pixelData, _mm256_set1_pd(factor));
            pixelData = _mm256_add_pd(pixelData, _mm256_set1_pd(0.5));
            pixelData = _mm256_mul_pd(pixelData, _mm256_set1_pd(255.0));

            double values[4];
            _mm256_storeu_pd(values, pixelData);
            image.at<Vec3b>(y,x) = {static_cast<unsigned char>(values[0]),
                                    static_cast<unsigned char>(values[1]),
                                    static_cast<unsigned char>(values[2])}
        }
    }
}



void init(){
    int photoNumber;
    int typeProcess;

    cout << "choose the photo:" << endl << "1. 300x300, " << "2. 400x400, " << "3. 500x500, " << "4. 600x600, " <<
         "5. 950x950, " << "6. 2400x2400" << endl;
    cin >> photoNumber;
    cout << endl;
    cout << "choose the type process:" << endl << "1: consistent " << "2: parallel " << "3: vectorization "<< endl;
    cin >> typeProcess;

    if (photoNumber == 1) {
        Mat image = imread("/Users/rodionrybko/CLionProjects/untitled1/parallel/img/300x300.png");
        if (typeProcess == 1) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;
//
            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        if (typeProcess == 2) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaicOmp(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrastOmp(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        else{
            auto startMosaic = chrono::high_resolution_clock::now();
            vectorizationMosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            vectorizationContrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
    }
    if (photoNumber == 2) {
        Mat image = imread("/Users/rodionrybko/CLionProjects/untitled1/parallel/img/400x400.png");
        if (typeProcess == 1) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        if (typeProcess == 2) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaicOmp(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrastOmp(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        else{
            auto startMosaic = chrono::high_resolution_clock::now();
            vectorizationMosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            vectorizationContrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
    }
    if (photoNumber == 3) {
        Mat image = imread("/Users/rodionrybko/CLionProjects/untitled1/parallel/img/500x500.png");
        if (typeProcess == 1) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        if (typeProcess == 2) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaicOmp(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrastOmp(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        else{
            auto startMosaic = chrono::high_resolution_clock::now();
            vectorizationMosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            vectorizationContrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
    }
    if (photoNumber == 4) {
        Mat image = imread("/Users/rodionrybko/CLionProjects/untitled1/parallel/img/600x600.png");
        if (typeProcess == 1) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        if (typeProcess == 2) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaicOmp(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrastOmp(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        else{
            auto startMosaic = chrono::high_resolution_clock::now();
            vectorizationMosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            vectorizationContrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
    }
    if (photoNumber == 5) {
        Mat image = imread("/Users/rodionrybko/CLionProjects/untitled1/parallel/img/950x950.png");
        if (typeProcess == 1) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        if (typeProcess == 2) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaicOmp(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrastOmp(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        else{
            auto startMosaic = chrono::high_resolution_clock::now();
            vectorizationMosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            vectorizationContrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
    }
    if (photoNumber == 6) {
        Mat image = imread("/Users/rodionrybko/CLionProjects/untitled1/parallel/img/2400x2400.png");
        if (typeProcess == 1) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        if (typeProcess == 2) {
            auto startMosaic = chrono::high_resolution_clock::now();
            mosaicOmp(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            contrastOmp(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
        else{
            auto startMosaic = chrono::high_resolution_clock::now();
            vectorizationMosaic(image);
            auto endMosaic = chrono::high_resolution_clock::now();
            auto durationMosaic = chrono::duration_cast<chrono::microseconds>(endMosaic - startMosaic).count();
            cout << "duration time mosaic" << " " << durationMosaic << " " << "microsec." << endl;

            auto startContrast = chrono::high_resolution_clock::now();
            vectorizationContrast(image);
            auto endContrast = chrono::high_resolution_clock::now();
            auto durationContrast = chrono::duration_cast<chrono::microseconds>(
                    endContrast - startContrast).count();
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec." << endl;

            imwrite("/Users/rodionrybko/CLionProjects/untitled1/parallel/result.png", image);

            imshow("PythonBlyaaaaaaad", image);

            waitKey(0);
        }
    }
}

int main(){
    init();
    return 0;
}