#include "iostream"
#include "opencv2/opencv.hpp"
#include "chrono"

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

    double factor = (259.0 * (-100.0 + 255.0)) / (255.0 * (259.0 - (-100.0)));

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b intensity = image.at<Vec3b>(y, x);

            for (int c = 0; c < 3; c++) {
//                int new_intensity = alpha * intensity[c] + beta;
                int new_intensity = factor * (intensity[c] - 128) + 128;
                if (new_intensity < 0) {
                    intensity[c] = 0;
                }
                else if (new_intensity > 255) {
                    intensity[c] = 255;
                }
                else {
                    intensity[c] = new_intensity;
                }
            }

            image.at<Vec3b>(y, x) = intensity;
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

    double factor = (259.0 * (-100.0 + 255.0)) / (255.0 * (259.0 - (-100.0)));

#pragma omp parallel for
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b intensity = image.at<Vec3b>(y, x);

            for (int c = 0; c < 3; c++) {
//                int new_intensity = alpha * intensity[c] + beta;
                int new_intensity = factor * (intensity[c] - 128) + 128;
                if (new_intensity < 0) {
                    intensity[c] = 0;
                }
                else if (new_intensity > 255) {
                    intensity[c] = 255;
                }
                else {
                    intensity[c] = new_intensity;
                }
            }

            image.at<Vec3b>(y, x) = intensity;
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
    cout << "choose the type process:" << endl << "1: consistent " << "2: parallel" << endl;
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
        else {
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
        else {
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
        else {
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
        else {
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
        else {
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
        else {
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
            cout << "duration time contrast" << " " << durationContrast << " " << "microsec.microsec." << endl;

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