#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    Mat image = imread("/Users/rodionrybko/CLionProjects/untitled1/img/500x500.png");

    int blockSize = 7; // размер блока

    for (int y = 0; y < image.rows; y += blockSize) {
        for (int x = 0; x < image.cols; x += blockSize) {
            Rect roi(x, y, blockSize, blockSize);
            Mat block = image(roi);

            Scalar avgColor = mean(block);
            block.setTo(avgColor); // замена блока на средний цвет
        }
    }

    imshow("python_blyad", image);
    waitKey(0);
    return 0;
}