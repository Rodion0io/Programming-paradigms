#include "iostream"

using namespace std;

int* copy(int* array, int lenght, int index = 0){
    int* copyArray = new int[lenght];
    return index == lenght ? copyArray : (copyArray = copy(array, lenght, index + 1), copyArray[index] = array[index] ,copyArray);
}

int* creatArray(int* array, int lenght, int index = 0){
    int* copyArray = copy(array, lenght);
    return index != lenght ? (((cin >> copyArray[index])), creatArray(copyArray, lenght, index + 1)) : copyArray;
}

int* checkToEven(int* array, int lenght, function<bool(int)> check, int index = 0){
    int* copyArray = copy(array, lenght);
    return index != lenght ? (!check(copyArray[index]) ? (copyArray[index] = 1), checkToEven(copyArray, lenght, check, index + 1) : checkToEven(copyArray, lenght, check, index + 1)) : copyArray;
}

int* multiplyThree(int* array, int lenght, function<int(int)> multiply, int index = 0){
    int* copyArray = copy(array, lenght);
    return index != lenght ? (copyArray[index] != 1 ? ((copyArray[index] = multiply(copyArray[index])), multiplyThree(copyArray, lenght, multiply, index + 1)) : multiplyThree(copyArray, lenght, multiply, index + 1)) : copyArray;
}

void printArray(int* array, int lenght, int index = 0){
    int* copyArray = copy(array, lenght);
    copyArray[index] != 1 ? cout << copyArray[index] << " " : cout << "";
    index != lenght - 1 ? printArray(copyArray, lenght, index + 1) : void();
}

int main(){
    printArray(multiplyThree(checkToEven(creatArray(new int[3], 3), 3, [](int x){return x % 2 == 0;}), 3, [](int x){return x * 3;}), 3);
    return 0;
}


