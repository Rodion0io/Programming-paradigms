#include <iostream>
#include <queue>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <csignal>
#include <cstdlib>

using namespace std;

int countAtm;
int capacityAtm;
queue<int> requests;
mutex requestMutex;
mutex atmMutex;
volatile bool stop = false;

void signalHandler(int count) {
    cout << "Received for interruption: " << count << endl;
    stop = true;
}

void expectation() {
    int time = rand() % 5 + 1;
    this_thread::sleep_for(chrono::seconds(time));
    for (int i = time; i >= 1; i--) {
        cout << "There's still time to get out of the queue " << i << " seconds" << endl;
        this_thread::sleep_for(chrono::seconds(1));
    }
}

void depersonalization() {
    while (!stop) {
        atmMutex.lock();
        if (!requests.empty()) {
            int number = requests.front();
            requests.pop();
            atmMutex.unlock();
            cout << "ATM under the number " << number << " is undergoing the procedure" << endl;
            expectation();
            cout << "The ATM under the number " << number << " has left the queue" << endl;
        } else {
            atmMutex.unlock();
        }
    }
}


void generateRequests() {
    int currentRequest = 0;
    while (!stop) {
        this_thread::sleep_for(chrono::seconds(rand() % 5 + 1));
        unique_lock<mutex> lock(requestMutex);
        if (requests.size() < capacityAtm) {
            requests.push(currentRequest + 1);
            cout << "Request " << currentRequest + 1 << " added, current queue size: " << requests.size() << endl;
        }
        lock.unlock();
        currentRequest++;
    }
    cout << "generateQueue is over" << endl;
}

int main() {
    cout << "Enter the count of ATMs: ";
    cin >> countAtm;
    cout << "Enter the ATM capacity: ";
    cin >> capacityAtm;

    signal(SIGINT, signalHandler);

    thread generatorThread(generateRequests);
    vector<thread> atmThreads;

    for (int i = 0; i < countAtm; i++) {
        atmThreads.emplace_back(depersonalization);
    }

    generatorThread.join();
    for (auto& t : atmThreads) {
        t.join();
    }

    return 0;
}
