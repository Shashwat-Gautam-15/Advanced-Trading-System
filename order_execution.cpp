// order_execution.cpp
#include <queue>
#include <mutex>
#include <atomic>
#include <chrono>
#include <pybind11/pybind11.h>
//#include "C:\Users\shash\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\include\pybind11\pybind11.h"

namespace py = pybind11;

class OrderExecutionEngine {
private:
    std::queue<std::tuple<char, double, double>> orderQueue;
    std::mutex queueMutex;
    std::atomic<bool> running{false};
    
public:
    void start() {
        running = true;
        while (running) {
            processOrders();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
    
    void stop() {
        running = false;
    }
    
    void addOrder(char side, double price, double quantity) {
        std::lock_guard<std::mutex> lock(queueMutex);
        orderQueue.emplace(side, price, quantity);
    }
    
    void processOrders() {
        std::lock_guard<std::mutex> lock(queueMutex);
        while (!orderQueue.empty()) {
            auto order = orderQueue.front();
            orderQueue.pop();
            
            char side = std::get<0>(order);
            double price = std::get<1>(order);
            double quantity = std::get<2>(order);
            
            // In a real implementation, this would send to exchange API
            // For now, just print
            printf("Executing %s order: %.2f @ %.2f\n", 
                side == 'B' ? "BUY" : "SELL", quantity, price);
        }
    }
    
    size_t queueSize() const {
        return orderQueue.size();
    }
};

PYBIND11_MODULE(order_execution, m) {
    py::class_<OrderExecutionEngine>(m, "OrderExecutionEngine")
        .def(py::init<>())
        .def("start", &OrderExecutionEngine::start)
        .def("stop", &OrderExecutionEngine::stop)
        .def("add_order", &OrderExecutionEngine::addOrder)
        .def("queue_size", &OrderExecutionEngine::queueSize);
}