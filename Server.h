#pragma once

#include <libwebsockets.h>
#include <thread>
#include <vector>

class Server {

public:
    Server(int port = 5005);
    ~Server();

    void start();

    void stop();

    void send(std::vector<std::vector<float>> data);

private:
    lws_context_creation_info info;
    lws_context* context;
    std::thread t;
    bool running{false};

    std::vector<std::vector<float>> trackingData;

};