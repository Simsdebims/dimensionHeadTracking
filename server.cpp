
#include "server.h"
#include <iostream>

#define EXAMPLE_RX_BUFFER_BYTES (10)

static int callback_http(struct lws* wsi, enum lws_callback_reasons reason, void* user, void* in, size_t len) {
    switch (reason) {
        case LWS_CALLBACK_HTTP:
            lws_serve_http_file(wsi, "websocket.html", "text/html", NULL, 0);
            break;
        default:
            break;
    }

    return 0;
}


struct payload {
    unsigned char data[LWS_SEND_BUFFER_PRE_PADDING + EXAMPLE_RX_BUFFER_BYTES + LWS_SEND_BUFFER_POST_PADDING];
    size_t len;
} received_payload;

static int tracking_callback(struct lws* wsi, enum lws_callback_reasons reason, void* user, void* in, size_t len) {
    switch (reason) {
        case LWS_CALLBACK_RECEIVE:
            memcpy(&received_payload.data[LWS_SEND_BUFFER_PRE_PADDING], in, len);
            received_payload.len = len;
            lws_callback_on_writable_all_protocol(lws_get_context(wsi), lws_get_protocol(wsi));
            break;

        case LWS_CALLBACK_SERVER_WRITEABLE: {
            std::vector<std::vector<float>>* data = (std::vector<std::vector<float>>*) lws_context_user(
                    lws_get_context(wsi));

            std::string result = "\"positions\":[";

            for (size_t i = 0; i < data->size(); ++i) {
                result += "{\"id\":" + std::to_string((int) data->at(i)[0]) + ",\"position\":[";
                result += std::to_string(data->at(i)[1]) + ",";
                result += std::to_string(data->at(i)[2]) + ",";
                result += std::to_string(data->at(i)[3]) + "]}";

                if (i < data->size() - 1) result += ",";
            }

            result += "]";

            unsigned char buf[LWS_SEND_BUFFER_PRE_PADDING + result.size() + LWS_SEND_BUFFER_POST_PADDING];
            unsigned char* p = &buf[LWS_SEND_BUFFER_PRE_PADDING];

            size_t n = sprintf((char*) p, "{ %s }", result.c_str());
            lws_write(wsi, p, n, LWS_WRITE_TEXT);
            break;
        }

        default:
            break;
    }

    return 0;
}

static struct lws_protocols protocols[] =
        {
                /* The first protocol must always be the HTTP handler */
                {
                        "http-only",   /* name */
                        callback_http, /* callback */
                             0,             /* No per session data. */
                                0,             /* max frame size / rx buffer */
                },
                {
                        "tracking-protocol",
                        tracking_callback,
                             0,
                        EXAMPLE_RX_BUFFER_BYTES,
                },
                {NULL, NULL, 0, 0} /* terminator */
        };


Server::Server(int port) {
    memset(&info, 0, sizeof(info));

    info.port = port;
    info.protocols = protocols;
    info.gid = -1;
    info.uid = -1;
    info.user = &trackingData;

    lws_set_log_level(LLL_WARN, NULL);

    context = lws_create_context(&info);
}

Server::~Server() {
    if (running) stop();
}

void Server::start() {
    auto run = [&]() {
        while (running) {
            lws_service(context, /* timeout_ms = */ 1000000);
        }
        lws_context_destroy(context);
    };

    running = true;

    t = std::thread(run);
}

void Server::stop() {
    running = false;
    //t.join();
}

void Server::send(std::vector<std::vector<float> > data) {

    trackingData = data;

    lws_callback_on_writable_all_protocol(context, &protocols[1]);
    lws_service(context, /* timeout_ms = */ 100);
}
