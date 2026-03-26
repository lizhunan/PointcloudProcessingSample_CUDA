#include "logger.h"

namespace logger {

Level Logger::m_level = Level::INFO;

Logger::Logger(Level level)
{
    m_level = level;
}

void Logger::log(Level level, char* msg)
{   
    // Only active in DEBUG mode.
    // Acts as a conditional forwarding wrapper.
    if (m_level >= Level::DEBUG) __log_info(level, "%s", msg);
}

void Logger::__log_info(Level level, const char* format, ...) {
    char msg[1024];     // Fixed-size buffer for formatted message.
    va_list args;
    va_start(args, format);
    int n = 0;
    
    // Prefix formatting with color.
    switch (level) {
        case Level::DEBUG: n += snprintf(msg + n, sizeof(msg) - n, DGREEN "[DEBUG]" CLEAR); break;
        case Level::VERB:  n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[VERB]" CLEAR); break;
        case Level::INFO:  n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[INFO]" CLEAR); break;
        case Level::WARN:  n += snprintf(msg + n, sizeof(msg) - n, BLUE "[WARN]" CLEAR); break;
        case Level::ERROR: n += snprintf(msg + n, sizeof(msg) - n, RED "[ERROR]" CLEAR); break;
        default:           n += snprintf(msg + n, sizeof(msg) - n, RED "[FATAL]" CLEAR); break;
    }

    // Append formatted message body.
    n += vsnprintf(msg + n, sizeof(msg) - n, format, args);

    va_end(args);

    // Log filtering based on severity level.
    if (level <= m_level) 
        fprintf(stdout, "%s\n", msg);

    // Fatal/Error handling:
    //  - Flush output
    //  - Terminate program
    if (level <= Level::ERROR) {
        fflush(stdout);
        exit(0);
    }
}

Logger::~Logger(){}

};