#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <string>
#include <stdarg.h>
#include <memory>

/**
 * @brief Logging macro interface.
 *
 * @details
 *  These macros provide a convenient front-end API for logging with
 *  different severity levels. Internally, they forward to:
 *
 *      Logger::__log_info(...)
 *
 *  Example:
 *
 *      LOG("Initialization done");
 *      LOGE("Failed to allocate memory: %d", err_code);
 */
#define LOGF(...) logger::Logger::__log_info(logger::Level::FATAL, __VA_ARGS__)
#define LOGE(...) logger::Logger::__log_info(logger::Level::ERROR, __VA_ARGS__)
#define LOGW(...) logger::Logger::__log_info(logger::Level::WARN,  __VA_ARGS__)
#define LOG(...)  logger::Logger::__log_info(logger::Level::INFO,  __VA_ARGS__)
#define LOGV(...) logger::Logger::__log_info(logger::Level::VERB,  __VA_ARGS__)
#define LOGD(...) logger::Logger::__log_info(logger::Level::DEBUG, __VA_ARGS__)

/**
 * @brief ANSI color definitions for terminal output.
 *
 * @details
 *  These escape sequences are used to colorize log messages
 *  for better readability in terminal environments.
 *
 *  Note:
 *      - Supported in most UNIX terminals
 *      - May not work on Windows CMD without ANSI support
 */
#define DGREEN    "\033[1;36m"
#define BLUE      "\033[1;34m"
#define PURPLE    "\033[1;35m"
#define GREEN     "\033[1;32m"
#define YELLOW    "\033[1;33m"
#define RED       "\033[1;31m"
#define CLEAR     "\033[0m"

namespace logger {

/**
 * @brief Log severity levels.
 *
 * @details
 *  Defines a hierarchical logging system:
 *
 *      FATAL < ERROR < WARN < INFO < VERB < DEBUG
 *
 *  Filtering rule:
 *
 *      Only logs with:
 *
 *          level <= current_level
 *
 *      will be printed.
 *
 *  Example:
 *
 *      current_level = INFO
 *      → INFO, WARN, ERROR, FATAL will be printed
 *      → DEBUG, VERB will be ignored
 */
enum class Level : int32_t
{
    FATAL = 0,
    ERROR = 1,
    WARN  = 2,
    INFO  = 3,
    VERB  = 4,
    DEBUG = 5
};

/**
 * @brief Lightweight logging utility (static-based design).
 *
 * @details
 *  This class implements a minimal logging system with:
 *
 *  - Severity filtering
 *  - Colorized output
 *  - printf-style formatting (variadic arguments)
 *
 *
 *  Typical usage:
 *
 *      Logger logger(Level::DEBUG);
 *      LOG("System initialized");
 *
 */
class Logger
{

public:

    /**
     * @brief Default constructor.
     *
     * @details
     *  Initializes logger with default level:
     *
     *      Level::INFO
     */
    Logger();

    /**
     * @brief Construct logger with specified level.
     *
     * @param level Logging level threshold
     *
     * @details
     *  Sets global logging level:
     *
     *      m_level = level
     */
    Logger(Level level);

    /**
     * @brief Destructor.
     */
    ~Logger();
    
public:

    /**
     * @brief Log a preformatted message.
     *
     * @param level Log severity
     * @param msg   Null-terminated message string
     *
     * @details
     *  Wrapper function:
     *
     *      Calls __log_info internally if DEBUG enabled
     *
     *  Note:
     *      This API is rarely used directly (macros preferred)
     */
    void log(Level level, char* msg);

    /**
     * @brief Core logging function (printf-style).
     *
     * @param level  Log severity
     * @param format Format string (printf-style)
     * @param ...    Variable arguments
     *
     * @details
     *  Workflow:
     *
     *  1. Prefix generation:
     *
     *      [DEBUG] / [INFO] / [ERROR] ...
     *
     *  2. Format expansion:
     *
     *      vsnprintf(...)
     *
     *  3. Severity filtering:
     *
     *      if (level <= m_level)
     *          → print
     *
     *  4. Error handling:
     *
     *      if level <= ERROR:
     *          → flush + terminate
     *
     *  Important:
     *      - Uses fixed-size buffer (1000 bytes)
     *      - Potential truncation if message too long
     */
    static void __log_info(Level level, const char* format, ...);

private:

    /**
     * @brief Global logging level.
     *
     * @details
     *  Shared across all Logger instances.
     *
     *  Controls verbosity of output.
     */
    static Level m_level;

};

};

#endif