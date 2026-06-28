#include <stdio.h>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <algorithm>
#include "nlohmann/json.hpp"
#include <chrono>

using namespace std;

namespace constants {
    const double pi = std::acos(-1);
    const double ker_year_day = 426;
    const double ker_day_hour = 6;
    const double ker_hour_min = 60;
    const double ker_min_sec = 60;
    const double ker_year_sec = 9201600.0;
    const double ker_day_sec = 21600.0;
};

/**
 * @brief Calculates the Universal Time (UT) timestamp in seconds.
 * 
 * @details Computes total elapsed seconds based on calendar parameters.
 * Supports both C++20 Earth UTC (Unix Epoch) and custom Kerbal Space 
 * Program (KSP) planetary time tracking configurations.
 * 
 * @param year The target calendar year (1-indexed).
 * @param day The day of the year (1-indexed, e.g., 1 to 365/426).
 * @param hour The hour of the day.
 * @param minutes The minutes past the hour.
 * @param seconds The seconds past the minute.
 * @param ker_time True when calculating Kerbal time, False when calculating real time.
 * 
 * @return double High-precision floating-point epoch timestamp in seconds.
 * 
 * @note This function is declared inline to remove execution overhead in orbital propagation loops.
 * @see constants
 */
inline double get_ut_secs(int year, int day, int hour, int minutes, int seconds, bool ker_time = false) {
    if (ker_time) {
        double elapsed_years = year - 1;
        double elapsed_days = day - 1;
        
        return (elapsed_years * constants::ker_year_sec) + 
               (elapsed_days * constants::ker_day_sec) + 
               (hour * 3600.0) + 
               (minutes * 60.0) + 
               seconds;
    } else {
        std::chrono::year_month_day base_ymd{
            std::chrono::year{year},
            std::chrono::month{1},
            std::chrono::day{1}
        };
        
        auto time_period = std::chrono::sys_days{base_ymd} + 
                           std::chrono::days{day - 1} + 
                           std::chrono::hours{hour} + 
                           std::chrono::minutes{minutes} + 
                           std::chrono::seconds{seconds};
                           
        return static_cast<double>(time_period.time_since_epoch().count());
    }
}