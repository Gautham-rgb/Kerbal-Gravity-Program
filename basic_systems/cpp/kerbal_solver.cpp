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

double get_ut_secs(int year, int day, int hour, int minutes, int seconds, bool ker_time = false) {
    if (ker_time) {
        double elapsed_years = year - 1;
        double elapsed_day = day - 1;
        double total_secs = (elapsed_years * constants::ker_year_sec) + (elapsed_day * constants::ker_day_sec) + 
        (hour * constants::ker_hour_min * constants::ker_min_sec) + (minutes * constants::ker_min_sec) + seconds;

        return total_secs;
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
                           
        return static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(time_period.time_since_epoch()).count());
    }
}