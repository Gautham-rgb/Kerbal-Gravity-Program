#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <algorithm>
#include "nlohmann/json.hpp"
#include <chrono>
#include <string>

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
        double hour_secs = constants::ker_min_sec * constants::ker_hour_min;
        
        return (elapsed_years * constants::ker_year_sec) + 
               (elapsed_days * constants::ker_day_sec) + 
               (hour * hour_secs) + 
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
};

inline double solve_eccen_anomaly(double mean_anomaly, double eccen) {
    double M = std::fmod(mean_anomaly, 2.0 * constants::pi);
    if (M > constants::pi) {M -= 2.0 * constants::pi;};
    if (M < -constants::pi) {M += 2.0 * constants::pi;};

    double E = M;
    double epsilon = 1e-12;
    int max_iter = 100;

    for (int i = 0; i < max_iter; ++i) {
        double delta_E = (E - eccen * std::sin(E) - M) / (1.0 - eccen * std::cos(E));
        E -= delta_E;
        if (std::abs(delta_E) < epsilon) {
            break;
        }
    }
    return E;
    }

struct Orbit {
    double semi_major_axis;
    double eccen;
    double arg_periapsis;
    double inclination;
    double lon_of_asc;
    double mean_anomaly_at_t0;
    Body* parent;

    Orbit(double a, double e, double arg_p, double inclination, double lon_of_asc, double MA_at_t0, Body* parent = nullptr) : 
        semi_major_axis(a), eccen(e), arg_periapsis(arg_p), inclination(inclination), lon_of_asc(lon_of_asc), mean_anomaly_at_t0(MA_at_t0), parent(parent) {}

    Orbit() : semi_major_axis(0), eccen(0), arg_periapsis(0), inclination(0), lon_of_asc(0), mean_anomaly_at_t0(0) {}
};

class Body {
public:
    std::string name; 
    double mu;
    double radius;
    Orbit orbit;
    double atm_height; 

    std::vector<Body*> moons;

    Body(std::string name, double mu, double radius, Orbit orbit, double atm_height = 0.0) :
        mu(mu), radius(radius), orbit(orbit), atm_height(atm_height) {this->name = name;}
    
    Body(std::string name, double mu, double radius, double atm_height = 0.0) :
    name(name), mu(mu), radius(radius), orbit(Orbit()), atm_height(atm_height) {}

    virtual ~Body() = default;

    Eigen::Vector3d get_pos_at_ut(double ut) {
        if (orbit.parent == nullptr) {
            return Eigen::Vector3d::Zero();
        };

        double n = std::sqrt(orbit.parent->mu / std::pow(orbit.semi_major_axis, 3.0));
        double mean_anomaly = orbit.mean_anomaly_at_t0 + n * ut;

        double E = solve_eccen_anomaly(mean_anomaly, orbit.eccen);

        double cosE = std::cos(E);
        double sinE = std::sin(E);
        double x_perifocal = orbit.semi_major_axis * (cosE - orbit.eccen);
        double y_perifocal = orbit.semi_major_axis * (1.0 - orbit.eccen * orbit.eccen);

        Eigen::Vector3d pos_perifocal(x_perifocal, y_perifocal, 0.0);

        double cos_lan = std::cos(orbit.lon_of_asc);
        double sin_lan = std::sin(orbit.lon_of_asc);
        double cos_inc = std::cos(orbit.inclination);
        double sin_inc = std::sin(orbit.inclination);
        double cos_ap = std::cos(orbit.arg_periapsis);
        double sin_ap = std::sin(orbit.arg_periapsis);

        Eigen::Matrix3d R_lan;
        R_lan << cos_lan, -sin_lan, 0.0, sin_lan, cos_lan, 0.0, 0.0, 0.0, 1.0;

        Eigen::Matrix3d R_inc;
        R_inc << 1.0, 0.0, 0.0, 0.0, cos_inc, -sin_inc, 0.0, sin_inc, cos_inc;

        Eigen::Matrix3d R_ap;
        R_ap << cos_ap, -sin_ap, 0.0, sin_ap, cos_ap, 0.0, 0.0, 0.0, 1.0;

        Eigen::Matrix3d R = R_lan * R_inc * R_ap;
        return R * pos_perifocal;
    }

    Eigen::Vector3d get_absolute_pos_at_ut(double ut) {
        if (orbit.parent == nullptr) {
            return Eigen::Vector3d::Zero();
        }

        return this->get_pos_at_ut(ut) + orbit.parent->get_absolute_pos_at_ut(ut);
    }
};

int main() {
    Body sun("Sun", 1.1723328e18, 261600000);
    Orbit kerbin_orbit(13599840256.0, 0.01, 0.0, 0.174533, 0.785398, 0.0, &sun);
    Body kerbin("Kerbin", 3.5316000e12, 600000, kerbin_orbit);
    Eigen::Vector3d pos = kerbin.get_pos_at_ut(10000.0);
    std::cout << pos.x() << " " << pos.y() << " " << pos.z() << std::endl;
    return 0;
}