#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <algorithm>
#include "nlohmann/json.hpp"
#include <chrono>
#include <string>
#include <vector>
#include <array>
#include <limits>
#include <stdexcept>

using namespace std;

namespace constants {
    const double pi = std::acos(-1);
    const double ker_year_day = 426;
    const double ker_day_hour = 6;
    const double ker_hour_min = 60;
    const double ker_min_sec = 60;
    const double ker_year_sec = 9201600.0;
    const double ker_day_sec = 21600.0;
}

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
}

inline double solve_anomaly(double mean_anomaly, double eccen) {
    double epsilon = 1e-12;
    int max_iter = 100;

    if (eccen < 1.0) {
        double M = std::fmod(mean_anomaly, 2.0 * constants::pi);
        if (M > constants::pi) M -= 2.0 * constants::pi;
        if (M < -constants::pi) M += 2.0 * constants::pi;
        
        double E = M;
        for (int i = 0; i < max_iter; ++i) {
            double delta_E = (E - eccen * std::sin(E) - M) / (1.0 - eccen * std::cos(E));
            E -= delta_E;
            if (std::abs(delta_E) < epsilon) break;
        }
        return E;
    } else if (eccen > 1.0) {
        double M = mean_anomaly;
        double H = 0.0;

        if (std::abs(M) > 0.1) H = std::asinh(M / eccen);

        for (int i = 0; i < max_iter; ++i) {
            double delta_H = (eccen * std::sinh(H) - H - M) / (eccen * std::cosh(H) - 1.0);
            H -= delta_H;
            if (std::abs(delta_H) < epsilon) break;
        }
        return H;
    } else {
        double e_eff = 0.9999999999;
        double M = std::fmod(mean_anomaly, 2.0 * constants::pi);
        if (M > constants::pi) M -= 2.0 * constants::pi;
        if (M < -constants::pi) M += 2.0 * constants::pi;
        
        double E = M;
        for (int i = 0; i < max_iter; ++i) {
            double delta_E = (E - e_eff * std::sin(E) - M) / (1.0 - e_eff * std::cos(E));
            E -= delta_E;
            if (std::abs(delta_E) < epsilon) break;
        }
        return E;
    }
}

class Body;

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

    Orbit() : semi_major_axis(0), eccen(0), arg_periapsis(0), inclination(0), lon_of_asc(0), mean_anomaly_at_t0(0), parent(nullptr) {}
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
        mu(mu), radius(radius), orbit(orbit), atm_height(atm_height) { this->name = name; }
    
    Body(std::string name, double mu, double radius, double atm_height = 0.0) :
        name(name), mu(mu), radius(radius), orbit(Orbit()), atm_height(atm_height) {}

    virtual ~Body() = default;

    Eigen::Vector3d get_pos_at_ut(double ut) {
        if (orbit.parent == nullptr) return Eigen::Vector3d::Zero();

        double n = std::sqrt(orbit.parent->mu / std::pow(std::abs(orbit.semi_major_axis), 3.0));
        double mean_anomaly = orbit.mean_anomaly_at_t0 + n * ut;
        double anomaly = solve_anomaly(mean_anomaly, orbit.eccen);

        double x_perifocal, y_perifocal;
        if (orbit.eccen < 1.0) {
            double cosA = std::cos(anomaly);
            double sinA = std::sin(anomaly);
            x_perifocal = orbit.semi_major_axis * (cosA - orbit.eccen);
            y_perifocal = orbit.semi_major_axis * std::sqrt(1.0 - orbit.eccen * orbit.eccen) * sinA;
        } else if (orbit.eccen > 1.0) {
            double a_abs = std::abs(orbit.semi_major_axis);
            double coshA = std::cosh(anomaly);
            double sinhA = std::sinh(anomaly);
            x_perifocal = a_abs * (orbit.eccen - coshA);
            y_perifocal = a_abs * std::sqrt(orbit.eccen * orbit.eccen - 1.0) * sinhA;
        } else {
            double cosA = std::cos(anomaly);
            double sinA = std::sin(anomaly);
            x_perifocal = orbit.semi_major_axis * (cosA - orbit.eccen);
            y_perifocal = orbit.semi_major_axis * sinA;
        }

        Eigen::Vector3d pos_perifocal(x_perifocal, y_perifocal, 0.0);

        double cos_lan = std::cos(orbit.lon_of_asc);
        double sin_lan = std::sin(orbit.lon_of_asc);
        double cos_inc = std::cos(orbit.inclination);
        double sin_inc = std::sin(orbit.inclination);
        double cos_ap = std::cos(orbit.arg_periapsis);
        double sin_ap = std::sin(orbit.arg_periapsis);

        Eigen::Matrix3d R_lan;
        R_lan << cos_lan, -sin_lan, 0.0, 
                 sin_lan,  cos_lan, 0.0, 
                 0.0, 0.0, 1.0;

        Eigen::Matrix3d R_inc;
        R_inc << 1.0, 0.0, 0.0, 
                 0.0, cos_inc, -sin_inc, 
                 0.0, sin_inc, cos_inc;

        Eigen::Matrix3d R_ap;
        R_ap << cos_ap, -sin_ap, 0.0, 
                sin_ap,  cos_ap, 0.0, 
                0.0, 0.0, 1.0;

        return (R_lan * R_inc * R_ap) * pos_perifocal;
    }

    Eigen::Vector3d get_vel_at_ut(double ut) {
        if (orbit.parent == nullptr) return Eigen::Vector3d::Zero();

        double n = std::sqrt(orbit.parent->mu / std::pow(std::abs(orbit.semi_major_axis), 3.0));
        double mean_anomaly = orbit.mean_anomaly_at_t0 + n * ut;
        double anomaly = solve_anomaly(mean_anomaly, orbit.eccen);

        double vx_perifocal, vy_perifocal;
        if (orbit.eccen < 1.0) {
            double cosA = std::cos(anomaly);
            double sinA = std::sin(anomaly);
            double v_coef = std::sqrt(orbit.parent->mu * orbit.semi_major_axis) / (orbit.semi_major_axis * (1.0 - orbit.eccen * cosA));
            vx_perifocal = -v_coef * sinA;
            vy_perifocal = v_coef * std::sqrt(1.0 - orbit.eccen * orbit.eccen) * cosA;
        } else if (orbit.eccen > 1.0) {
            double a_abs = std::abs(orbit.semi_major_axis);
            double coshA = std::cosh(anomaly);
            double sinhA = std::sinh(anomaly);
            double v_coef = std::sqrt(orbit.parent->mu * a_abs) / (a_abs * (orbit.eccen * coshA - 1.0));
            vx_perifocal = -v_coef * sinhA;
            vy_perifocal = v_coef * std::sqrt(orbit.eccen * orbit.eccen - 1.0) * coshA;
        } else {
            double cosA = std::cos(anomaly);
            double sinA = std::sin(anomaly);
            double v_coef = std::sqrt(orbit.parent->mu / orbit.semi_major_axis);
            vx_perifocal = -v_coef * sinA;
            vy_perifocal = v_coef * cosA;
        }

        Eigen::Vector3d vel_perifocal(vx_perifocal, vy_perifocal, 0.0);

        double cos_lan = std::cos(orbit.lon_of_asc);
        double sin_lan = std::sin(orbit.lon_of_asc);
        double cos_inc = std::cos(orbit.inclination);
        double sin_inc = std::sin(orbit.inclination);
        double cos_ap = std::cos(orbit.arg_periapsis);
        double sin_ap = std::sin(orbit.arg_periapsis);

        Eigen::Matrix3d R_lan;
        R_lan << cos_lan, -sin_lan, 0.0, 
                 sin_lan,  cos_lan, 0.0, 
                 0.0, 0.0, 1.0;

        Eigen::Matrix3d R_inc;
        R_inc << 1.0, 0.0, 0.0, 
                 0.0, cos_inc, -sin_inc, 
                 0.0, sin_inc, cos_inc;

        Eigen::Matrix3d R_ap;
        R_ap << cos_ap, -sin_ap, 0.0, 
                sin_ap,  cos_ap, 0.0, 
                0.0, 0.0, 1.0;

        return (R_lan * R_inc * R_ap) * vel_perifocal;
    }

    Eigen::Vector3d get_absolute_pos_at_ut(double ut) {
        if (orbit.parent == nullptr) return Eigen::Vector3d::Zero();
        return this->get_pos_at_ut(ut) + orbit.parent->get_absolute_pos_at_ut(ut);
    }

    Eigen::Vector3d get_absolute_vel_at_ut(double ut) {
        if (orbit.parent == nullptr) return Eigen::Vector3d::Zero();
        return this->get_vel_at_ut(ut) + orbit.parent->get_absolute_vel_at_ut(ut);
    }
};

class Barycenter: public Body {
public:
    Body* a;
    Body* b;

    Barycenter(std::string sys_name, Body* a, Body* b, double total_sma, double ecc, double inc, double lan, double arg_p) :
        Body(sys_name, a->mu + b->mu, 0.0) 
    {
        this->a = a;
        this->b = b;
        double a_sma = total_sma * (b->mu / this->mu);
        double b_sma = total_sma * (a->mu / this->mu);

        a->orbit = Orbit(a_sma, ecc, arg_p, inc, lan, 0.0, this);
        b->orbit = Orbit(b_sma, ecc, arg_p, inc, lan, constants::pi, this);

        this->moons.push_back(a);
        this->moons.push_back(b); 
    }
};

class LambertSolver {
    double mu;
    bool solved;

    static std::array<double, 2> stumpff(double z) {
        if (z > 1e-8) {
            double root_z = std::sqrt(z);
            return {
                (1.0 - std::cos(root_z)) / z,
                (root_z - std::sin(root_z)) / std::pow(root_z, 3.0)
            };
        }
        if (z < -1e-8) {
            double root_neg_z = std::sqrt(-z);
            return {
                (std::cosh(root_neg_z) - 1.0) / -z,
                (std::sinh(root_neg_z) - root_neg_z) / std::pow(root_neg_z, 3.0)
            };
        }
        return {0.5, 1.0 / 6.0};
    }

    double tof_for_z(double z, double r1_norm, double r2_norm, double A) const {
        auto [C, S] = stumpff(z);
        if (C <= 0.0) return std::numeric_limits<double>::infinity();

        double y = r1_norm + r2_norm + A * (z * S - 1.0) / std::sqrt(C);
        if (y < 0.0) return std::numeric_limits<double>::infinity();

        double x = std::sqrt(y / C);
        return (std::pow(x, 3.0) * S + A * std::sqrt(y)) / std::sqrt(mu);
    }

    double solve_universal_z(double r1_norm, double r2_norm, double A, double tof) const {
        double z = 0.0;
        for (int i = 0; i < 100; ++i) {
            double trial_tof = tof_for_z(z, r1_norm, r2_norm, A);
            if (!std::isfinite(trial_tof)) {
                z += 0.1;
                continue;
            }
            if (std::abs(trial_tof - tof) < 1e-8) return z;

            double step_size = std::max(1e-5, std::abs(z) * 1e-5);
            double high_tof = tof_for_z(z + step_size, r1_norm, r2_norm, A);
            double low_tof = tof_for_z(z - step_size, r1_norm, r2_norm, A);
            double derivative = (high_tof - low_tof) / (2.0 * step_size);
            if (!std::isfinite(derivative) || std::abs(derivative) < 1e-12) {
                z += (trial_tof < tof) ? 0.1 : -0.1;
                continue;
            }

            double step = std::clamp((trial_tof - tof) / derivative, -1.0, 1.0);
            z -= step;
        }

        throw std::runtime_error("Lambert solve did not converge.");
    }
    
public:
    LambertSolver(double root_mu) : mu(root_mu), solved(false) {}

    double get_mu() const { return mu; }

    std::array<Eigen::Vector3d, 2> solve(Eigen::Vector3d& r1, Eigen::Vector3d& r2, double tof, bool long_way = false) {
        double r1_norm = r1.norm();
        double r2_norm = r2.norm();
        if (r1_norm == 0.0 || r2_norm == 0.0) {
            throw std::invalid_argument("Lambert solve requires non-zero position vectors.");
        }
        if (tof <= 0.0) {
            throw std::invalid_argument("Lambert solve requires a positive time of flight.");
        }

        Eigen::Vector3d cross_prod = r1.cross(r2);        
        double cos_nu = std::clamp(r1.dot(r2) / (r1_norm * r2_norm), -1.0, 1.0);
        double sin_nu_mag = cross_prod.norm() / (r1_norm * r2_norm);
        double sin_nu = long_way ? -sin_nu_mag : sin_nu_mag;
        if (std::abs(sin_nu) < 1e-12) {
            throw std::invalid_argument("Lambert solve is undefined for collinear transfer positions.");
        }

        double A = sin_nu * std::sqrt((r1_norm * r2_norm) / (1.0 - cos_nu));
        if (std::abs(A) < 1e-12) {
            throw std::invalid_argument("Lambert solve has degenerate transfer geometry.");
        }

        double z = solve_universal_z(r1_norm, r2_norm, A, tof);
        auto [C, S] = stumpff(z);
        double y = r1_norm + r2_norm + A * (z * S - 1.0) / std::sqrt(C);
        if (y < 0.0) {
            throw std::runtime_error("Lambert solve failed to find a physical transfer.");
        }

        double f = 1.0 - y / r1_norm;
        double g = A * std::sqrt(y / mu);
        double gdot = 1.0 - y / r2_norm;
        if (std::abs(g) < 1e-12) {
            throw std::runtime_error("Lambert solve produced a singular transfer.");
        }
        
        Eigen::Vector3d v1 = (r2 - f * r1) / g;
        Eigen::Vector3d v2 = (gdot * r2 - r1) / g;
        
        this->solved = true;
        return {v1, v2};
    }
};

struct ManeuverNode {
    Eigen::Vector3d delta_v_vector;
    Eigen::Vector3d prograde;
    Eigen::Vector3d normal;
    Eigen::Vector3d radial;
    double total_mag;
};

int main() {
    Body sun("Sun", 1.1723328e18, 261600000);
    
    Orbit kerbin_orbit(13599840256.0, 0.01, 0.0, 0.174533, 0.785398, 0.0, &sun);
    Body kerbin("Kerbin", 3.5316000e12, 600000, kerbin_orbit);
    
    Orbit duna_orbit(20726155264.0, 0.051, 0.0, 0.010472, 2.364921, constants::pi, &sun);
    Body duna("Duna", 3.0136321e11, 320000, duna_orbit);

    double start_ut = 14148000.0;
    
    Eigen::Vector3d r_kerbin = kerbin.get_pos_at_ut(start_ut);
    Eigen::Vector3d v_kerbin = kerbin.get_vel_at_ut(start_ut);
    Eigen::Vector3d r_duna = duna.get_pos_at_ut(start_ut);

    std::cout << "--- Kerbal Solver C++ Test ---" << std::endl;
    std::cout << "Kerbin Position at " << start_ut << "s:" << r_kerbin.transpose() << std::endl;
    std::cout << "Kerbin Velocity at " << start_ut << "s:" << v_kerbin.transpose() << std::endl;
    std::cout << "Duna Position at " << start_ut << "s:" << r_duna.transpose() << std::endl;

    return 0;
}
