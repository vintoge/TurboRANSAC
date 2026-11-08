#ifndef __TICKTOCK_H__
#define __TICKTOCK_H__

#include <chrono>
#include <string>
#include <iostream>
#include <unordered_map>

// Simple tick-tock in seconds
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) std::cout << #x " time: " << std::chrono::duration_cast<std::chrono::duration<double>>( \
                std::chrono::steady_clock::now() - bench_##x).count() << " s" << std::endl;


// Simple tick-tock in milliseconds
#define TICK_MS(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK_MS(x) std::cout << #x " time: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>( \
                                 std::chrono::steady_clock::now() - bench_##x).count() << " ms" << std::endl;

                                 
// Accumulated timing in milliseconds
inline std::unordered_map<std::string, std::chrono::steady_clock::time_point> tick_acc_ms_times;
inline std::unordered_map<std::string, double> total_acc_ms_times;

#define ACC_TICK_MS(x) tick_acc_ms_times[#x] = std::chrono::steady_clock::now();
#define ACC_TOCK_MS(x) total_acc_ms_times[#x] += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>( \
    std::chrono::steady_clock::now() - tick_acc_ms_times[#x]).count(); \
std::cout << #x " accumulated time: " << total_acc_ms_times[#x] << " ms" << std::endl;


/*
#define ACC_TOCK_MS(x) total_acc_ms_times[#x] += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>( \
                                                 std::chrono::steady_clock::now() - tick_acc_ms_times[#x]).count();
#define ACC_PRINT_MS(x) std::cout << #x " accumulated time: " << total_acc_ms_times[#x] << " ms" << std::endl;
*/

#endif
