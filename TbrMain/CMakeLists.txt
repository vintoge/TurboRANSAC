file(GLOB src_inc CONFIGURE_DEPENDS src/*.cu src/*.cpp include/TbrMain/*.cuh include/TbrMain/*.h)
add_executable(TbrMain ${src_inc})
target_include_directories(TbrMain PUBLIC include)


target_link_libraries(TbrMain PUBLIC TbrCudaPnP)    # MY-LIB

########################################################################################
################################## Thired-Party-LIB ####################################
########################################################################################

# find_package(Threads REQUIRED)  
# target_link_libraries(TbrMain PUBLIC Threads::Threads)


find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(TbrMain PUBLIC Eigen3::Eigen)


find_package(nlohmann_json REQUIRED)
target_link_libraries(TbrMain PUBLIC nlohmann_json::nlohmann_json)


########################################################################################
################################## Thired-Party-LIB ####################################
########################################################################################


