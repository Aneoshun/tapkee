project (minimal)

add_executable (minimal_physics minimal_physics.cpp)

if (ARPACK_FOUND)
	target_link_libraries(minimal_physics arpack)
endif()

