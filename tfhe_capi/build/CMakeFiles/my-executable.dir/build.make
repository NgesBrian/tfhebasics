# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi/build

# Include any dependencies generated for this target.
include CMakeFiles/my-executable.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/my-executable.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/my-executable.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/my-executable.dir/flags.make

CMakeFiles/my-executable.dir/main.c.o: CMakeFiles/my-executable.dir/flags.make
CMakeFiles/my-executable.dir/main.c.o: ../main.c
CMakeFiles/my-executable.dir/main.c.o: CMakeFiles/my-executable.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/my-executable.dir/main.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/my-executable.dir/main.c.o -MF CMakeFiles/my-executable.dir/main.c.o.d -o CMakeFiles/my-executable.dir/main.c.o -c /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi/main.c

CMakeFiles/my-executable.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/my-executable.dir/main.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi/main.c > CMakeFiles/my-executable.dir/main.c.i

CMakeFiles/my-executable.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/my-executable.dir/main.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi/main.c -o CMakeFiles/my-executable.dir/main.c.s

# Object files for target my-executable
my__executable_OBJECTS = \
"CMakeFiles/my-executable.dir/main.c.o"

# External object files for target my-executable
my__executable_EXTERNAL_OBJECTS =

my-executable: CMakeFiles/my-executable.dir/main.c.o
my-executable: CMakeFiles/my-executable.dir/build.make
my-executable: /path/to/tfhe-rs/binaries/and/header/libtfhe.a
my-executable: CMakeFiles/my-executable.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable my-executable"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my-executable.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/my-executable.dir/build: my-executable
.PHONY : CMakeFiles/my-executable.dir/build

CMakeFiles/my-executable.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/my-executable.dir/cmake_clean.cmake
.PHONY : CMakeFiles/my-executable.dir/clean

CMakeFiles/my-executable.dir/depend:
	cd /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi/build /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi/build /home/local/ASURITE/nnjungle/Documents/FHE/tfhe_capi/build/CMakeFiles/my-executable.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/my-executable.dir/depend

