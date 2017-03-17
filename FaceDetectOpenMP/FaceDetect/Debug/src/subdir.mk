################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/FaceDetect.cpp \
../src/ImageUtil.cpp \
../src/NeuralNet.cpp 

OBJS += \
./src/FaceDetect.o \
./src/ImageUtil.o \
./src/NeuralNet.o 

CPP_DEPS += \
./src/FaceDetect.d \
./src/ImageUtil.d \
./src/NeuralNet.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


