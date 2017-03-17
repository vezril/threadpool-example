################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/ImageUtil.cpp \
../src/NeuralNet.cpp \
../src/TrainNetworkPThread.cpp 

OBJS += \
./src/ImageUtil.o \
./src/NeuralNet.o \
./src/TrainNetworkPThread.o 

CPP_DEPS += \
./src/ImageUtil.d \
./src/NeuralNet.d \
./src/TrainNetworkPThread.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	mpic++ -O3 -g3 -Wall -c -fmessage-length=0 -pthread -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


