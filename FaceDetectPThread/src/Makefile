CC=gcc
CFLAGS=-c -Wall
LDFLAGS=-lm
SOURCES=FaceDetect.cpp ImageUtil.cpp NeuralNet.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=Facedetect

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@


