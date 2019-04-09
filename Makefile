TARGET := benchmark
CUDA_PATH := ${CUDA_PATH}
CXX := $(CUDA_PATH)/bin/nvcc
CUDNN_PATH := ${CUDNN_PATH}
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64
CXXFLAGS := -std=c++11

root_source_dir := src
bin_dir := bin

all: conv

conv: $(root_source_dir)/$(TARGET).cu bin_dir
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(root_source_dir)/$(TARGET).cu -o $(bin_dir)/$(TARGET) \
	-lcudnn -lcurand

bin_dir:
	mkdir -p $(bin_dir)

.phony: clean

clean:
	rm -rf $(bin_dir)
