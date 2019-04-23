TARGET := benchmark
CUDA_PATH := ${CUDA_PATH}
CXX := $(CUDA_PATH)/bin/nvcc
CUDNN_PATH := ${CUDNN_PATH}
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64
CXXFLAGS := -std=c++11

root_source_dir := src
bin_dir := bin
debug_dir := debug
debug_file := conv_example.txt

all: bin_ debug_

bin_: $(root_source_dir)/$(TARGET).cu bin_dir
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(root_source_dir)/$(TARGET).cu -o $(bin_dir)/$(TARGET) \
	-lcudnn -lcurand

debug_: $(root_source_dir)/$(TARGET).cu debug_file
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(root_source_dir)/$(TARGET).cu -o $(debug_dir)/$(TARGET) \
	-lcudnn -lcurand

bin_dir:
	mkdir -p $(bin_dir)

debug_file: debug_dir
	cp $(debug_file) $(debug_dir)/debug_conv.txt

debug_dir:
	mkdir -p $(debug_dir)

.phony: clean

clean:
	rm -rf $(bin_dir)
