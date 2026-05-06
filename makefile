FLAGS= -DDEBUG
OMPFLAGS=
LIBS= -lm
ALWAYS_REBUILD=makefile
NVCC=nvcc
CUDAFLAGS=

ifeq ($(USE_CUDA),1)
FLAGS += -DUSE_CUDA
CUDAFLAGS += -x cu
LIBS += -lcudart
endif

nbody: nbody.o compute.o
ifeq ($(USE_CUDA),1)
	$(NVCC) $(FLAGS) $(OMPFLAGS) $^ -o $@ $(LIBS)
else
	gcc $(FLAGS) $(OMPFLAGS) $^ -o $@ $(LIBS)
endif
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) $(OMPFLAGS) -c $< 
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
ifeq ($(USE_CUDA),1)
	$(NVCC) $(FLAGS) $(OMPFLAGS) $(CUDAFLAGS) -c $< -o $@
else
	gcc $(FLAGS) $(OMPFLAGS) -c $<
endif
clean:
	rm -f *.o nbody 
