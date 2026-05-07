FLAGS= -DDEBUG
OMPFLAGS=
LIBS= -lm -lcudart
ALWAYS_REBUILD=makefile
NVCC=nvcc
CUDAFLAGS= -x cu

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) $(OMPFLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) $(OMPFLAGS) -c $< 
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) $(OMPFLAGS) $(CUDAFLAGS) -c $< -o $@
clean:
	rm -f *.o nbody 
