all: simple libsimple.so

simple.o: simple.ll
	llc -filetype=obj --relocation-model=pic simple.ll -o simple.o

libsimple.so: simple.o
	clang -shared -fPIC simple.o -o libsimple.so

simple: simple.o
	clang simple.o -o simple

clean:
	rm -f simple.o libsimple.so simple
