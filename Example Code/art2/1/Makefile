OBJECTS = in_vec.o rk4d.o std_fun.o art2.o
LIBS = -lm
HDRS = art2.h in_vec.h std_fun.h rk4d.h
SOURCES = art2.c in_vec.c rk4d.c std_fun.c
MISC = README Makefile art2.n
DATA = p1.inp p2.inp p3.inp p4.inp ART2.inp
SHAR = /usr/local/unsupported/bin/shar -a
SHARFILES = $(MISC) $(HDRS) $(SOURCES) $(DATA)
FLAGS =

all: art2

art2:	$(OBJECTS)
	cc $(FLAGS) -o art2 $(OBJECTS) $(LIBS)

art2.o: art2.c $(HDRS)
	cc $(FLAGS) -c art2.c -o art2.o

in_vec.o: in_vec.c in_vec.h
	cc $(FLAGS) -c in_vec.c -o in_vec.o

rk4d.o: rk4d.c rk4d.h
	cc $(FLAGS) -c -o rk4d.o rk4d.c

std_fun.o: std_fun.c std_fun.h
	cc $(FLAGS) -c -o std_fun.o std_fun.c

shar:	$(SHARFILES)
	rm -f art2.shar
	$(SHAR) $(SHARFILES) > art2.shar

clean:
	-/bin/rm *.out *.o core

clear: clean
	-/bin/rm art2
