#/******************************************************************************
# * Makefile v.1 (7/5/01)
# *
# * Description:
# *	Makefile for artmap_c.zip
# *	'make' compiles,
# *     'make clean' removes the executable, zip AND temporary files
# *	'make zip' makes the zip of this package
# * Authors:
# *	Suhas Chelian, Norbert Kopco
#*******************************************************************************
all:	targ
SRCS = artmap.cpp\
	README\
	input.dat\
	output.dat\
	te_input.dat\
	te_output.dat\
	Makefile
OBJS = artmap\
	artmap_c.zip

targ:	artmap.cpp
	gcc artmap.cpp -o artmap -lm

zip:	$(SRCS)
	zip artmap_c $(SRCS)

clean:
	rm -f $(OBJS) *~