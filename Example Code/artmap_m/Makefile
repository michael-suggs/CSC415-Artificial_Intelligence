#/******************************************************************************
# * Makefile v.1 (7/5/01)
# *
# * Description:
# *	Makefile for artmap_m.zip
# *	'make' starts Matlab
# *	'make clean' removes the zip of this package AND temporary files
# *	'make zip' makes the zip of this package
# * Authors:
# *	Suhas Chelian, Norbert Kopco
#*******************************************************************************
all:	targ
SRCS = artmap_shell.m\
	artmap_shell_vote.m\
	artmap_init.m\
	artmap_train_large.m\
	artmap_train_small.m\
	artmap_test_large.m\
	artmap_test_small.m\
	README\
	input.dat\
	output.dat\
	te_input.dat\
	te_output.dat\
	Makefile
OBJS = artmap_m.zip

targ:	artmap_shell.m
	matlab

zip:	$(SRCS)
	zip artmap_m $(SRCS)

clean:
	rm -f $(OBJS) *~